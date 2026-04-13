from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.eplb.expert_location import (
    ExpertLocationMetadata,
    ModelConfigForExpertLocation,
    get_global_expert_location_metadata,
)
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


@dataclass
class OnlineC2RLayerPolicy:
    familiar_set: torch.Tensor
    group_ids: torch.Tensor
    group_centers: torch.Tensor
    expert_counts: torch.Tensor


class OnlineC2RManager:
    def __init__(self, server_args: ServerArgs, model_config: ModelConfig):
        config = ModelConfigForExpertLocation.from_model_config(model_config)
        self.server_args = server_args
        self.model_config = model_config
        self.model_config_for_expert_location = config
        self.enabled = (
            server_args.enable_online_c2r
            and config is not None
            and server_args.ep_size > 1
        )
        self._lock = threading.Lock()
        self._current_forward_mode = None
        self._num_forward_passes = 0
        self._policy_version = 0
        self._last_rebalanced_version = -1
        self._device_policy_cache: Dict[tuple[int, str], torch.Tensor] = {}

        if not self.enabled:
            self.num_layers = 0
            self.num_logical_experts = 0
            self.num_local_physical_experts = 0
            self.cooccurrence = []
            self.expert_counts = []
            self.layer_policy = {}
            return

        expert_location = get_global_expert_location_metadata()
        if expert_location is None:
            raise RuntimeError(
                "Online C2R requires expert location metadata to be initialized first."
            )

        self.num_layers = config.num_layers
        self.num_logical_experts = config.num_logical_experts
        self.num_local_physical_experts = expert_location.num_local_physical_experts
        self.top_t = min(server_args.c2r_top_t, max(self.num_logical_experts - 1, 1))
        self.group_max_size = (
            server_args.c2r_group_max_size or self.num_local_physical_experts
        )
        self.cooccurrence = [
            torch.zeros(
                (self.num_logical_experts, self.num_logical_experts), dtype=torch.int64
            )
            for _ in range(self.num_layers)
        ]
        self.expert_counts = [
            torch.zeros((self.num_logical_experts,), dtype=torch.int64)
            for _ in range(self.num_layers)
        ]
        self.layer_policy: Dict[int, OnlineC2RLayerPolicy] = {}

    @classmethod
    def init_new(cls, server_args: ServerArgs, model_config: ModelConfig):
        return cls(server_args, model_config)

    def set_current_forward_mode(self, forward_mode) -> None:
        self._current_forward_mode = forward_mode

    def clear_current_forward_mode(self) -> None:
        self._current_forward_mode = None

    def _mode_enabled(self) -> bool:
        if not self.enabled:
            return False
        if self._current_forward_mode is None:
            return True
        if (
            hasattr(self._current_forward_mode, "is_prefill")
            and self._current_forward_mode.is_prefill()
        ):
            return self.server_args.c2r_enable_prefill
        if (
            hasattr(self._current_forward_mode, "is_decode")
            and self._current_forward_mode.is_decode()
        ):
            return self.server_args.c2r_enable_decode
        return True

    def record_topk(self, layer_id: Optional[int], logical_topk_ids: torch.Tensor) -> None:
        if not self.enabled or layer_id is None or not self._mode_enabled():
            return

        topk_ids = logical_topk_ids.detach().to(device="cpu", dtype=torch.int64)
        topk_ids = topk_ids[(topk_ids >= 0) & (topk_ids < self.num_logical_experts)]
        if topk_ids.numel() == 0:
            return

        topk_ids = logical_topk_ids.detach().to(device="cpu", dtype=torch.int64)
        valid_mask = (topk_ids >= 0) & (topk_ids < self.num_logical_experts)

        with self._lock:
            valid_flat = topk_ids[valid_mask]
            self.expert_counts[layer_id].index_add_(
                0,
                valid_flat,
                torch.ones_like(valid_flat, dtype=torch.int64),
            )

            num_choices = topk_ids.shape[1]
            for left in range(num_choices):
                lhs = topk_ids[:, left]
                lhs_valid = (lhs >= 0) & (lhs < self.num_logical_experts)
                for right in range(left + 1, num_choices):
                    rhs = topk_ids[:, right]
                    mask = lhs_valid & (rhs >= 0) & (rhs < self.num_logical_experts)
                    if not mask.any():
                        continue
                    lhs_masked = lhs[mask]
                    rhs_masked = rhs[mask]
                    indices = lhs_masked * self.num_logical_experts + rhs_masked
                    updates = torch.bincount(
                        indices,
                        minlength=self.num_logical_experts * self.num_logical_experts,
                    ).view(self.num_logical_experts, self.num_logical_experts)
                    self.cooccurrence[layer_id].add_(updates)
                    self.cooccurrence[layer_id].add_(updates.T)

    def maybe_apply_routing_constraints(
        self,
        layer_id: Optional[int],
        router_logits: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        *,
        use_grouped_topk: bool,
        num_expert_group: Optional[int],
        num_fused_shared_experts: int,
        renormalize: bool,
        scoring_func: str,
        correction_bias: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            not self.enabled
            or layer_id is None
            or not self._mode_enabled()
            or self._num_forward_passes < self.server_args.c2r_warmup_num_iterations
        ):
            return topk_ids, topk_weights

        policy = self.layer_policy.get(layer_id)
        if policy is None:
            return topk_ids, topk_weights

        routed_topk = topk_ids.shape[1] - num_fused_shared_experts
        if routed_topk <= 1:
            return topk_ids, topk_weights

        familiar_set = self._get_familiar_set_on_device(layer_id, router_logits.device)
        scores = _compute_routing_scores(router_logits, scoring_func=scoring_func)
        choice_scores = scores
        if correction_bias is not None:
            choice_scores = choice_scores + correction_bias.unsqueeze(0).to(
                dtype=choice_scores.dtype, device=choice_scores.device
            )

        constrained_ids = topk_ids[:, :routed_topk].clone()
        constrained_weights = topk_weights[:, :routed_topk].clone()
        top1 = constrained_ids[:, 0].clamp(min=0)

        allowed_mask = torch.zeros_like(choice_scores, dtype=torch.bool)
        candidates = familiar_set[top1]
        valid_candidates = candidates >= 0
        if valid_candidates.any():
            allowed_mask.scatter_(1, candidates.clamp(min=0), valid_candidates)

        if use_grouped_topk and num_expert_group:
            experts_per_group = max(choice_scores.shape[1] // num_expert_group, 1)
            group_mask = torch.zeros(
                (constrained_ids.shape[0], num_expert_group),
                dtype=torch.bool,
                device=constrained_ids.device,
            )
            chosen_groups = (
                constrained_ids.clamp(min=0, max=choice_scores.shape[1] - 1)
                // experts_per_group
            ).clamp(max=num_expert_group - 1)
            group_mask.scatter_(1, chosen_groups, True)
            expanded_group_mask = (
                group_mask.unsqueeze(-1)
                .expand(-1, -1, experts_per_group)
                .reshape(constrained_ids.shape[0], -1)
            )
            allowed_mask &= expanded_group_mask[:, : choice_scores.shape[1]]

        already_chosen = torch.zeros_like(allowed_mask)
        already_chosen.scatter_(1, constrained_ids[:, :1].clamp(min=0), True)

        constrained_slots = [1] if self.server_args.c2r_route_top2_only else list(
            range(1, routed_topk)
        )

        for slot in constrained_slots:
            slot_allowed = allowed_mask & (~already_chosen)
            no_candidate = ~slot_allowed.any(dim=1)
            candidate_scores = choice_scores.masked_fill(~slot_allowed, float("-inf"))
            candidate_ids = candidate_scores.argmax(dim=1)
            selected_scores = scores.gather(1, candidate_ids.unsqueeze(1)).squeeze(1)

            fallback = no_candidate
            if self.server_args.c2r_fallback_threshold > 0:
                fallback |= selected_scores < self.server_args.c2r_fallback_threshold

            if (~fallback).any():
                write_mask = ~fallback
                constrained_ids[write_mask, slot] = candidate_ids[write_mask].to(
                    dtype=constrained_ids.dtype
                )
                constrained_weights[write_mask, slot] = selected_scores[
                    write_mask
                ].to(dtype=constrained_weights.dtype)

            already_chosen.scatter_(
                1, constrained_ids[:, slot : slot + 1].clamp(min=0), True
            )

        constrained_weights = scores.gather(
            1, constrained_ids.clamp(min=0, max=choice_scores.shape[1] - 1)
        ).to(dtype=topk_weights.dtype)
        if renormalize:
            denom = constrained_weights.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
            constrained_weights = constrained_weights / denom

        topk_ids[:, :routed_topk] = constrained_ids
        topk_weights[:, :routed_topk] = constrained_weights
        return topk_ids, topk_weights

    def on_forward_pass_end(self, model_runner) -> None:
        if not self.enabled:
            return

        self._num_forward_passes += 1
        if (
            self._num_forward_passes >= self.server_args.c2r_warmup_num_iterations
            and self._num_forward_passes
            % self.server_args.c2r_policy_update_num_iterations
            == 0
        ):
            self._refresh_policy()

        if (
            self._policy_version > self._last_rebalanced_version
            and self._num_forward_passes >= self.server_args.c2r_warmup_num_iterations
            and self._num_forward_passes % self.server_args.c2r_rebalance_num_iterations
            == 0
        ):
            expert_location_metadata = self._build_rebalance_metadata()
            if expert_location_metadata is not None:
                update_layer_ids = sorted(
                    list(model_runner.model.routed_experts_weights_of_layer.keys())
                )
                model_runner.update_expert_location(
                    expert_location_metadata, update_layer_ids=update_layer_ids
                )
                self._last_rebalanced_version = self._policy_version

    def _refresh_policy(self) -> None:
        updated = False
        new_policies: Dict[int, OnlineC2RLayerPolicy] = {}
        with self._lock:
            for layer_id in range(self.num_layers):
                counts = self.expert_counts[layer_id].clone()
                if counts.sum().item() < self.server_args.c2r_min_layer_coverage:
                    continue
                cooccurrence = self.cooccurrence[layer_id].clone()
                familiar_set = _build_familiar_set(
                    cooccurrence=cooccurrence,
                    expert_counts=counts,
                    top_t=self.top_t,
                    min_expert_samples=self.server_args.c2r_min_expert_samples,
                )
                group_ids = _build_group_ids(
                    cooccurrence=cooccurrence,
                    expert_counts=counts,
                    max_group_size=self.group_max_size,
                )
                group_centers = _build_group_centers(
                    group_ids=group_ids,
                    cooccurrence=cooccurrence,
                    expert_counts=counts,
                )
                new_policies[layer_id] = OnlineC2RLayerPolicy(
                    familiar_set=familiar_set,
                    group_ids=group_ids,
                    group_centers=group_centers,
                    expert_counts=counts,
                )
                updated = True

        if updated:
            self.layer_policy.update(new_policies)
            self._policy_version += 1
            self._device_policy_cache.clear()
            logger.info(
                "[OnlineC2R] Refreshed policy version=%s active_layers=%s",
                self._policy_version,
                sorted(new_policies.keys()),
            )

    def _build_rebalance_metadata(self) -> Optional[ExpertLocationMetadata]:
        if not self.layer_policy:
            return None

        current_metadata = get_global_expert_location_metadata()
        if current_metadata is None:
            return None

        physical_to_logical_map = []
        for layer_id in range(self.num_layers):
            policy = self.layer_policy.get(layer_id)
            if policy is None:
                physical_to_logical_map.append(
                    current_metadata.physical_to_logical_map_cpu[layer_id].tolist()
                )
                continue
            physical_to_logical_map.append(
                _build_layer_physical_to_logical_map(
                    group_ids=policy.group_ids,
                    group_centers=policy.group_centers,
                    expert_counts=policy.expert_counts,
                    num_physical_experts=current_metadata.num_physical_experts,
                    ep_size=current_metadata.ep_size,
                    num_local_physical_experts=current_metadata.num_local_physical_experts,
                )
            )

        new_mapping = torch.tensor(physical_to_logical_map, dtype=torch.int64)
        if torch.equal(new_mapping, current_metadata.physical_to_logical_map_cpu):
            return None

        return ExpertLocationMetadata.init_by_mapping(
            self.server_args,
            self.model_config,
            physical_to_logical_map=new_mapping,
        )

    def _get_familiar_set_on_device(self, layer_id: int, device: torch.device) -> torch.Tensor:
        key = (layer_id, str(device))
        cached = self._device_policy_cache.get(key)
        if cached is None:
            policy = self.layer_policy[layer_id]
            cached = policy.familiar_set.to(device=device, non_blocking=True)
            self._device_policy_cache[key] = cached
        return cached


def _compute_routing_scores(
    router_logits: torch.Tensor,
    *,
    scoring_func: str,
) -> torch.Tensor:
    if scoring_func == "softmax":
        return router_logits.float().softmax(dim=-1)
    if scoring_func == "sigmoid":
        return router_logits.float().sigmoid()
    raise ValueError(f"Unsupported scoring_func={scoring_func}")


def _build_familiar_set(
    cooccurrence: torch.Tensor,
    expert_counts: torch.Tensor,
    top_t: int,
    min_expert_samples: int,
) -> torch.Tensor:
    num_experts = cooccurrence.shape[0]
    familiar_set = torch.full((num_experts, top_t), -1, dtype=torch.int64)
    if top_t == 0:
        return familiar_set

    for expert_id in range(num_experts):
        if expert_counts[expert_id].item() < min_expert_samples:
            continue
        scores = cooccurrence[expert_id].clone()
        scores[expert_id] = -1
        positive = torch.nonzero(scores > 0, as_tuple=False).flatten()
        if positive.numel() == 0:
            continue
        k = min(top_t, positive.numel())
        top_indices = torch.topk(scores, k=k).indices
        familiar_set[expert_id, :k] = top_indices
    return familiar_set


def _build_group_ids(
    cooccurrence: torch.Tensor,
    expert_counts: torch.Tensor,
    max_group_size: int,
) -> torch.Tensor:
    num_experts = cooccurrence.shape[0]
    group_ids = torch.full((num_experts,), -1, dtype=torch.int32)
    if num_experts == 0:
        return group_ids

    sorted_experts = torch.argsort(expert_counts, descending=True)
    group_id = 0
    for seed in sorted_experts.tolist():
        if group_ids[seed] != -1:
            continue
        group_ids[seed] = group_id
        if max_group_size > 1:
            scores = cooccurrence[seed].clone()
            scores[group_ids != -1] = -1
            scores[seed] = -1
            num_take = min(max_group_size - 1, int((scores > 0).sum().item()))
            if num_take > 0:
                neighbors = torch.topk(scores, k=num_take).indices
                group_ids[neighbors] = group_id
        group_id += 1

    unassigned = torch.nonzero(group_ids == -1, as_tuple=False).flatten()
    if unassigned.numel() > 0:
        group_ids[unassigned] = torch.arange(
            group_id, group_id + unassigned.numel(), dtype=group_ids.dtype
        )
    return group_ids


def _build_group_centers(
    group_ids: torch.Tensor,
    cooccurrence: torch.Tensor,
    expert_counts: torch.Tensor,
) -> torch.Tensor:
    num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0
    centers = torch.full((num_groups,), -1, dtype=torch.int64)
    for group_id in range(num_groups):
        members = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
        if members.numel() == 0:
            continue
        member_scores = expert_counts[members] + cooccurrence[members][:, members].sum(
            dim=1
        )
        centers[group_id] = members[member_scores.argmax()].item()
    return centers


def _build_layer_physical_to_logical_map(
    *,
    group_ids: torch.Tensor,
    group_centers: torch.Tensor,
    expert_counts: torch.Tensor,
    num_physical_experts: int,
    ep_size: int,
    num_local_physical_experts: int,
) -> List[int]:
    num_logical_experts = expert_counts.shape[0]
    gpu_slots: List[List[int]] = [[] for _ in range(ep_size)]
    group_order = []
    num_groups = int(group_ids.max().item()) + 1 if group_ids.numel() > 0 else 0
    for group_id in range(num_groups):
        members = torch.nonzero(group_ids == group_id, as_tuple=False).flatten()
        if members.numel() == 0:
            continue
        group_score = int(expert_counts[members].sum().item())
        group_order.append((group_score, members.tolist()))
    group_order.sort(key=lambda item: item[0], reverse=True)

    for _, members in group_order:
        members = sorted(members, key=lambda expert: int(expert_counts[expert]), reverse=True)
        while members:
            target_gpu = max(
                range(ep_size),
                key=lambda rank: (
                    num_local_physical_experts - len(gpu_slots[rank]),
                    -len(gpu_slots[rank]),
                ),
            )
            remaining = num_local_physical_experts - len(gpu_slots[target_gpu])
            if remaining <= 0:
                target_gpu = min(range(ep_size), key=lambda rank: len(gpu_slots[rank]))
                remaining = max(num_local_physical_experts - len(gpu_slots[target_gpu]), 0)
            take = max(1, min(len(members), remaining or 1))
            gpu_slots[target_gpu].extend(members[:take])
            members = members[take:]

    # Ensure every expert appears at least once even if the grouping logic missed any.
    placed = {expert for slots in gpu_slots for expert in slots}
    for expert in range(num_logical_experts):
        if expert in placed:
            continue
        target_gpu = min(range(ep_size), key=lambda rank: len(gpu_slots[rank]))
        gpu_slots[target_gpu].append(expert)

    # Fill redundant expert capacity with group centers first, then hottest experts.
    while sum(len(slots) for slots in gpu_slots) < num_physical_experts:
        for center in group_centers.tolist():
            if center < 0:
                continue
            target_gpu = min(range(ep_size), key=lambda rank: len(gpu_slots[rank]))
            if len(gpu_slots[target_gpu]) < num_local_physical_experts:
                gpu_slots[target_gpu].append(center)
            if sum(len(slots) for slots in gpu_slots) >= num_physical_experts:
                break
        else:
            hottest = torch.argsort(expert_counts, descending=True).tolist()
            for expert in hottest:
                target_gpu = min(range(ep_size), key=lambda rank: len(gpu_slots[rank]))
                if len(gpu_slots[target_gpu]) < num_local_physical_experts:
                    gpu_slots[target_gpu].append(expert)
                if sum(len(slots) for slots in gpu_slots) >= num_physical_experts:
                    break

    physical_to_logical_map: List[int] = []
    for rank in range(ep_size):
        slots = gpu_slots[rank][:num_local_physical_experts]
        if len(slots) < num_local_physical_experts:
            hottest = torch.argsort(expert_counts, descending=True).tolist()
            for expert in hottest:
                if len(slots) >= num_local_physical_experts:
                    break
                slots.append(expert)
        physical_to_logical_map.extend(slots[:num_local_physical_experts])

    return physical_to_logical_map[:num_physical_experts]


_global_online_c2r_manager: Optional[OnlineC2RManager] = None


def get_global_online_c2r_manager() -> Optional[OnlineC2RManager]:
    return _global_online_c2r_manager


def set_global_online_c2r_manager(value: Optional[OnlineC2RManager]) -> None:
    global _global_online_c2r_manager
    _global_online_c2r_manager = value
