"""Mixed prefill/speculative-decode orchestration for EAGLE V2.

This module intentionally owns no model, graph runner, or KV pool.  It receives
the resources owned by :class:`EAGLEWorkerV2` and coordinates one mixed turn.
Keeping the orchestration isolated makes the target and draft composition paths
independently testable without introducing a second speculative worker.
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass, replace
from typing import Any, Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.logits_processor import (
    split_composition_logits_output,
    split_draft_extend_composition_output,
)
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardCompositionTensorScratch,
    pack_draft_prefill_and_decode_extend_forward,
    pack_prefill_and_verify_forward,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.speculative.eagle_info import EagleVerifyInput
from sglang.srt.speculative.eagle_utils import eagle_prepare_for_verify
from sglang.srt.speculative.eagle_worker_common import run_eagle_verify
from sglang.srt.speculative.parity import (
    AttentionTrace,
    KVRows,
    OperatorTrace,
    attention_parity,
    install_operator_trace_hooks,
    logits_parity,
    operator_parity,
    operator_parity_enabled,
    parity_max_steps,
    parity_output_dir,
    record_attention,
    record_operators,
    remove_operator_trace_hooks,
    write_parity_report,
)
from sglang.srt.speculative.spec_utils import spec_stage_span
from sglang.srt.runtime_context import get_spec

logger = logging.getLogger(__name__)

# The packed path pays a fixed metadata/dispatch cost. 4090 A/B measurements
# show that prefix-hit batches with only a handful of new tokens do not amortize
# it, while P >= 256 is the first design target shape with a stable benefit.
MIN_FUSED_PREFILL_TOKENS = 256


@dataclass(slots=True)
class PreparedDraftPrefillSegment:
    forward_batch: ForwardBatch
    batch: ScheduleBatch
    next_token_ids: torch.Tensor
    seed_from_extend: bool
    num_requests: int


@dataclass(slots=True)
class PreparedDraftDecodeExtendSegment:
    forward_batch: ForwardBatch
    batch: ScheduleBatch
    batch_result: GenerationBatchResult
    select_index: torch.Tensor
    can_run_cuda_graph: bool
    seed_from_extend: bool
    num_requests: int


class EagleMixedWorkerV2:
    """Coordinate one mixed EAGLE iteration using owner-provided resources."""

    def __init__(
        self,
        *,
        target_worker: Any,
        draft_worker: Any,
        adaptive_controller: Any,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
        device: str,
        topk: int,
        speculative_num_steps: int,
        speculative_num_draft_tokens: int,
        plan_stream: Any,
        plan_stream_ctx: Any,
    ) -> None:
        self.target_worker = target_worker
        self.draft_worker = draft_worker
        self.adaptive_controller = adaptive_controller
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator
        self.device = device
        self.topk = topk
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = speculative_num_draft_tokens
        self.plan_stream = plan_stream
        self.plan_stream_ctx = plan_stream_ctx

        # Overlap retains two forward generations. Alternate grow-only arenas
        # so the next pack cannot overwrite tensors from the previous target.
        self._target_pack_scratch_slots = [
            ForwardCompositionTensorScratch(),
            ForwardCompositionTensorScratch(),
        ]
        self._target_pack_scratch_cursor = 0
        self._draft_pack_scratch_slots = [
            ForwardCompositionTensorScratch(),
            ForwardCompositionTensorScratch(),
        ]
        self._draft_pack_scratch_cursor = 0
        self._parity_index = 0
        self._last_target_composition_graph = False
        self._logged_fusion_decisions: set[bool] = set()

    def bind_memory_pools(
        self,
        *,
        req_to_token_pool: Any,
        token_to_kv_pool_allocator: Any,
    ) -> None:
        """Bind pools allocated after the speculative worker is constructed."""
        self.req_to_token_pool = req_to_token_pool
        self.token_to_kv_pool_allocator = token_to_kv_pool_allocator

    def _activate_step_by_batch(self, batch_size: int) -> None:
        if self.adaptive_controller is not None:
            self.adaptive_controller.activate_step_by_batch(batch_size)

    def forward_batch_generation(
        self,
        batch: ScheduleBatch,
        on_publish=None,
        grammar_barrier=None,
    ) -> GenerationBatchResult:
        """Run the existing packed-target mixed path.

        This extraction is deliberately behavior preserving.  Draft composition
        and fused attention are added in later changes behind explicit gates.
        """
        prefill_batch = batch.spec_mixed_prefill_batch
        verify_batch = batch.spec_mixed_verify_batch
        if prefill_batch is None or verify_batch is None:
            raise ValueError("Speculative mixed batch requires both source views")
        if (
            self.req_to_token_pool is None
            or self.token_to_kv_pool_allocator is None
        ):
            raise RuntimeError("EAGLE mixed worker memory pools are not initialized")
        if self.topk != 1:
            raise NotImplementedError(
                "Speculative mixed batching currently requires topk=1"
            )
        target_backend = self.target_worker.model_runner.attn_backend
        if not target_backend.supports_forward_composition(
            "prefill_spec_verify",
            topk=self.topk,
            fixed_q_len=self.speculative_num_draft_tokens,
        ):
            raise NotImplementedError(
                "The active target attention backend cannot run spec mixed composition"
            )

        prefill_forward_batch = ForwardBatch.init_new(
            prefill_batch,
            self.target_worker.model_runner,
            capture_hidden_mode=CaptureHiddenMode.FULL,
            return_hidden_states_before_norm=False,
        )

        self._activate_step_by_batch(verify_batch.seq_lens.shape[0])
        if verify_batch.spec_info is None:
            raise RuntimeError("Speculative mixed verify view has no draft state")
        with (
            self.draft_worker.draft_tp_context(self.draft_worker.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft"),
        ):
            verify_input: EagleVerifyInput = self.draft_worker.draft(verify_batch)
        verify_batch.spec_info = verify_input

        with self.plan_stream_ctx:
            verify_forward_batch, _ = eagle_prepare_for_verify(
                verify_input,
                self.req_to_token_pool,
                verify_batch,
                self.target_worker,
                allow_cuda_graph=False,
            )
        if self.plan_stream:
            torch.get_device_module(self.device).current_stream().wait_stream(
                self.plan_stream
            )

        pack_scratch = self._target_pack_scratch_slots[
            self._target_pack_scratch_cursor
        ]
        self._target_pack_scratch_cursor = (self._target_pack_scratch_cursor + 1) % len(
            self._target_pack_scratch_slots
        )
        packed_forward_batch = pack_prefill_and_verify_forward(
            prefill_forward_batch, verify_forward_batch, scratch=pack_scratch
        )
        draft_backend = self.draft_worker.draft_runner.attn_backend
        fusion_checks = {
            "enabled": envs.SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION.get(),
            "profitable_shape": (
                prefill_forward_batch.input_ids.shape[0]
                >= MIN_FUSED_PREFILL_TOKENS
            ),
            "no_rejection_sampling": not get_spec().speculative_use_rejection_sampling,
            "no_dsa_extend_seed": not self.draft_worker.seed_dsa_topk_from_draft_extend,
            "target_backend": target_backend.supports_fused_forward_composition(
                "prefill_spec_verify",
                topk=self.topk,
                fixed_q_len=self.speculative_num_draft_tokens,
            ),
            "draft_backend": draft_backend.supports_fused_forward_composition(
                "draft_prefill_decode_extend",
                topk=self.topk,
                fixed_q_len=self.speculative_num_draft_tokens,
            ),
        }
        fused_composition = all(fusion_checks.values())
        if fused_composition not in self._logged_fusion_decisions:
            logger.info(
                "EAGLE mixed fused attention %s: %s",
                "enabled" if fused_composition else "disabled",
                fusion_checks,
            )
            self._logged_fusion_decisions.add(fused_composition)
        packed_forward_batch.composition.fused_attention = fused_composition
        parity_dir = parity_output_dir()
        parity_index = self._parity_index
        parity_enabled = parity_dir is not None and parity_index < parity_max_steps()
        operator_enabled = parity_enabled and operator_parity_enabled()
        operator_handles = (
            install_operator_trace_hooks(self.target_worker.model_runner.model)
            if operator_enabled
            else []
        )
        try:
            parity_state = None
            if parity_enabled:
                parity_state = self._run_target_reference(
                    prefill_forward_batch,
                    verify_forward_batch,
                    operator_enabled=operator_enabled,
                )

            candidate_attention = AttentionTrace() if parity_enabled else None
            candidate_operator = OperatorTrace() if operator_enabled else None
            candidate_trace_ctx = (
                record_attention(candidate_attention)
                if candidate_attention is not None
                else contextlib.nullcontext()
            )
            candidate_operator_ctx = (
                record_operators(
                    candidate_operator, prefill_forward_batch.input_ids.shape[0]
                )
                if candidate_operator is not None
                else contextlib.nullcontext()
            )
            with candidate_trace_ctx, candidate_operator_ctx:
                packed_target_result = self.target_worker.forward_batch_generation(
                    batch=None,
                    forward_batch=packed_forward_batch,
                    is_verify=True,
                )
            self._last_target_composition_graph = (
                packed_target_result.can_run_cuda_graph
            )

            if parity_state is not None:
                self._finish_target_parity(
                    parity_dir,
                    parity_index,
                    parity_state,
                    candidate_attention,
                    candidate_operator,
                    packed_target_result.logits_output.next_token_logits,
                    prefill_forward_batch,
                    verify_forward_batch,
                )
                self._parity_index = parity_index + 1
        finally:
            remove_operator_trace_hooks(operator_handles)

        prefill_logits, verify_logits = split_composition_logits_output(
            packed_target_result.logits_output,
            prefill_requests=prefill_forward_batch.batch_size,
            prefill_tokens=prefill_forward_batch.input_ids.shape[0],
            verify_tokens=verify_forward_batch.input_ids.shape[0],
        )

        prefill_next_token_ids = self.target_worker.model_runner.sample(
            prefill_logits, prefill_forward_batch
        )
        prefill_result = GenerationBatchResult(
            logits_output=prefill_logits,
            next_token_ids=prefill_next_token_ids,
            new_seq_lens=prefill_batch.seq_lens,
        )
        prefill_result.can_run_cuda_graph = packed_target_result.can_run_cuda_graph

        verify_target_result = GenerationBatchResult(logits_output=verify_logits)
        verify_result = run_eagle_verify(
            verify_batch,
            target_worker=self.target_worker,
            req_to_token_pool=self.req_to_token_pool,
            token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
            plan_stream=self.plan_stream,
            plan_stream_ctx=self.plan_stream_ctx,
            topk=self.topk,
            num_steps=self.speculative_num_steps,
            num_draft_tokens=self.speculative_num_draft_tokens,
            device=self.device,
            metadata_ready_pre_pad=False,
            finalize_tree_path=True,
            grammar_barrier=grammar_barrier,
            prepared_verify_forward_batch=verify_forward_batch,
            precomputed_forward_batch_output=verify_target_result,
        )
        verify_result.can_run_cuda_graph = packed_target_result.can_run_cuda_graph

        combined_seq_lens = torch.cat(
            (prefill_result.new_seq_lens, verify_result.new_seq_lens)
        )
        if on_publish is not None:
            on_publish(combined_seq_lens)

        with (
            self.draft_worker.draft_tp_context(self.draft_worker.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            spec_stage_span("draft_extend"),
        ):
            prepared_prefill = self.draft_worker.prepare_draft_prefill_segment(
                prefill_batch,
                prefill_logits.hidden_states,
                prefill_next_token_ids,
                prefill_logits.mm_input_embeds,
            )
            if fused_composition:
                prepared_decode = (
                    self.draft_worker.prepare_draft_decode_extend_segment(
                        verify_batch, verify_result
                    )
                )
                draft_scratch = self._draft_pack_scratch_slots[
                    self._draft_pack_scratch_cursor
                ]
                self._draft_pack_scratch_cursor = (
                    self._draft_pack_scratch_cursor + 1
                ) % len(self._draft_pack_scratch_slots)
                packed_draft_batch = pack_draft_prefill_and_decode_extend_forward(
                    prepared_prefill.forward_batch,
                    prepared_decode.forward_batch,
                    prepared_decode.select_index,
                    scratch=draft_scratch,
                )
                packed_draft_output = self.draft_worker.draft_runner.forward(
                    packed_draft_batch
                ).logits_output
                prefill_draft_output, decode_draft_output = (
                    split_draft_extend_composition_output(
                        packed_draft_output,
                        prefill_requests=prepared_prefill.num_requests,
                        decode_requests=prepared_decode.num_requests,
                    )
                )
                prefill_next_draft = (
                    self.draft_worker.finalize_draft_prefill_segment(
                        prepared_prefill, prefill_draft_output
                    )
                )
                self.draft_worker.finalize_draft_decode_extend_segment(
                    prepared_decode,
                    decode_draft_output,
                    output_layout="selected_per_request",
                )
            else:
                prefill_draft_output = (
                    self.draft_worker.run_prepared_draft_prefill_segment(
                        prepared_prefill
                    )
                )
                prefill_next_draft = (
                    self.draft_worker.finalize_draft_prefill_segment(
                        prepared_prefill, prefill_draft_output
                    )
                )
                # Preserve the legacy dependency in fallback mode: decode
                # draft-extend planning observes draft-prefill's completed KV
                # writes and pool state.
                prepared_decode = (
                    self.draft_worker.prepare_draft_decode_extend_segment(
                        verify_batch, verify_result
                    )
                )
                decode_draft_output = (
                    self.draft_worker.run_prepared_draft_decode_extend_segment(
                        prepared_decode
                    )
                )
                self.draft_worker.finalize_draft_decode_extend_segment(
                    prepared_decode,
                    decode_draft_output,
                    output_layout="full_window",
                )

        verify_next_draft = verify_result.next_draft_input
        merged_next_draft = replace(prefill_next_draft)
        merged_next_draft.merge_batch(verify_next_draft)
        for source_batch, next_draft in (
            (prefill_batch, prefill_next_draft),
            (verify_batch, verify_next_draft),
        ):
            next_draft.future_indices = source_batch.req_pool_indices
            next_draft.future_dsa_topk_indices_available = (
                next_draft.dsa_topk_indices is not None
            )
            source_batch.spec_info = next_draft
        return GenerationBatchResult(
            logits_output=packed_target_result.logits_output,
            next_draft_input=merged_next_draft,
            new_seq_lens=combined_seq_lens,
            spec_mixed_prefill_result=prefill_result,
            spec_mixed_verify_result=verify_result,
            routed_experts_output=packed_target_result.routed_experts_output,
            indexer_topk_output=packed_target_result.indexer_topk_output,
            extra_keep_alive_refs=[
                packed_forward_batch,
                prefill_forward_batch,
                verify_forward_batch,
                *([packed_draft_batch] if fused_composition else []),
            ],
            can_run_cuda_graph=packed_target_result.can_run_cuda_graph,
        )

    def _target_eager_forward(self, forward_batch: ForwardBatch):
        runner = self.target_worker.model_runner
        with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            return runner.eager_runner.execute(forward_batch)

    def _run_target_reference(
        self,
        prefill_forward_batch: ForwardBatch,
        verify_forward_batch: ForwardBatch,
        operator_enabled: bool = False,
    ) -> dict:
        runner = self.target_worker.model_runner
        pool = runner.token_to_kv_pool
        layer_ids = [
            layer.layer_id for layer in runner.attention_layers if layer is not None
        ]
        locations = torch.cat(
            (prefill_forward_batch.out_cache_loc, verify_forward_batch.out_cache_loc)
        )
        initial_kv = KVRows.capture(pool, layer_ids, locations)
        reference_attention = AttentionTrace()
        reference_operator = OperatorTrace() if operator_enabled else None
        with record_attention(reference_attention):
            prefill_output = self._target_eager_forward(prefill_forward_batch)
            prefill_logits = prefill_output.next_token_logits.detach().clone()
            operator_ctx = (
                record_operators(reference_operator, 0)
                if reference_operator is not None
                else contextlib.nullcontext()
            )
            with operator_ctx:
                verify_output = self._target_eager_forward(verify_forward_batch)
            verify_logits = verify_output.next_token_logits.detach().clone()

        reference_kv = KVRows.capture(pool, layer_ids, locations)
        initial_kv.restore(pool)
        return {
            "attention": reference_attention,
            "operator": reference_operator,
            "kv": reference_kv,
            "logits": torch.cat((prefill_logits, verify_logits), dim=0),
        }

    def _finish_target_parity(
        self,
        output_dir,
        parity_index: int,
        reference: dict,
        candidate_attention: AttentionTrace,
        candidate_operator: Optional[OperatorTrace],
        candidate_logits: torch.Tensor,
        prefill_forward_batch: ForwardBatch,
        verify_forward_batch: ForwardBatch,
    ) -> None:
        runner = self.target_worker.model_runner
        locations = torch.cat(
            (prefill_forward_batch.out_cache_loc, verify_forward_batch.out_cache_loc)
        )
        layer_ids = [
            layer.layer_id for layer in runner.attention_layers if layer is not None
        ]
        candidate_kv = KVRows.capture(runner.token_to_kv_pool, layer_ids, locations)
        prefill_locations = prefill_forward_batch.out_cache_loc
        verify_locations = verify_forward_batch.out_cache_loc
        overlap_locations = prefill_locations[torch.isin(
            prefill_locations, verify_locations
        )]

        def count_history_overlaps(
            child: ForwardBatch,
            per_request_widths: list[int],
            history_lens: torch.Tensor,
        ) -> tuple[int, Optional[int]]:
            total = 0
            first = None
            offset = 0
            for request_index, width in enumerate(per_request_widths):
                width = int(width)
                writes = child.out_cache_loc[offset : offset + width]
                offset += width
                history_len = int(history_lens[request_index])
                history = runner.req_to_token_pool.req_to_token[
                    child.req_pool_indices[request_index], :history_len
                ]
                collisions = writes[torch.isin(writes, history)]
                if collisions.numel() > 0 and first is None:
                    first = int(collisions[0])
                total += int(collisions.numel())
            return total, first

        prefill_history_overlap = count_history_overlaps(
            prefill_forward_batch,
            list(prefill_forward_batch.extend_seq_lens_cpu),
            prefill_forward_batch.extend_prefix_lens,
        )
        verify_width = verify_forward_batch.input_ids.shape[0] // max(
            verify_forward_batch.batch_size, 1
        )
        verify_history_overlap = count_history_overlaps(
            verify_forward_batch,
            [verify_width] * verify_forward_batch.batch_size,
            verify_forward_batch.seq_lens,
        )
        logits_report = logits_parity(reference["logits"], candidate_logits)
        first_row = logits_report["first_divergent_row"]
        prefill_rows = prefill_forward_batch.batch_size
        if first_row is not None:
            if first_row < prefill_rows:
                row_location = {"segment": "prefill", "request_index": first_row}
            else:
                verify_row = first_row - prefill_rows
                row_location = {
                    "segment": "verify",
                    "request_index": verify_row // verify_width,
                    "draft_index": verify_row % verify_width,
                }
            logits_report["first_divergence"]["location"] = row_location

        report = {
            "device": torch.cuda.get_device_name(runner.device),
            "dtype": str(runner.dtype),
            "prefill_requests": prefill_forward_batch.batch_size,
            "prefill_tokens": prefill_forward_batch.input_ids.shape[0],
            "verify_requests": verify_forward_batch.batch_size,
            "verify_tokens": verify_forward_batch.input_ids.shape[0],
            "candidate_used_cuda_graph": bool(
                self._last_target_composition_graph
            ),
            "out_cache_loc": {
                "prefill_unique": int(torch.unique(prefill_locations).numel()),
                "verify_unique": int(torch.unique(verify_locations).numel()),
                "cross_segment_overlap_count": int(overlap_locations.numel()),
                "first_cross_segment_overlap": (
                    int(overlap_locations[0])
                    if overlap_locations.numel() > 0
                    else None
                ),
                "prefill_history_overlap_count": prefill_history_overlap[0],
                "first_prefill_history_overlap": prefill_history_overlap[1],
                "verify_history_overlap_count": verify_history_overlap[0],
                "first_verify_history_overlap": verify_history_overlap[1],
            },
            "logits": logits_report,
            "attention": attention_parity(
                reference["attention"], candidate_attention
            ),
            "kv": reference["kv"].compare(candidate_kv),
        }
        if reference["operator"] is not None:
            report["operator"] = operator_parity(
                reference["operator"], candidate_operator
            )
        path = write_parity_report(output_dir, parity_index, report)
        logger.info("Wrote speculative mixed GPU parity report to %s", path)
