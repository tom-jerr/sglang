"""Numerical diagnostics for heterogeneous speculative target forwards.

The helpers in this module are intentionally opt-in.  A parity run keeps one
reference copy of the touched KV rows and attention outputs, so enabling it on
a serving process is expensive.  Normal inference does not allocate, clone, or
synchronize any of these tensors.
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch


@dataclass(frozen=True)
class TensorError:
    shape: list[int]
    max_abs: float
    mean_abs: float
    max_rel: float
    cosine_error: float


def tensor_error(reference: torch.Tensor, actual: torch.Tensor) -> TensorError:
    """Return stable fp32 error statistics for two same-shaped GPU tensors."""
    if reference.shape != actual.shape:
        raise ValueError(f"Tensor shapes differ: {reference.shape} != {actual.shape}")
    ref = reference.float()
    got = actual.float()
    diff = (ref - got).abs()
    if diff.numel() == 0:
        return TensorError(list(reference.shape), 0.0, 0.0, 0.0, 0.0)
    denom = ref.abs().clamp_min(torch.finfo(torch.float32).tiny)
    ref_flat = ref.flatten()
    got_flat = got.flatten()
    cosine = torch.nn.functional.cosine_similarity(
        ref_flat.unsqueeze(0), got_flat.unsqueeze(0), dim=1, eps=1e-12
    )
    return TensorError(
        shape=list(reference.shape),
        max_abs=float(diff.max().item()),
        mean_abs=float(diff.mean().item()),
        max_rel=float((diff / denom).max().item()),
        cosine_error=float((1.0 - cosine).item()),
    )


def logits_parity(reference: torch.Tensor, actual: torch.Tensor) -> dict:
    """Locate the first argmax fork and report top-2 decision margins."""
    if reference.shape != actual.shape or reference.ndim != 2:
        raise ValueError(
            "Logits parity requires equal rank-2 tensors, got "
            f"{reference.shape} and {actual.shape}"
        )
    ref = reference.float()
    got = actual.float()
    ref_top = torch.topk(ref, k=2, dim=-1)
    got_top = torch.topk(got, k=2, dim=-1)
    divergent = torch.nonzero(ref_top.indices[:, 0] != got_top.indices[:, 0])
    first_row = int(divergent[0].item()) if divergent.numel() else None

    report = {
        "tensor_error": asdict(tensor_error(reference, actual)),
        "num_rows": reference.shape[0],
        "num_argmax_divergences": int(divergent.numel()),
        "first_divergent_row": first_row,
        "minimum_reference_top2_margin": float(
            (ref_top.values[:, 0] - ref_top.values[:, 1]).min().item()
        ),
    }
    if first_row is not None:
        report["first_divergence"] = {
            "row": first_row,
            "reference_top2_ids": ref_top.indices[first_row].tolist(),
            "actual_top2_ids": got_top.indices[first_row].tolist(),
            "reference_top2_logits": ref_top.values[first_row].tolist(),
            "actual_top2_logits": got_top.values[first_row].tolist(),
            "reference_margin": float(
                ref_top.values[first_row, 0] - ref_top.values[first_row, 1]
            ),
            "actual_margin": float(
                got_top.values[first_row, 0] - got_top.values[first_row, 1]
            ),
        }
    return report


class AttentionTrace:
    """Collect per-layer backend outputs for one or more target forwards."""

    def __init__(self) -> None:
        self.outputs: dict[int, list[torch.Tensor]] = {}

    def record(self, layer_id: int, output: torch.Tensor) -> None:
        # Debug-only clone: backend output buffers are reused by subsequent
        # layers/replays.  Keeping a view here would compare overwritten data.
        self.outputs.setdefault(layer_id, []).append(output.detach().clone())

    def joined(self, layer_id: int) -> torch.Tensor:
        values = self.outputs[layer_id]
        return values[0] if len(values) == 1 else torch.cat(values, dim=0)


_active_attention_trace: ContextVar[Optional[AttentionTrace]] = ContextVar(
    "spec_mixed_attention_trace", default=None
)


@contextmanager
def record_attention(trace: AttentionTrace):
    token = _active_attention_trace.set(trace)
    try:
        yield
    finally:
        _active_attention_trace.reset(token)


def maybe_record_attention(layer_id: int, output: torch.Tensor) -> None:
    trace = _active_attention_trace.get()
    if trace is not None:
        trace.record(layer_id, output)


class OperatorTrace:
    """Selected token rows at model operator boundaries, in execution order."""

    def __init__(self) -> None:
        self.outputs: dict[str, torch.Tensor] = {}
        self.order: list[str] = []

    def record(self, name: str, value: torch.Tensor, token_start: int) -> None:
        if value.ndim == 0 or value.shape[0] <= token_start:
            return
        if name not in self.outputs:
            self.order.append(name)
        self.outputs[name] = value[token_start:].detach().clone()


_active_operator_trace: ContextVar[Optional[tuple[OperatorTrace, int]]] = ContextVar(
    "spec_mixed_operator_trace", default=None
)


@contextmanager
def record_operators(trace: OperatorTrace, token_start: int):
    token = _active_operator_trace.set((trace, token_start))
    try:
        yield
    finally:
        _active_operator_trace.reset(token)


def _record_operator_value(name: str, value, token_start: int, trace) -> None:
    if isinstance(value, torch.Tensor):
        trace.record(name, value, token_start)
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _record_operator_value(f"{name}.{index}", item, token_start, trace)


_OPERATOR_TRACE_SUFFIXES = (
    "input_layernorm",
    "self_attn.qkv_proj",
    "self_attn.rotary_emb",
    "self_attn.attn",
    "self_attn.o_proj",
    "post_attention_layernorm",
    "mlp.gate_up_proj",
    "mlp.down_proj",
)


def install_operator_trace_hooks(model) -> list:
    """Install debug-only hooks on token-shaped Qwen/Llama operator boundaries."""
    handles = []

    def make_hook(module_name: str):
        def hook(_module, inputs, output):
            active = _active_operator_trace.get()
            if active is None:
                return
            trace, token_start = active
            _record_operator_value(f"{module_name}.input", inputs, token_start, trace)
            _record_operator_value(f"{module_name}.output", output, token_start, trace)

        return hook

    for name, module in model.named_modules():
        if name.endswith(_OPERATOR_TRACE_SUFFIXES):
            handles.append(module.register_forward_hook(make_hook(name)))
    if not handles:
        raise RuntimeError("Operator parity found no supported model modules")
    return handles


def remove_operator_trace_hooks(handles: Iterable) -> None:
    for handle in handles:
        handle.remove()


def operator_parity(reference: OperatorTrace, actual: OperatorTrace) -> dict:
    missing = [name for name in reference.order if name not in actual.outputs]
    extra = [name for name in actual.order if name not in reference.outputs]
    if missing or extra:
        raise ValueError(f"Operator traces differ: missing={missing}, extra={extra}")
    entries = []
    first_mismatch = None
    for name in reference.order:
        error = asdict(tensor_error(reference.outputs[name], actual.outputs[name]))
        entry = {"name": name, **error}
        entries.append(entry)
        if first_mismatch is None and error["max_abs"] != 0.0:
            first_mismatch = entry
    return {"first_mismatch": first_mismatch, "operators": entries}


class KVRows:
    """A snapshot of selected physical MHA KV rows across target layers."""

    def __init__(
        self,
        locations: torch.Tensor,
        rows: dict[int, tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        self.locations = locations
        self.rows = rows

    @classmethod
    def capture(cls, pool, layer_ids: Iterable[int], locations: torch.Tensor):
        locations = torch.unique(locations.to(dtype=torch.int64))
        rows = {}
        for layer_id in sorted(set(layer_ids)):
            k_buffer, v_buffer = pool.get_kv_buffer(layer_id)
            rows[layer_id] = (
                k_buffer[locations].detach().clone(),
                v_buffer[locations].detach().clone(),
            )
        return cls(locations.detach().clone(), rows)

    def restore(self, pool) -> None:
        for layer_id, (key, value) in self.rows.items():
            k_buffer, v_buffer = pool.get_kv_buffer(layer_id)
            # Advanced indexing returns a copy, so ``buf[idx].copy_`` would not
            # restore the pool. index_copy_ performs the required scatter.
            k_buffer.index_copy_(0, self.locations, key)
            v_buffer.index_copy_(0, self.locations, value)

    def compare(self, other: "KVRows") -> dict:
        if not torch.equal(self.locations, other.locations):
            raise ValueError("KV snapshots use different physical locations")
        report = {}
        for layer_id, (ref_k, ref_v) in self.rows.items():
            got_k, got_v = other.rows[layer_id]
            report[str(layer_id)] = {
                "key": asdict(tensor_error(ref_k, got_k)),
                "value": asdict(tensor_error(ref_v, got_v)),
            }
        return report


def attention_parity(reference: AttentionTrace, actual: AttentionTrace) -> dict:
    if reference.outputs.keys() != actual.outputs.keys():
        raise ValueError(
            "Attention traces cover different layers: "
            f"{reference.outputs.keys()} != {actual.outputs.keys()}"
        )
    return {
        str(layer_id): asdict(
            tensor_error(reference.joined(layer_id), actual.joined(layer_id))
        )
        for layer_id in sorted(reference.outputs)
    }


def parity_output_dir() -> Optional[Path]:
    value = os.environ.get("SGLANG_SPEC_MIXED_PARITY_DIR")
    return Path(value) if value else None


def parity_max_steps() -> int:
    value = int(os.environ.get("SGLANG_SPEC_MIXED_PARITY_STEPS", "1"))
    if value < 0:
        raise ValueError("SGLANG_SPEC_MIXED_PARITY_STEPS must be non-negative")
    return value


def operator_parity_enabled() -> bool:
    return os.environ.get("SGLANG_SPEC_MIXED_OPERATOR_PARITY", "0") == "1"


def write_parity_report(output_dir: Path, index: int, report: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"spec_mixed_parity_{index:04d}.json"
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")
    tmp_path.replace(path)
    return path
