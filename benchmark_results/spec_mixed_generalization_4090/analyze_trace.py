#!/usr/bin/env python3
"""Summarize Mixed+spec CPU/GPU overlap and synchronization from a Kineto trace."""

from __future__ import annotations

import argparse
import collections
import gzip
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable


STEP_RE = re.compile(r"^step\[(?P<mode>[^ ]+)(?: bs=(?P<bs>\d+))?")
SYNC_NAMES = {
    "aten::item",
    "aten::_local_scalar_dense",
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cudaCtxSynchronize",
}
PACK_NAMES = {"aten::cat", "aten::_foreach_copy_", "aten::cumsum", "aten::arange"}
MODEL_ENTRY_NAMES = {"aten::embedding", "sglang::unified_attention_with_output"}


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def distribution(values_us: Iterable[float]) -> dict[str, float | int | None]:
    values = list(values_us)
    return {
        "count": len(values),
        "mean_ms": None if not values else statistics.mean(values) / 1000,
        "p50_ms": None if not values else statistics.median(values) / 1000,
        "p90_ms": None if not values else percentile(values, 0.90) / 1000,
        "p99_ms": None if not values else percentile(values, 0.99) / 1000,
        "min_ms": None if not values else min(values) / 1000,
        "max_ms": None if not values else max(values) / 1000,
    }


def load_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as file:
        payload = json.load(file)
    return payload["traceEvents"] if isinstance(payload, dict) else payload


def contained(event: dict[str, Any], parent: dict[str, Any], lookback_us: float = 0) -> bool:
    timestamp = event.get("ts")
    return (
        event.get("ph") == "X"
        and timestamp is not None
        and parent["ts"] - lookback_us <= timestamp < parent["ts"] + parent["dur"]
        and event.get("pid") == parent.get("pid")
        and event.get("tid") == parent.get("tid")
    )


def analyze(path: Path) -> dict[str, Any]:
    events = load_events(path)
    steps = []
    for event in events:
        if event.get("cat") != "user_annotation" or event.get("ph") != "X":
            continue
        match = STEP_RE.match(str(event.get("name", "")))
        if match:
            event = dict(event)
            event["mode"] = match.group("mode")
            event["batch_size"] = int(match.group("bs")) if match.group("bs") else None
            steps.append(event)

    gpu_annotations = {
        event.get("args", {}).get("External id"): event
        for event in events
        if event.get("cat") == "gpu_user_annotation" and event.get("ph") == "X"
    }
    gpu_by_correlation: dict[int, list[dict[str, Any]]] = collections.defaultdict(list)
    for event in events:
        correlation = event.get("args", {}).get("correlation")
        if correlation is not None and event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}:
            gpu_by_correlation[correlation].append(event)

    by_mode: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for step in steps:
        by_mode[step["mode"]].append(step)
    mixed = by_mode.get("MIXED", [])

    mixed_cpu_ops = [
        event
        for event in events
        if event.get("cat") == "cpu_op" and any(contained(event, step) for step in mixed)
    ]
    mixed_runtime = [
        event
        for event in events
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and any(contained(event, step) for step in mixed)
    ]
    correlations = {
        event.get("args", {}).get("correlation")
        for event in mixed_runtime
        if event.get("args", {}).get("correlation") is not None
    }
    launched_gpu_events = [event for corr in correlations for event in gpu_by_correlation[corr]]

    gpu_mixed_durations = []
    cpu_finishes_before_gpu_us = []
    for step in mixed:
        external_id = step.get("args", {}).get("External id")
        gpu = gpu_annotations.get(external_id)
        if gpu is None:
            continue
        gpu_mixed_durations.append(gpu["dur"])
        cpu_finishes_before_gpu_us.append(
            gpu["ts"] + gpu["dur"] - (step["ts"] + step["dur"])
        )

    pre_attention_offsets = []
    pre_model_offsets = []
    pre_model_cpu_ops = []
    for step in mixed:
        attention = [
            event
            for event in events
            if event.get("cat") == "cpu_op"
            and event.get("name") == "sglang::unified_attention_with_output"
            and contained(event, step)
        ]
        if attention:
            first_attention_ts = min(event["ts"] for event in attention)
            pre_attention_offsets.append(first_attention_ts - step["ts"])
        model_entry = [
            event
            for event in events
            if event.get("cat") == "cpu_op"
            and event.get("name") in MODEL_ENTRY_NAMES
            and contained(event, step)
        ]
        if model_entry:
            first_model_ts = min(event["ts"] for event in model_entry)
            pre_model_offsets.append(first_model_ts - step["ts"])
            pre_model_cpu_ops.extend(
                event
                for event in events
                if event.get("cat") == "cpu_op"
                and contained(event, step)
                and event.get("ts", first_model_ts) < first_model_ts
            )

    sync_counts = collections.Counter(
        event.get("name")
        for event in mixed_cpu_ops + mixed_runtime
        if event.get("name") in SYNC_NAMES
    )
    pack_counts = collections.Counter(
        event.get("name") for event in mixed_cpu_ops if event.get("name") in PACK_NAMES
    )
    pre_model_pack_counts = collections.Counter(
        event.get("name")
        for event in pre_model_cpu_ops
        if event.get("name") in PACK_NAMES
    )
    memcpy = collections.Counter()
    memcpy_bytes = collections.Counter()
    for event in launched_gpu_events:
        if event.get("cat") != "gpu_memcpy":
            continue
        name = event.get("name", "unknown")
        memcpy[name] += 1
        memcpy_bytes[name] += int(event.get("args", {}).get("bytes", 0))

    op_duration = collections.Counter()
    op_count = collections.Counter()
    for event in mixed_cpu_ops:
        op_duration[event.get("name", "unknown")] += event.get("dur", 0)
        op_count[event.get("name", "unknown")] += 1

    def gpu_group(name: str) -> str:
        if name == "_fwd_kernel":
            return "triton_attention::_fwd_kernel"
        if "matmul_kernel" in name:
            return "matmul"
        if "store_kvcache" in name:
            return "store_kvcache"
        if "fused_rope" in name:
            return "rope"
        if "act_and_mul" in name:
            return "activation"
        if "copy" in name.lower():
            return "copy_kernel"
        return name[:120]

    gpu_kernel_duration = collections.Counter()
    gpu_kernel_count = collections.Counter()
    for event in launched_gpu_events:
        if event.get("cat") != "kernel":
            continue
        group = gpu_group(str(event.get("name", "unknown")))
        gpu_kernel_duration[group] += event.get("dur", 0)
        gpu_kernel_count[group] += 1

    mode_summary = {}
    for mode, mode_steps in sorted(by_mode.items()):
        mode_summary[mode] = {
            "duration": distribution(step["dur"] for step in mode_steps),
            "batch_sizes": dict(
                sorted(collections.Counter(step["batch_size"] for step in mode_steps).items())
            ),
        }

    return {
        "trace": str(path),
        "event_count": len(events),
        "steps": mode_summary,
        "mixed": {
            "cpu_duration": distribution(step["dur"] for step in mixed),
            "gpu_duration": distribution(gpu_mixed_durations),
            "pre_model_cpu": distribution(pre_model_offsets),
            "pre_attention_cpu": distribution(pre_attention_offsets),
            "gpu_tail_after_cpu_submission": distribution(cpu_finishes_before_gpu_us),
            "cpu_submission_hidden_count": sum(value >= 0 for value in cpu_finishes_before_gpu_us),
            "cpu_submission_pair_count": len(cpu_finishes_before_gpu_us),
            "forbidden_sync_counts": dict(sync_counts),
            "pre_model_pack_operator_counts": dict(pre_model_pack_counts),
            "whole_step_pack_operator_counts": dict(pack_counts),
            "launched_memcpy_counts": dict(memcpy),
            "launched_memcpy_bytes": dict(memcpy_bytes),
            "cuda_launch_count": sum(
                event.get("name") in {"cudaLaunchKernel", "cudaLaunchKernelExC"}
                for event in mixed_runtime
            ),
            "top_cpu_ops_inclusive_us": [
                {"name": name, "count": op_count[name], "duration_us": duration}
                for name, duration in op_duration.most_common(20)
            ],
            "top_gpu_kernel_groups_us": [
                {"name": name, "count": gpu_kernel_count[name], "duration_us": duration}
                for name, duration in gpu_kernel_duration.most_common(20)
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = [analyze(path) for path in args.trace]
    payload: Any = results[0] if len(results) == 1 else results
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
