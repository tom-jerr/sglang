#!/usr/bin/env python3
"""Compare same-workload Mixed-enabled and separated result artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return statistics.mean(values)


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile, matching analyze_trace.py."""
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p99": percentile(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def delta_pct(on: float, off: float) -> float:
    return 100 * (on / off - 1)


def tpot_ms(latency_ms: float, ttft_ms: float, output_tokens: int) -> float:
    if output_tokens <= 1:
        raise ValueError("TPOT requires at least two output tokens")
    return (latency_ms - ttft_ms) / (output_tokens - 1)


def compare_distribution(
    mixed_values: list[float], separated_values: list[float]
) -> dict[str, Any]:
    mixed = distribution(mixed_values)
    separated = distribution(separated_values)
    return {
        "mixed": mixed,
        "separated": separated,
        "delta_pct": {
            key: delta_pct(float(mixed[key]), float(separated[key]))
            for key in ("mean", "p50", "p99")
        },
    }


def load_records(path: Path) -> dict[str, dict[str, Any]]:
    payload = json.loads(path.read_text())
    return {record["name"]: record for record in payload["records"]}


def running_stat(record: dict[str, Any], name: str) -> float:
    values = [value for value in record[name] if value is not None]
    return mean(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed", type=Path, required=True)
    parser.add_argument("--separated", type=Path, required=True)
    parser.add_argument("--fairness-mixed", type=Path, required=True)
    parser.add_argument("--fairness-separated", type=Path, required=True)
    parser.add_argument("--profile-hit-mixed", type=Path)
    parser.add_argument("--profile-hit-separated", type=Path)
    parser.add_argument("--profile-miss-mixed", type=Path)
    parser.add_argument("--profile-miss-separated", type=Path)
    parser.add_argument("--running-output-len", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    mixed = load_records(args.mixed)
    separated = load_records(args.separated)
    if mixed.keys() != separated.keys():
        raise ValueError("matrix case sets differ")

    cases = []
    for name in mixed:
        on, off = mixed[name], separated[name]
        mixed_running_mean = mean(on["running_latency_ms"])
        separated_running_mean = mean(off["running_latency_ms"])
        mixed_probe_tpot = tpot_ms(
            on["probe_latency_ms"],
            on["probe_ttft_ms"],
            len(on["probe_output_ids"]),
        )
        separated_probe_tpot = tpot_ms(
            off["probe_latency_ms"],
            off["probe_ttft_ms"],
            len(off["probe_output_ids"]),
        )
        cases.append(
            {
                "name": name,
                "cache": on["case"]["cache"],
                "context": on["case"]["context"],
                "running_bs": on["case"]["running_bs"],
                "suffix": on["case"]["suffix"],
                "mixed_ttft_ms": on["probe_ttft_ms"],
                "separated_ttft_ms": off["probe_ttft_ms"],
                "ttft_delta_pct": delta_pct(on["probe_ttft_ms"], off["probe_ttft_ms"]),
                "mixed_e2e_ms": on["probe_latency_ms"],
                "separated_e2e_ms": off["probe_latency_ms"],
                "e2e_delta_pct": delta_pct(on["probe_latency_ms"], off["probe_latency_ms"]),
                "mixed_probe_tpot_ms": mixed_probe_tpot,
                "separated_probe_tpot_ms": separated_probe_tpot,
                "probe_tpot_delta_pct": delta_pct(
                    mixed_probe_tpot, separated_probe_tpot
                ),
                "mixed_running_e2e_mean_ms": mixed_running_mean,
                "separated_running_e2e_mean_ms": separated_running_mean,
                "running_e2e_delta_pct": delta_pct(
                    mixed_running_mean, separated_running_mean
                ),
                "mixed_output_match": on["output_match"],
                "separated_output_match": off["output_match"],
            }
        )

    aggregate = {}
    for cache in ("hit", "miss"):
        rows = [row for row in cases if row["cache"] == cache]
        aggregate[cache] = {
            "cases": len(rows),
            "median_per_case_ttft_delta_pct": statistics.median(
                row["ttft_delta_pct"] for row in rows
            ),
            "median_per_case_e2e_delta_pct": statistics.median(
                row["e2e_delta_pct"] for row in rows
            ),
            "median_per_case_running_e2e_delta_pct": statistics.median(
                row["running_e2e_delta_pct"] for row in rows
            ),
            "distributions": {
                "probe_ttft_ms": compare_distribution(
                    [row["mixed_ttft_ms"] for row in rows],
                    [row["separated_ttft_ms"] for row in rows],
                ),
                "probe_e2e_ms": compare_distribution(
                    [row["mixed_e2e_ms"] for row in rows],
                    [row["separated_e2e_ms"] for row in rows],
                ),
                "probe_tpot_ms": compare_distribution(
                    [row["mixed_probe_tpot_ms"] for row in rows],
                    [row["separated_probe_tpot_ms"] for row in rows],
                ),
                # One mean over concurrently running requests per case, so a
                # large running_bs case does not dominate the matrix aggregate.
                "running_e2e_case_mean_ms": compare_distribution(
                    [row["mixed_running_e2e_mean_ms"] for row in rows],
                    [row["separated_running_e2e_mean_ms"] for row in rows],
                ),
                "per_case_ttft_delta_pct": distribution(
                    [row["ttft_delta_pct"] for row in rows]
                ),
                "per_case_e2e_delta_pct": distribution(
                    [row["e2e_delta_pct"] for row in rows]
                ),
                "per_case_probe_tpot_delta_pct": distribution(
                    [row["probe_tpot_delta_pct"] for row in rows]
                ),
                "per_case_running_e2e_delta_pct": distribution(
                    [row["running_e2e_delta_pct"] for row in rows]
                ),
            },
        }

    fairness_on = load_records(args.fairness_mixed)
    fairness_off = load_records(args.fairness_separated)
    fairness = []
    for name in fairness_on:
        on, off = fairness_on[name], fairness_off[name]
        row = {"name": name}
        mixed_probe_tpot = tpot_ms(
            on["probe_latency_ms"],
            on["probe_ttft_ms"],
            len(on["probe_output_ids"]),
        )
        separated_probe_tpot = tpot_ms(
            off["probe_latency_ms"],
            off["probe_ttft_ms"],
            len(off["probe_output_ids"]),
        )
        mixed_running_tpot = [
            tpot_ms(latency, ttft, args.running_output_len)
            for latency, ttft in zip(
                on["running_latency_ms"], on["running_ttft_ms"], strict=True
            )
        ]
        separated_running_tpot = [
            tpot_ms(latency, ttft, args.running_output_len)
            for latency, ttft in zip(
                off["running_latency_ms"], off["running_ttft_ms"], strict=True
            )
        ]
        metrics = {
            "probe_ttft": (on["probe_ttft_ms"], off["probe_ttft_ms"]),
            "probe_e2e": (on["probe_latency_ms"], off["probe_latency_ms"]),
            "probe_tpot": (mixed_probe_tpot, separated_probe_tpot),
            "running_e2e_mean": (
                mean(on["running_latency_ms"]),
                mean(off["running_latency_ms"]),
            ),
            "running_p99_itl_mean": (
                running_stat(on, "running_itl_p99_ms"),
                running_stat(off, "running_itl_p99_ms"),
            ),
            "running_max_itl": (
                max(value for value in on["running_itl_max_ms"] if value is not None),
                max(value for value in off["running_itl_max_ms"] if value is not None),
            ),
        }
        for metric, (on_value, off_value) in metrics.items():
            row[f"mixed_{metric}_ms"] = on_value
            row[f"separated_{metric}_ms"] = off_value
            row[f"{metric}_delta_pct"] = delta_pct(on_value, off_value)
        row["running_distributions"] = {
            "ttft_ms": compare_distribution(
                [value for value in on["running_ttft_ms"] if value is not None],
                [value for value in off["running_ttft_ms"] if value is not None],
            ),
            "e2e_ms": compare_distribution(
                [value for value in on["running_latency_ms"] if value is not None],
                [value for value in off["running_latency_ms"] if value is not None],
            ),
            "tpot_ms": compare_distribution(
                mixed_running_tpot, separated_running_tpot
            ),
            # Inputs are already one p99/max ITL value per running request.
            "per_request_p99_itl_ms": compare_distribution(
                [value for value in on["running_itl_p99_ms"] if value is not None],
                [value for value in off["running_itl_p99_ms"] if value is not None],
            ),
            "per_request_max_itl_ms": compare_distribution(
                [value for value in on["running_itl_max_ms"] if value is not None],
                [value for value in off["running_itl_max_ms"] if value is not None],
            ),
        }
        fairness.append(row)

    fairness_aggregate = {}
    for metric, field in (
        ("running_ttft_ms", "running_ttft_ms"),
        ("running_e2e_ms", "running_latency_ms"),
        ("per_request_p99_itl_ms", "running_itl_p99_ms"),
        ("per_request_max_itl_ms", "running_itl_max_ms"),
    ):
        fairness_aggregate[metric] = compare_distribution(
            [
                value
                for record in fairness_on.values()
                for value in record[field]
                if value is not None
            ],
            [
                value
                for record in fairness_off.values()
                for value in record[field]
                if value is not None
            ],
        )
    fairness_aggregate["running_tpot_ms"] = compare_distribution(
        [
            tpot_ms(latency, ttft, args.running_output_len)
            for record in fairness_on.values()
            for latency, ttft in zip(
                record["running_latency_ms"],
                record["running_ttft_ms"],
                strict=True,
            )
        ],
        [
            tpot_ms(latency, ttft, args.running_output_len)
            for record in fairness_off.values()
            for latency, ttft in zip(
                record["running_latency_ms"],
                record["running_ttft_ms"],
                strict=True,
            )
        ],
    )

    profile_paths = {
        "hit": (args.profile_hit_mixed, args.profile_hit_separated),
        "miss": (args.profile_miss_mixed, args.profile_miss_separated),
    }
    profiler = {}
    for cache, (mixed_path, separated_path) in profile_paths.items():
        if mixed_path is None or separated_path is None:
            continue
        mixed_profile = json.loads(mixed_path.read_text())
        separated_profile = json.loads(separated_path.read_text())
        profiler[cache] = {
            "mixed_step": mixed_profile["steps"].get("MIXED"),
            "separated_extend_step": separated_profile["steps"].get("EXTEND"),
            "mixed_target_verify_step": mixed_profile["steps"].get(
                "TARGET_VERIFY"
            ),
            "separated_target_verify_step": separated_profile["steps"].get(
                "TARGET_VERIFY"
            ),
            "mixed_cpu_gpu_overlap": {
                key: mixed_profile["mixed"][key]
                for key in (
                    "cpu_duration",
                    "gpu_duration",
                    "pre_model_cpu",
                    "gpu_tail_after_cpu_submission",
                    "cpu_submission_hidden_count",
                    "cpu_submission_pair_count",
                    "forbidden_sync_counts",
                    "launched_memcpy_counts",
                    "cuda_launch_count",
                )
            },
        }

    payload = {
        "aggregate": aggregate,
        "cases": cases,
        "fairness": fairness,
        "fairness_aggregate": fairness_aggregate,
        "profiler": profiler,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
