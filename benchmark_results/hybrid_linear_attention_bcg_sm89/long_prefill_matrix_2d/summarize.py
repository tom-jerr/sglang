#!/usr/bin/env python3
"""Validate and summarize the controlled Qwen3.8 prefill BCG matrix."""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
BASELINE = "per_layer_break_2d"
OPTIMIZED = "body_2d"
INPUT_LENGTHS = (1024, 2048, 4096, 8192, 16384)
CONCURRENCIES = (1, 4, 8, 16, 32)


def load(config: str, input_len: int, concurrency: int) -> dict:
    path = ROOT / config / f"i{input_len}_c{concurrency}.json"
    with path.open() as file:
        result = json.loads(file.readline())
    expected_requests = 4 if concurrency <= 4 else concurrency
    assert result["completed"] == expected_requests, (path, result["completed"])
    assert result["total_input_tokens"] == input_len * expected_requests, (
        path,
        result["total_input_tokens"],
    )
    return result


def geometric_mean(values: list[float]) -> float:
    return math.exp(sum(math.log(value) for value in values) / len(values))


def change(new: float, old: float) -> float:
    return (new / old - 1) * 100


def main() -> None:
    data = {
        (config, input_len, concurrency): load(config, input_len, concurrency)
        for config in (BASELINE, OPTIMIZED)
        for input_len in INPUT_LENGTHS
        for concurrency in CONCURRENCIES
    }

    for metric, lower_is_better in (
        ("mean_ttft_ms", True),
        ("input_throughput", False),
    ):
        ratios = [
            data[OPTIMIZED, input_len, concurrency][metric]
            / data[BASELINE, input_len, concurrency][metric]
            for input_len in INPUT_LENGTHS
            for concurrency in CONCURRENCIES
        ]
        changes = [(ratio - 1) * 100 for ratio in ratios]
        wins = sum(ratio < 1 for ratio in ratios) if lower_is_better else sum(
            ratio > 1 for ratio in ratios
        )
        print(
            f"{metric}: geometric change={change(geometric_mean(ratios), 1):+.2f}% "
            f"median={statistics.median(changes):+.2f}% "
            f"range=[{min(changes):+.2f}%, {max(changes):+.2f}%] wins={wins}/25"
        )

    print("\n| input | conc | TTFT A | TTFT C | delta | tok/s A | tok/s C | delta |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for input_len in INPUT_LENGTHS:
        for concurrency in CONCURRENCIES:
            baseline = data[BASELINE, input_len, concurrency]
            optimized = data[OPTIMIZED, input_len, concurrency]
            print(
                f"| {input_len} | {concurrency} | "
                f"{baseline['mean_ttft_ms']:.2f} | {optimized['mean_ttft_ms']:.2f} | "
                f"{change(optimized['mean_ttft_ms'], baseline['mean_ttft_ms']):+.2f}% | "
                f"{baseline['input_throughput']:.2f} | {optimized['input_throughput']:.2f} | "
                f"{change(optimized['input_throughput'], baseline['input_throughput']):+.2f}% |"
            )


if __name__ == "__main__":
    main()
