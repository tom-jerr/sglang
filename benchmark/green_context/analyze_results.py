#!/usr/bin/env python3
"""Compare baseline and green-context benchmark JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def case_key(case: dict) -> tuple:
    if case["suite"] == "homogeneous":
        return case["suite"], case["prompt_tokens"], case["concurrency"]
    return (
        case["suite"],
        case["long_prompt_tokens"],
        case["foreground_concurrency"],
        case["background_concurrency"],
    )


def improvement(baseline: float | None, green: float | None) -> float | None:
    if baseline in (None, 0) or green is None:
        return None
    return (baseline - green) / baseline * 100


def number(value: float | None, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def build_report(baseline: dict, green: dict, pdmux_streams: dict | None = None) -> str:
    baseline_cases = {case_key(case): case for case in baseline["cases"]}
    green_cases = {case_key(case): case for case in green["cases"]}
    common_keys = sorted(set(baseline_cases) & set(green_cases))

    lines = [
        "# CUDA green-context benchmark comparison",
        "",
        "Positive improvement means lower latency with green contexts.",
        "",
        "## Homogeneous serving",
        "",
        "| Prompt | Concurrency | Baseline TTFT p95 (ms) | Green TTFT p95 (ms) | TTFT improvement | Baseline ITL p95 (ms) | Green ITL p95 (ms) | ITL improvement |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key in common_keys:
        if key[0] != "homogeneous":
            continue
        base = baseline_cases[key]["summary"]
        candidate = green_cases[key]["summary"]
        ttft_gain = improvement(base["ttft_ms_p95"], candidate["ttft_ms_p95"])
        itl_gain = improvement(base["itl_ms_p95"], candidate["itl_ms_p95"])
        lines.append(
            f"| {key[1]} | {key[2]} | {number(base['ttft_ms_p95'])} | "
            f"{number(candidate['ttft_ms_p95'])} | {number(ttft_gain)}% | "
            f"{number(base['itl_ms_p95'])} | {number(candidate['itl_ms_p95'])} | "
            f"{number(itl_gain)}% |"
        )

    lines.extend(
        [
            "",
            "## Short-decode latency under long-prefill interference",
            "",
            "| Background prompt | Foreground concurrency | Baseline ITL p95 (ms) | Green ITL p95 (ms) | ITL p95 improvement | Baseline ITL p99 (ms) | Green ITL p99 (ms) | ITL p99 improvement |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for key in common_keys:
        if key[0] != "interference":
            continue
        base = baseline_cases[key]["foreground_summary"]
        candidate = green_cases[key]["foreground_summary"]
        p95_gain = improvement(base["itl_ms_p95"], candidate["itl_ms_p95"])
        p99_gain = improvement(base["itl_ms_p99"], candidate["itl_ms_p99"])
        lines.append(
            f"| {key[1]} | {key[2]} | {number(base['itl_ms_p95'])} | "
            f"{number(candidate['itl_ms_p95'])} | {number(p95_gain)}% | "
            f"{number(base['itl_ms_p99'])} | {number(candidate['itl_ms_p99'])} | "
            f"{number(p99_gain)}% |"
        )

    if pdmux_streams is not None:
        stream_cases = {case_key(case): case for case in pdmux_streams["cases"]}
        control_keys = sorted(set(stream_cases) & set(green_cases))
        lines.extend(
            [
                "",
                "## Net green-context effect (same PDMux scheduler)",
                "",
                "| Background prompt | Foreground concurrency | Ordinary-stream ITL p95 (ms) | Green-context ITL p95 (ms) | Green-context improvement | Ordinary-stream ITL p99 (ms) | Green-context ITL p99 (ms) | Green-context improvement |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for key in control_keys:
            if key[0] != "interference":
                continue
            control = stream_cases[key]["foreground_summary"]
            candidate = green_cases[key]["foreground_summary"]
            p95_gain = improvement(control["itl_ms_p95"], candidate["itl_ms_p95"])
            p99_gain = improvement(control["itl_ms_p99"], candidate["itl_ms_p99"])
            lines.append(
                f"| {key[1]} | {key[2]} | {number(control['itl_ms_p95'])} | "
                f"{number(candidate['itl_ms_p95'])} | {number(p95_gain)}% | "
                f"{number(control['itl_ms_p99'])} | {number(candidate['itl_ms_p99'])} | "
                f"{number(p99_gain)}% |"
            )

    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--pdmux-streams", type=Path)
    parser.add_argument("--green-context", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    baseline = json.loads(args.baseline.read_text())
    green = json.loads(args.green_context.read_text())
    pdmux_streams = (
        json.loads(args.pdmux_streams.read_text()) if args.pdmux_streams else None
    )
    report = build_report(baseline, green, pdmux_streams)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
    print(report)


if __name__ == "__main__":
    main()
