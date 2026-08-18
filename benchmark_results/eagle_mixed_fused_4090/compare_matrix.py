#!/usr/bin/env python3
"""Compare latency_matrix.py artifacts and verify token-stream parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUANTILES = ("mean", "p50", "p95", "p99")


def delta_pct(value: float, baseline: float) -> float:
    return 100 * (value / baseline - 1)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def record_map(payload: dict[str, Any]) -> dict[tuple[int, int], dict[str, Any]]:
    return {
        (record["context"], record["sample"]): record
        for record in payload["records"]
    }


def pairwise_summary(
    candidate: dict[str, Any], reference: dict[str, Any]
) -> dict[str, Any]:
    result = {}
    for context, stats in candidate["summary"].items():
        if context == "all":
            continue
        reference_stats = reference["summary"][context]
        result[context] = {}
        for metric in ("ttft_ms", "tpot_ms"):
            result[context][metric] = {
                "candidate": {
                    quantile: stats[metric][quantile] for quantile in QUANTILES
                },
                "reference": {
                    quantile: reference_stats[metric][quantile]
                    for quantile in QUANTILES
                },
                "delta_pct": {
                    quantile: delta_pct(
                        stats[metric][quantile],
                        reference_stats[metric][quantile],
                    )
                    for quantile in QUANTILES
                },
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="LABEL=JSON; first input is the delta/accuracy baseline",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    inputs = []
    for item in args.input:
        label, separator, raw_path = item.partition("=")
        if not separator:
            parser.error(f"invalid --input {item!r}; expected LABEL=JSON")
        inputs.append((label, load(Path(raw_path))))

    baseline_label, baseline = inputs[0]
    baseline_records = record_map(baseline)
    comparisons = {}
    for label, payload in inputs:
        records = record_map(payload)
        if records.keys() != baseline_records.keys():
            raise ValueError(f"sample set differs for {label}")
        mismatches = []
        for key in records:
            if records[key]["output_ids"] != baseline_records[key]["output_ids"]:
                mismatches.append({"context": key[0], "sample": key[1]})
        comparisons[label] = {
            "output_matches": len(records) - len(mismatches),
            "output_total": len(records),
            "mismatches": mismatches,
            "contexts": {},
        }
        for context, stats in payload["summary"].items():
            if context == "all":
                continue
            baseline_stats = baseline["summary"][context]
            row = {}
            for metric in ("ttft_ms", "tpot_ms"):
                row[metric] = {
                    quantile: stats[metric][quantile]
                    for quantile in QUANTILES
                }
                row[metric]["delta_pct"] = {
                    quantile: delta_pct(
                        stats[metric][quantile],
                        baseline_stats[metric][quantile],
                    )
                    for quantile in QUANTILES
                }
            comparisons[label]["contexts"][context] = row

    payloads = dict(inputs)
    pairwise = {}
    for name, candidate_label, reference_label in (
        (
            "fused_without_mixed_control",
            "mixed_off_fused_on",
            "mixed_off_fused_off",
        ),
        (
            "mixed_chunk_effect",
            "mixed_on_fused_off",
            "mixed_off_fused_off",
        ),
        (
            "fused_attention_increment",
            "mixed_on_fused_on",
            "mixed_on_fused_off",
        ),
    ):
        if candidate_label in payloads and reference_label in payloads:
            pairwise[name] = {
                "candidate": candidate_label,
                "reference": reference_label,
                "contexts": pairwise_summary(
                    payloads[candidate_label], payloads[reference_label]
                ),
            }

    result = {
        "baseline": baseline_label,
        "all_outputs_match": all(
            row["output_matches"] == row["output_total"]
            for row in comparisons.values()
        ),
        "comparisons": comparisons,
        "pairwise": pairwise,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
