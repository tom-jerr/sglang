"""Compare outputs from baseline and PIC-observer validation runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("pic", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = json.loads(args.baseline.read_text())
    pic = json.loads(args.pic.read_text())
    baseline_by_id = {row["case_id"]: row for row in baseline["results"]}
    pic_by_id = {row["case_id"]: row for row in pic["results"]}
    common_ids = sorted(baseline_by_id.keys() & pic_by_id.keys())
    rows = []
    for case_id in common_ids:
        left = baseline_by_id[case_id]
        right = pic_by_id[case_id]
        token_match = left["output_token_ids"] == right["output_token_ids"]
        text_match = left["text"] == right["text"]
        rows.append(
            {
                "case_id": case_id,
                "token_match": token_match,
                "text_match": text_match,
                "baseline_latency_s": left["latency_s"],
                "pic_latency_s": right["latency_s"],
            }
        )
    report = {
        "cases": len(rows),
        "token_exact_matches": sum(row["token_match"] for row in rows),
        "text_exact_matches": sum(row["text_match"] for row in rows),
        "rows": rows,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
