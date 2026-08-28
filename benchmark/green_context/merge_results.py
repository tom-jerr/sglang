#!/usr/bin/env python3
"""Merge checkpointed/segmented benchmark JSON files by case key."""

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("inputs", type=Path, nargs="+")
    args = parser.parse_args()

    cases = {}
    records = []
    metadata = {}
    for path in args.inputs:
        payload = json.loads(path.read_text())
        metadata.update(payload.get("metadata", {}))
        for case in payload.get("cases", []):
            cases[case_key(case)] = case
        records.extend(payload.get("records", []))

    metadata["segmented_run"] = len(args.inputs) > 1
    metadata["source_files"] = [str(path) for path in args.inputs]
    result = {
        "metadata": metadata,
        "cases": [cases[key] for key in sorted(cases)],
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Wrote {args.output} with {len(cases)} cases")


if __name__ == "__main__":
    main()
