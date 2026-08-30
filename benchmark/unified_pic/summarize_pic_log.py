"""Summarize one observer decision per request from a PP/TP server log."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_PIC_RE = re.compile(
    r"PIC observer rid=(\w+) prefix_tokens=(\d+) pic_hit_tokens=(\d+) "
    r"pic_spans=(\d+) misses=(\d+)"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    parser.add_argument("eval_report", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    decisions: dict[str, tuple[int, int, int]] = {}
    conflicts = []
    for match in _PIC_RE.finditer(args.log.read_text(errors="replace")):
        rid, _prefix, hit_tokens, spans, misses = match.groups()
        values = (int(hit_tokens), int(spans), int(misses))
        # A request can have a small ordinary Radix hit (typically BOS plus the
        # request-local metadata) and still have position-independent spans.
        # Chunked prefill then emits later zero-increment observer records, so
        # retain the first positive PIC decision per rid and deduplicate PP logs.
        if values[0] == 0:
            continue
        previous = decisions.setdefault(rid, values)
        if previous != values:
            conflicts.append({"rid": rid, "first": previous, "other": values})

    eval_report = json.loads(args.eval_report.read_text())
    prompt_tokens = sum(row["prompt_tokens"] for row in eval_report["results"])
    hit_tokens = sum(values[0] for values in decisions.values())
    report = {
        "requests_with_hits": len(decisions),
        "hit_tokens": hit_tokens,
        "prompt_tokens": prompt_tokens,
        "candidate_coverage": hit_tokens / max(prompt_tokens, 1),
        "spans": sum(values[1] for values in decisions.values()),
        "misses": sum(values[2] for values in decisions.values()),
        "stage_conflicts": conflicts,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
