"""Offline recovery probe for the phase-1 Unified PIC component.

Each JSONL row must contain ``input_ids`` and may contain ``tenant_id`` or
``session_id``.  Requests are processed in file order, matching the runtime
observer's lookup-before-publish behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sglang.srt.mem_cache.pic.component import PicSpanComponent
from sglang.srt.mem_cache.pic.config import PicConfig
from sglang.srt.mem_cache.pic.types import PicNamespace, ShareScope


def longest_exact_prefix(tokens: tuple[int, ...], prior: list[tuple[int, ...]]) -> int:
    best = 0
    for candidate in prior:
        limit = min(len(tokens), len(candidate))
        matched = 0
        while matched < limit and tokens[matched] == candidate[matched]:
            matched += 1
        best = max(best, matched)
    return best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path, help="JSONL file containing input_ids")
    parser.add_argument("--model-fingerprint", required=True)
    parser.add_argument("--tokenizer-fingerprint", required=True)
    parser.add_argument("--min-chunk-tokens", type=int, default=32)
    parser.add_argument("--target-chunk-tokens", type=int, default=128)
    parser.add_argument("--max-chunk-tokens", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PicConfig(
        min_chunk_tokens=args.min_chunk_tokens,
        target_chunk_tokens=args.target_chunk_tokens,
        max_chunk_tokens=args.max_chunk_tokens,
        model_fingerprint=args.model_fingerprint,
        tokenizer_fingerprint=args.tokenizer_fingerprint,
    )
    component = PicSpanComponent(config)
    prior_by_tenant: dict[str, list[tuple[int, ...]]] = {}
    totals = {
        "requests": 0,
        "prompt_tokens": 0,
        "exact_prefix_tokens": 0,
        "pic_unique_tokens": 0,
        "pic_span_hits": 0,
    }

    with args.trace.open(encoding="utf-8") as trace_file:
        for line_number, line in enumerate(trace_file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "input_ids" not in row:
                raise ValueError(f"line {line_number}: missing input_ids")
            tokens = tuple(int(token) for token in row["input_ids"])
            tenant = str(row.get("tenant_id", "__default_tenant__"))
            session = row.get("session_id")
            namespace = PicNamespace(
                tenant_id=tenant,
                session_id=session,
                share_scope=(
                    ShareScope.SESSION if session is not None else ShareScope.TENANT
                ),
                model_fingerprint=config.model_fingerprint,
                tokenizer_fingerprint=config.tokenizer_fingerprint,
                cache_format=config.cache_format,
            )
            prior = prior_by_tenant.setdefault(tenant, [])
            prefix = longest_exact_prefix(tokens, prior)
            plan = component.observe_match(
                tokens,
                namespace=namespace,
                prefix_tokens=prefix,
            )
            component.observe_publish(tokens, namespace=namespace)
            prior.append(tokens)

            totals["requests"] += 1
            totals["prompt_tokens"] += len(tokens)
            totals["exact_prefix_tokens"] += prefix
            totals["pic_unique_tokens"] += plan.hit_tokens
            totals["pic_span_hits"] += len(plan.hits)

    denominator = max(totals["prompt_tokens"], 1)
    totals["exact_prefix_rate"] = totals["exact_prefix_tokens"] / denominator
    totals["pic_unique_rate"] = totals["pic_unique_tokens"] / denominator
    totals["total_cache_rate"] = (
        totals["exact_prefix_tokens"] + totals["pic_unique_tokens"]
    ) / denominator
    print(json.dumps(totals, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
