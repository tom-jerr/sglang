#!/usr/bin/env python3
"""Latency/accuracy workload for EAGLE mixed-chunk and fused-attention A/B.

The server is intentionally managed outside this program. Run the exact same
workload once for each server configuration, then compare the JSON artifacts
with ``compare_matrix.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import random
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests
from transformers import AutoTokenizer


@dataclass(frozen=True)
class WorkloadConfig:
    contexts: list[int]
    samples_per_context: int
    probe_output_len: int
    running_batch_size: int
    running_context: int
    running_output_len: int
    probe_stagger_ms: float
    seed: int


def nearest_rank(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    if not 0 <= quantile <= 1:
        raise ValueError(f"invalid quantile: {quantile}")
    ordered = sorted(values)
    rank = max(1, math.ceil(quantile * len(ordered)))
    return ordered[rank - 1]


def distribution(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": nearest_rank(values, 0.95),
        "p99": nearest_rank(values, 0.99),
        "min": min(values),
        "max": max(values),
    }


def token_ids(bank: list[int], length: int, seed: int, first: int) -> list[int]:
    if not bank:
        rng = random.Random(seed)
        values = [rng.randrange(1000, 120000) for _ in range(length)]
    else:
        offset = seed % len(bank)
        values = [bank[(offset + index) % len(bank)] for index in range(length)]
    if values:
        values[0] = first
    return values


class Client:
    def __init__(self, base_url: str, timeout: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def flush(self) -> None:
        response = requests.post(f"{self.base_url}/flush_cache", timeout=60)
        response.raise_for_status()

    def generate(self, ids: list[int], output_len: int) -> dict[str, Any]:
        payload = {
            "input_ids": ids,
            "sampling_params": {
                "max_new_tokens": output_len,
                "temperature": 0.0,
                "ignore_eos": True,
            },
            "stream": True,
        }
        start = time.perf_counter()
        first_token_at = None
        output_ids: list[int] = []
        meta: dict[str, Any] = {}
        with requests.post(
            f"{self.base_url}/generate",
            json=payload,
            stream=True,
            timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                if line == b"[DONE]":
                    continue
                chunk = json.loads(line)
                chunk_ids = list(chunk.get("output_ids") or ())
                meta = chunk.get("meta_info") or meta
                if chunk_ids and first_token_at is None:
                    first_token_at = time.perf_counter()
                completion = int(meta.get("completion_tokens", len(chunk_ids)))
                if len(chunk_ids) == completion:
                    output_ids = chunk_ids
                elif chunk_ids:
                    output_ids.extend(chunk_ids)

        end = time.perf_counter()
        if first_token_at is None:
            raise RuntimeError("stream completed without an output token")
        ttft_ms = (first_token_at - start) * 1000
        latency_ms = (end - start) * 1000
        completion_tokens = int(meta.get("completion_tokens", len(output_ids)))
        if completion_tokens <= 1:
            raise RuntimeError("TPOT requires at least two completion tokens")
        return {
            "ttft_ms": ttft_ms,
            "tpot_ms": (latency_ms - ttft_ms) / (completion_tokens - 1),
            "latency_ms": latency_ms,
            "completion_tokens": completion_tokens,
            "cached_tokens": int(meta.get("cached_tokens", 0)),
            "spec_accept_length": meta.get("spec_accept_length"),
            "output_ids": output_ids,
        }


def run_context(
    client: Client,
    bank: list[int],
    config: WorkloadConfig,
    context: int,
) -> list[dict[str, Any]]:
    client.flush()
    running_prompts = [
        token_ids(
            bank,
            config.running_context,
            config.seed + context * 1000 + index,
            first=120000 + index,
        )
        for index in range(config.running_batch_size)
    ]
    probes = [
        token_ids(
            bank,
            context,
            config.seed + context * 10000 + sample,
            first=121000 + sample,
        )
        for sample in range(config.samples_per_context)
    ]

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.running_batch_size + config.samples_per_context
    ) as pool:
        running = [
            pool.submit(client.generate, prompt, config.running_output_len)
            for prompt in running_prompts
        ]
        time.sleep(0.08)
        probe_futures = []
        for prompt in probes:
            probe_futures.append(
                pool.submit(client.generate, prompt, config.probe_output_len)
            )
            time.sleep(config.probe_stagger_ms / 1000)
        probe_results = [future.result() for future in probe_futures]
        # Surface errors in the running decode load as workload failures.
        for future in running:
            future.result()

    return [
        {
            "sample": sample,
            "context": context,
            **result,
        }
        for sample, result in enumerate(probe_results)
    ]


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    contexts = sorted({record["context"] for record in records})
    result = {}
    for context in contexts:
        rows = [record for record in records if record["context"] == context]
        result[str(context)] = {
            "requests": len(rows),
            "ttft_ms": distribution([row["ttft_ms"] for row in rows]),
            "tpot_ms": distribution([row["tpot_ms"] for row in rows]),
            "e2e_ms": distribution([row["latency_ms"] for row in rows]),
            "cached_tokens": distribution(
                [float(row["cached_tokens"]) for row in rows]
            ),
        }
    result["all"] = {
        "requests": len(records),
        "ttft_ms": distribution([row["ttft_ms"] for row in records]),
        "tpot_ms": distribution([row["tpot_ms"] for row in records]),
        "e2e_ms": distribution([row["latency_ms"] for row in records]),
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", default="/workspace/models/Qwen3-4B")
    parser.add_argument("--contexts", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--samples-per-context", type=int, default=12)
    parser.add_argument("--probe-output-len", type=int, default=64)
    parser.add_argument("--running-batch-size", type=int, default=4)
    parser.add_argument("--running-context", type=int, default=512)
    parser.add_argument("--running-output-len", type=int, default=128)
    parser.add_argument("--probe-stagger-ms", type=float, default=30)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    corpus = (
        "Analyze a high performance inference scheduler that combines chunked "
        "prefill with speculative decoding. Explain metadata ownership, cache "
        "positions, attention planning, stream dependencies, and latency. "
    )
    bank = tokenizer.encode(corpus * 64, add_special_tokens=False)
    config = WorkloadConfig(
        contexts=args.contexts,
        samples_per_context=args.samples_per_context,
        probe_output_len=args.probe_output_len,
        running_batch_size=args.running_batch_size,
        running_context=args.running_context,
        running_output_len=args.running_output_len,
        probe_stagger_ms=args.probe_stagger_ms,
        seed=args.seed,
    )
    client = Client(args.url, args.timeout)

    # Warm ordinary kernels first, then exercise a real mixed iteration. The
    # latter is required to keep first-use Triton compilation out of ctx=512.
    client.generate(token_ids(bank, 256, args.seed - 1, first=119999), 8)
    client.flush()
    warmup_config = WorkloadConfig(
        contexts=[512],
        samples_per_context=4,
        probe_output_len=16,
        running_batch_size=config.running_batch_size,
        running_context=config.running_context,
        running_output_len=64,
        probe_stagger_ms=config.probe_stagger_ms,
        seed=config.seed - 1000,
    )
    run_context(client, bank, warmup_config, 512)
    client.flush()

    records: list[dict[str, Any]] = []
    for context in config.contexts:
        rows = run_context(client, bank, config, context)
        records.extend(rows)
        stats = aggregate(rows)[str(context)]
        print(
            f"ctx={context:4d} requests={len(rows):2d} "
            f"TTFT p50={stats['ttft_ms']['p50']:.2f} ms "
            f"TPOT p50={stats['tpot_ms']['p50']:.2f} ms",
            flush=True,
        )

    payload = {
        "label": args.label,
        "workload": asdict(config),
        "summary": aggregate(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
