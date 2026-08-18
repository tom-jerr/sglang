#!/usr/bin/env python3
"""Deterministic prefix-hit/miss generalization workload for EAGLE Mixed+spec."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import random
import statistics
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


_TOKEN_BANK: list[int] | None = None


@dataclass(frozen=True)
class Case:
    cache: str
    context: int
    running_bs: int
    suffix: int

    @property
    def name(self) -> str:
        return f"{self.cache}-ctx{self.context}-bs{self.running_bs}-s{self.suffix}"


CASES = [
    Case("hit", 512, 1, 1),
    Case("hit", 512, 8, 2),
    Case("hit", 2048, 1, 2),
    Case("hit", 2048, 4, 1),
    Case("hit", 2048, 4, 4),
    Case("hit", 2048, 4, 5),
    Case("hit", 2048, 4, 16),
    Case("hit", 2048, 8, 2),
    Case("hit", 7168, 1, 2),
    Case("hit", 7168, 4, 2),
    Case("hit", 7168, 8, 2),
    Case("miss", 128, 1, 128),
    Case("miss", 128, 8, 128),
    Case("miss", 512, 1, 512),
    Case("miss", 512, 4, 512),
    Case("miss", 512, 8, 512),
    Case("miss", 2048, 1, 2048),
    Case("miss", 2048, 4, 2048),
    Case("miss", 2048, 8, 2048),
    Case("miss", 7168, 4, 7168),
]


def token_ids(length: int, seed: int, first: int | None = None) -> list[int]:
    if _TOKEN_BANK:
        offset = seed % len(_TOKEN_BANK)
        ids = [_TOKEN_BANK[(offset + index) % len(_TOKEN_BANK)] for index in range(length)]
    else:
        rng = random.Random(seed)
        ids = [rng.randrange(1000, 120000) for _ in range(length)]
    if ids and first is not None:
        ids[0] = first
    return ids


class Client:
    def __init__(self, base_url: str, timeout: float):
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
        ttft = None
        output_ids: list[int] = []
        meta: dict[str, Any] = {}
        itl_ms: list[float] = []
        last_token_time = None
        last_completion = 0
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
                completion = int(meta.get("completion_tokens", len(chunk_ids)))
                if chunk_ids and completion > last_completion:
                    now = time.perf_counter()
                    if ttft is None:
                        ttft = now - start
                    elif last_token_time is not None:
                        per_token_ms = (
                            (now - last_token_time) * 1000
                            / (completion - last_completion)
                        )
                        itl_ms.extend([per_token_ms] * (completion - last_completion))
                    last_token_time = now
                    last_completion = completion
                if len(chunk_ids) == completion:
                    output_ids = chunk_ids
                elif chunk_ids:
                    output_ids.extend(chunk_ids)
        latency = time.perf_counter() - start
        return {
            "latency_ms": latency * 1000,
            "ttft_ms": None if ttft is None else ttft * 1000,
            "output_ids": output_ids,
            "meta_info": meta,
            "itl_ms": itl_ms,
        }


def first_difference(lhs: list[int], rhs: list[int]) -> int | None:
    for index, (left, right) in enumerate(zip(lhs, rhs)):
        if left != right:
            return index
    return None if len(lhs) == len(rhs) else min(len(lhs), len(rhs))


def run_case(client: Client, case: Case, repeat: int, output_len: int) -> dict[str, Any]:
    case_seed = 100000 * repeat + 1009 * case.context + 37 * case.running_bs + case.suffix
    base = token_ids(case.context, case_seed, first=121000 + repeat * 100 + case.running_bs)
    suffix = token_ids(case.suffix, case_seed + 1)
    probe = base + suffix if case.cache == "hit" else base

    client.flush()
    if case.cache == "hit":
        client.generate(base, 1)
    reference = client.generate(probe, output_len)
    client.flush()
    warm = None
    if case.cache == "hit":
        warm = client.generate(base, 1)

    # Running prompts cannot share the first token with the probe in miss cases.
    running_prompts = []
    for index in range(case.running_bs):
        if case.cache == "hit":
            running_prompts.append(base + token_ids(8, case_seed + 100 + index))
        else:
            running_prompts.append(
                token_ids(
                    min(max(case.context, 256), 512),
                    case_seed + 100 + index,
                    first=125000 + index,
                )
            )

    with concurrent.futures.ThreadPoolExecutor(max_workers=case.running_bs + 1) as pool:
        running = [pool.submit(client.generate, ids, 256) for ids in running_prompts]
        time.sleep(0.08)
        under_load = client.generate(probe, output_len)
        running_results = [future.result() for future in running]

    cached = int(under_load["meta_info"].get("cached_tokens", 0))
    expected_hit = case.cache == "hit"
    cache_ok = cached >= case.context - 1 if expected_hit else cached == 0
    diff = first_difference(reference["output_ids"], under_load["output_ids"])
    return {
        "case": asdict(case),
        "name": case.name,
        "repeat": repeat,
        "reference_latency_ms": reference["latency_ms"],
        "reference_ttft_ms": reference["ttft_ms"],
        "probe_latency_ms": under_load["latency_ms"],
        "probe_ttft_ms": under_load["ttft_ms"],
        "cached_tokens": cached,
        "cache_ok": cache_ok,
        "output_match": diff is None,
        "first_output_difference": diff,
        "reference_output_ids": reference["output_ids"],
        "probe_output_ids": under_load["output_ids"],
        "warm_cached_tokens": None
        if warm is None
        else warm["meta_info"].get("cached_tokens", 0),
        "running_latency_ms": [item["latency_ms"] for item in running_results],
        "running_ttft_ms": [item["ttft_ms"] for item in running_results],
        "running_itl_p99_ms": [
            percentile(item["itl_ms"], 0.99) for item in running_results
        ],
        "running_itl_max_ms": [
            max(item["itl_ms"]) if item["itl_ms"] else None
            for item in running_results
        ],
        "spec_accept_length": under_load["meta_info"].get("spec_accept_length"),
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_cache: dict[str, list[dict[str, Any]]] = {"hit": [], "miss": []}
    for record in records:
        by_cache[record["case"]["cache"]].append(record)

    def stats(items: list[dict[str, Any]]) -> dict[str, Any]:
        slowdown = [x["probe_latency_ms"] / x["reference_latency_ms"] for x in items]
        return {
            "cases": len(items),
            "cache_checks_passed": sum(x["cache_ok"] for x in items),
            "output_matches": sum(x["output_match"] for x in items),
            "probe_latency_ms_median": statistics.median(x["probe_latency_ms"] for x in items),
            "probe_ttft_ms_median": statistics.median(x["probe_ttft_ms"] for x in items),
            "load_slowdown_median": statistics.median(slowdown),
            "load_slowdown_max": max(slowdown),
        }

    return {
        "overall_pass": all(x["cache_ok"] and x["output_match"] for x in records),
        "records": len(records),
        "by_cache": {key: stats(value) for key, value in by_cache.items() if value},
    }


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction)))
    return ordered[index]


def profile_workload(
    client: Client,
    kind: str,
    output_dir: Path,
    num_steps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    client.flush()
    base = token_ids(7168 if kind == "hit" else 2048, 88001, first=130001)
    if kind == "hit":
        client.generate(base, 1)

    profile_payload = {
        "output_dir": str(output_dir.resolve()),
        "num_steps": num_steps,
        "activities": ["CPU", "GPU"],
        "with_stack": False,
        "record_shapes": True,
        "profile_prefix": f"prefix-{kind}",
    }
    profile_result: dict[str, Any] = {}

    def start_profile() -> None:
        response = requests.post(
            f"{client.base_url}/start_profile",
            json=profile_payload,
            timeout=client.timeout,
        )
        response.raise_for_status()
        if response.content:
            try:
                profile_result.update(response.json())
            except requests.exceptions.JSONDecodeError:
                profile_result["start_response"] = response.text

    profile_thread = threading.Thread(target=start_profile, daemon=True)
    profile_thread.start()
    time.sleep(0.5)

    if kind == "hit":
        running_prompts = [base + token_ids(8, 89000 + i) for i in range(4)]
        probes = [base + token_ids(1 + i % 4, 90000 + i) for i in range(12)]
    else:
        running_prompts = [token_ids(512, 89000 + i, first=131000 + i) for i in range(4)]
        probes = [token_ids(2048, 90000 + i, first=132000 + i) for i in range(8)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        futures = [pool.submit(client.generate, ids, 192) for ids in running_prompts]
        time.sleep(0.05)
        for ids in probes:
            futures.append(pool.submit(client.generate, ids, 12))
            time.sleep(0.02)
        results = [future.result() for future in futures]
    profile_thread.join(timeout=client.timeout)
    if profile_thread.is_alive():
        raise TimeoutError("profiler did not finish after workload completion")
    stop = requests.post(f"{client.base_url}/stop_profile", timeout=client.timeout)
    profile_result["stop_status"] = stop.status_code
    profile_result["stop_response"] = stop.text
    traces = sorted(str(path) for path in output_dir.glob("*.trace.json.gz"))
    return {
        "kind": kind,
        "num_steps": num_steps,
        "request_count": len(results),
        "profile_response": profile_result,
        "traces": traces,
    }


def main() -> None:
    global _TOKEN_BANK

    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-len", type=int, default=24)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--cache", choices=("all", "hit", "miss"), default="all")
    parser.add_argument("--profile-kind", choices=("hit", "miss"))
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--profile-steps", type=int, default=80)
    parser.add_argument("--model-path", default="/workspace/models/Qwen3-4B")
    parser.add_argument(
        "--random-token-stress",
        action="store_true",
        help="Use uniform random vocabulary IDs instead of the default natural-text bank.",
    )
    parser.add_argument(
        "--case-name",
        action="append",
        help="Run only the named matrix case; may be supplied more than once.",
    )
    args = parser.parse_args()

    if not args.random_token_stress:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model_path)
        corpus = (
            "You are reviewing a high performance inference server. Explain how an "
            "asynchronous GPU pipeline preserves dependency ordering without forcing "
            "the host to synchronize. Discuss metadata construction, prefix caching, "
            "latency budgets, speculative verification, and correctness checks. Use "
            "precise engineering language and distinguish measured evidence from "
            "assumptions. The implementation should preserve the ordinary decode fast "
            "path while allowing long prefills to share execution fairly. "
        )
        _TOKEN_BANK = tokenizer.encode(corpus * 8, add_special_tokens=False)

    client = Client(args.url, args.timeout)
    if args.profile_kind:
        if args.profile_dir is None:
            parser.error("--profile-dir is required with --profile-kind")
        result = profile_workload(client, args.profile_kind, args.profile_dir, args.profile_steps)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return

    selected = [case for case in CASES if args.cache == "all" or case.cache == args.cache]
    if args.case_name:
        wanted = set(args.case_name)
        selected = [case for case in selected if case.name in wanted]
        missing = wanted - {case.name for case in selected}
        if missing:
            parser.error(f"unknown or cache-filtered case names: {sorted(missing)}")
    records = []
    for repeat in range(args.repeats):
        for case in selected:
            record = run_case(client, case, repeat, args.output_len)
            records.append(record)
            print(
                f"{case.name:28s} repeat={repeat} cache={record['cached_tokens']:5d} "
                f"match={record['output_match']} ttft={record['probe_ttft_ms']:.1f} ms "
                f"e2e={record['probe_latency_ms']:.1f} ms",
                flush=True,
            )
    result = {
        "configuration": {
            "url": args.url,
            "repeats": args.repeats,
            "output_len": args.output_len,
            "cache": args.cache,
        },
        "summary": aggregate(records),
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
