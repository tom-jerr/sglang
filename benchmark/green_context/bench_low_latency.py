#!/usr/bin/env python3
"""Benchmark P/D multiplexing and CUDA green contexts on one GPU.

The homogeneous sweep measures normal serving behavior. The interference
sweep starts short requests first, waits until they enter decode, and then
injects long prefills. This directly measures the latency-sensitive behavior
green contexts are designed to protect.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import aiohttp


@dataclass
class RequestMetrics:
    role: str
    prompt_tokens: int
    requested_output_tokens: int
    start_time: float = 0.0
    end_time: float = 0.0
    ttft: float | None = None
    itl: list[float] = field(default_factory=list)
    output_tokens: int = 0
    success: bool = False
    error: str = ""

    @property
    def e2e(self) -> float | None:
        if not self.success:
            return None
        return self.end_time - self.start_time


def percentile(values: Iterable[float], quantile: float) -> float | None:
    values = sorted(values)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summarize(records: list[RequestMetrics]) -> dict:
    successes = [record for record in records if record.success]
    ttft = [record.ttft for record in successes if record.ttft is not None]
    itl = [value for record in successes for value in record.itl]
    e2e = [record.e2e for record in successes if record.e2e is not None]
    started = [record.start_time for record in records]
    ended = [record.end_time for record in records if record.end_time]
    wall_time = max(ended) - min(started) if started and ended else 0.0
    output_tokens = sum(record.output_tokens for record in successes)
    return {
        "requests": len(records),
        "successful_requests": len(successes),
        "success_rate": len(successes) / len(records) if records else 0.0,
        "ttft_ms_mean": statistics.fmean(ttft) * 1000 if ttft else None,
        "ttft_ms_p50": percentile(ttft, 0.50) * 1000 if ttft else None,
        "ttft_ms_p95": percentile(ttft, 0.95) * 1000 if ttft else None,
        "ttft_ms_p99": percentile(ttft, 0.99) * 1000 if ttft else None,
        "itl_ms_mean": statistics.fmean(itl) * 1000 if itl else None,
        "itl_ms_p50": percentile(itl, 0.50) * 1000 if itl else None,
        "itl_ms_p95": percentile(itl, 0.95) * 1000 if itl else None,
        "itl_ms_p99": percentile(itl, 0.99) * 1000 if itl else None,
        "e2e_ms_mean": statistics.fmean(e2e) * 1000 if e2e else None,
        "e2e_ms_p50": percentile(e2e, 0.50) * 1000 if e2e else None,
        "e2e_ms_p95": percentile(e2e, 0.95) * 1000 if e2e else None,
        "e2e_ms_p99": percentile(e2e, 0.99) * 1000 if e2e else None,
        "output_tokens_per_second": output_tokens / wall_time if wall_time else 0.0,
        "wall_time_seconds": wall_time,
    }


def make_input_ids(length: int, seed: int) -> list[int]:
    # Different first tokens prevent the radix cache from sharing long prefixes.
    first = 1000 + seed % 100000
    return [first] + [
        1000 + ((seed * 7919 + index * 104729) % 100000) for index in range(1, length)
    ]


async def stream_generate(
    session: aiohttp.ClientSession,
    url: str,
    prompt_tokens: int,
    output_tokens: int,
    seed: int,
    role: str,
    first_token_event: asyncio.Event | None = None,
) -> RequestMetrics:
    result = RequestMetrics(
        role=role,
        prompt_tokens=prompt_tokens,
        requested_output_tokens=output_tokens,
    )
    payload = {
        "input_ids": make_input_ids(prompt_tokens, seed),
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }
    result.start_time = time.perf_counter()
    last_token_time = result.start_time
    last_output_tokens = 0
    try:
        async with session.post(f"{url}/generate", json=payload) as response:
            if response.status != 200:
                result.error = f"HTTP {response.status}: {await response.text()}"
                return result
            async for line in response.content:
                line = line.strip()
                if not line or line == b"data: [DONE]":
                    continue
                if line.startswith(b"data: "):
                    line = line[6:]
                data = json.loads(line)
                meta = data.get("meta_info") or {}
                cumulative_tokens = int(meta.get("completion_tokens") or 0)
                if cumulative_tokens <= last_output_tokens:
                    continue

                now = time.perf_counter()
                new_tokens = cumulative_tokens - last_output_tokens
                if result.ttft is None:
                    result.ttft = now - result.start_time
                    if first_token_event is not None:
                        first_token_event.set()
                    if new_tokens > 1:
                        result.itl.extend(
                            [(now - result.start_time) / new_tokens] * (new_tokens - 1)
                        )
                else:
                    result.itl.extend(
                        [(now - last_token_time) / new_tokens] * new_tokens
                    )
                last_token_time = now
                last_output_tokens = cumulative_tokens

        result.output_tokens = last_output_tokens
        result.success = last_output_tokens > 0
    except Exception as exc:
        result.error = repr(exc)
    finally:
        result.end_time = time.perf_counter()
        if first_token_event is not None and not first_token_event.is_set():
            first_token_event.set()
    return result


async def flush_cache(session: aiohttp.ClientSession, url: str) -> None:
    async with session.post(f"{url}/flush_cache") as response:
        if response.status != 200:
            raise RuntimeError(f"flush_cache failed: {await response.text()}")


async def run_homogeneous_case(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    prompt_tokens: int,
    concurrency: int,
    seed_base: int,
) -> list[RequestMetrics]:
    await flush_cache(session, args.url)
    semaphore = asyncio.Semaphore(concurrency)

    async def one_request(index: int) -> RequestMetrics:
        if args.arrival_interval_ms:
            await asyncio.sleep(index * args.arrival_interval_ms / 1000)
        async with semaphore:
            return await stream_generate(
                session,
                args.url,
                prompt_tokens,
                args.output_tokens,
                seed_base + index,
                "homogeneous",
            )

    request_count = max(args.min_requests, concurrency * args.requests_per_worker)
    return await asyncio.gather(*(one_request(index) for index in range(request_count)))


async def run_interference_trial(
    session: aiohttp.ClientSession,
    args: argparse.Namespace,
    long_prompt_tokens: int,
    foreground_concurrency: int,
    seed_base: int,
) -> tuple[list[RequestMetrics], list[RequestMetrics]]:
    await flush_cache(session, args.url)
    first_token_events = [asyncio.Event() for _ in range(foreground_concurrency)]
    foreground_tasks = [
        asyncio.create_task(
            stream_generate(
                session,
                args.url,
                args.foreground_prompt_tokens,
                args.foreground_output_tokens,
                seed_base + index,
                "foreground",
                first_token_events[index],
            )
        )
        for index in range(foreground_concurrency)
    ]
    await asyncio.wait_for(
        asyncio.gather(*(event.wait() for event in first_token_events)),
        timeout=args.first_token_timeout,
    )
    background_tasks = [
        asyncio.create_task(
            stream_generate(
                session,
                args.url,
                long_prompt_tokens,
                args.background_output_tokens,
                seed_base + 10000 + index,
                "background",
            )
        )
        for index in range(args.background_concurrency)
    ]
    foreground = await asyncio.gather(*foreground_tasks)
    background = await asyncio.gather(*background_tasks)
    return foreground, background


async def run(args: argparse.Namespace) -> dict:
    timeout = aiohttp.ClientTimeout(total=args.request_timeout)
    connector = aiohttp.TCPConnector(limit=0)
    cases = []
    all_records = []
    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        # Warm both model paths and HTTP streaming before collecting data.
        warmup = await stream_generate(session, args.url, 32, 8, 42, "warmup")
        if not warmup.success:
            raise RuntimeError(f"Warmup failed: {warmup.error}")

        seed = 100000
        if args.suite in ("homogeneous", "both"):
            for prompt_tokens in args.prompt_tokens:
                for concurrency in args.concurrency:
                    records = await run_homogeneous_case(
                        session, args, prompt_tokens, concurrency, seed
                    )
                    seed += len(records) + 1
                    case = {
                        "suite": "homogeneous",
                        "prompt_tokens": prompt_tokens,
                        "concurrency": concurrency,
                        "summary": summarize(records),
                    }
                    cases.append(case)
                    all_records.extend(records)
                    print(json.dumps({"variant": args.variant, **case}), flush=True)
                    write_result(args, cases, all_records)

        if args.suite in ("interference", "both"):
            for long_prompt_tokens in args.long_prompt_tokens:
                for concurrency in args.concurrency:
                    foreground = []
                    background = []
                    for _ in range(args.interference_repeats):
                        trial_foreground, trial_background = (
                            await run_interference_trial(
                                session,
                                args,
                                long_prompt_tokens,
                                concurrency,
                                seed,
                            )
                        )
                        seed += concurrency + args.background_concurrency + 1
                        foreground.extend(trial_foreground)
                        background.extend(trial_background)
                    case = {
                        "suite": "interference",
                        "long_prompt_tokens": long_prompt_tokens,
                        "foreground_concurrency": concurrency,
                        "background_concurrency": args.background_concurrency,
                        "foreground_summary": summarize(foreground),
                        "background_summary": summarize(background),
                    }
                    cases.append(case)
                    all_records.extend(foreground)
                    all_records.extend(background)
                    print(json.dumps({"variant": args.variant, **case}), flush=True)
                    write_result(args, cases, all_records)

    return build_result(args, cases, all_records)


def build_result(
    args: argparse.Namespace,
    cases: list[dict],
    records: list[RequestMetrics],
) -> dict:
    return {
        "metadata": {
            "variant": args.variant,
            "url": args.url,
            "suite": args.suite,
            "prompt_tokens": args.prompt_tokens,
            "long_prompt_tokens": args.long_prompt_tokens,
            "concurrency": args.concurrency,
            "output_tokens": args.output_tokens,
            "foreground_prompt_tokens": args.foreground_prompt_tokens,
            "foreground_output_tokens": args.foreground_output_tokens,
            "background_output_tokens": args.background_output_tokens,
            "background_concurrency": args.background_concurrency,
            "interference_repeats": args.interference_repeats,
            "timestamp": time.time(),
        },
        "cases": cases,
        "records": [asdict(record) | {"e2e": record.e2e} for record in records],
    }


def write_result(
    args: argparse.Namespace,
    cases: list[dict],
    records: list[RequestMetrics],
) -> None:
    """Atomically checkpoint completed cases so a long sweep remains reusable."""
    result = build_result(args, cases, records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--variant", required=True)
    parser.add_argument(
        "--suite", choices=("homogeneous", "interference", "both"), default="both"
    )
    parser.add_argument(
        "--prompt-tokens", type=int, nargs="+", default=[128, 4096, 8192, 16384]
    )
    parser.add_argument(
        "--long-prompt-tokens", type=int, nargs="+", default=[4096, 8192, 16384]
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--min-requests", type=int, default=8)
    parser.add_argument("--requests-per-worker", type=int, default=1)
    parser.add_argument("--arrival-interval-ms", type=float, default=5.0)
    parser.add_argument("--foreground-prompt-tokens", type=int, default=32)
    parser.add_argument("--foreground-output-tokens", type=int, default=64)
    parser.add_argument("--background-output-tokens", type=int, default=8)
    parser.add_argument("--background-concurrency", type=int, default=1)
    parser.add_argument("--interference-repeats", type=int, default=3)
    parser.add_argument("--first-token-timeout", type=float, default=120.0)
    parser.add_argument("--request-timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = asyncio.run(run(args))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
