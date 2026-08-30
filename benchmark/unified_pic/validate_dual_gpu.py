"""End-to-end shifted-content workload for a two-GPU SGLang server."""

from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests


@dataclass(frozen=True, slots=True)
class CaseResult:
    case_id: int
    prefix_tokens: int
    prompt_tokens: int
    latency_s: float
    status_code: int
    text: str
    output_token_ids: tuple[int, ...]
    finish_reason: Any
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--shared-tokens", type=int, default=1536)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument(
        "--request-mode",
        choices=("concurrent", "batch", "sequential"),
        default="concurrent",
        help=(
            "Use independent synchronized requests, one batched request, or "
            "a sequential correctness oracle."
        ),
    )
    return parser.parse_args()


def build_tokens(shared_tokens: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    marker = tuple(50000 + i for i in range(64))
    shared = tuple(100 + ((i * 104729 + 1543) % 60000) for i in range(shared_tokens))
    return marker, shared


def build_prompt(
    case_id: int,
    *,
    prefix_tokens: int,
    marker: tuple[int, ...],
    shared: tuple[int, ...],
) -> tuple[int, ...]:
    # The first token differs for every request, forcing an exact-prefix miss.
    prefix = tuple(
        100 + ((case_id * 8191 + i * 131071 + 17) % 60000) for i in range(prefix_tokens)
    )
    return prefix + marker + shared


def extract_output_token_ids(response: dict[str, Any]) -> tuple[int, ...]:
    meta = response.get("meta_info") or {}
    token_logprobs = meta.get("output_token_logprobs") or ()
    result = []
    for item in token_logprobs:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            result.append(int(item[1]))
        elif isinstance(item, dict) and "token_id" in item:
            result.append(int(item["token_id"]))
    return tuple(result)


def issue_request(
    *,
    base_url: str,
    case_id: int,
    prompt: tuple[int, ...],
    prefix_tokens: int,
    max_new_tokens: int,
    timeout: float,
    barrier: threading.Barrier | None,
) -> CaseResult:
    if barrier is not None:
        barrier.wait()
    started = time.perf_counter()
    try:
        http_response = requests.post(
            f"{base_url.rstrip('/')}/generate",
            json={
                "input_ids": list(prompt),
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
                "return_logprob": True,
                "logprob_start_len": len(prompt),
                "top_logprobs_num": 0,
            },
            timeout=timeout,
        )
        latency = time.perf_counter() - started
        response = http_response.json()
        meta = response.get("meta_info") or {}
        return CaseResult(
            case_id=case_id,
            prefix_tokens=prefix_tokens,
            prompt_tokens=len(prompt),
            latency_s=latency,
            status_code=http_response.status_code,
            text=str(response.get("text", "")),
            output_token_ids=extract_output_token_ids(response),
            finish_reason=meta.get("finish_reason"),
        )
    except (requests.RequestException, ValueError) as exc:
        return CaseResult(
            case_id=case_id,
            prefix_tokens=prefix_tokens,
            prompt_tokens=len(prompt),
            latency_s=time.perf_counter() - started,
            status_code=0,
            text="",
            output_token_ids=(),
            finish_reason=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def issue_batch_request(
    *,
    base_url: str,
    prompts: list[tuple[int, ...]],
    prefix_lengths: list[int],
    max_new_tokens: int,
    timeout: float,
) -> list[CaseResult]:
    started = time.perf_counter()
    http_response = requests.post(
        f"{base_url.rstrip('/')}/generate",
        json={
            "input_ids": [list(prompt) for prompt in prompts],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
            "return_logprob": True,
            "logprob_start_len": [len(prompt) for prompt in prompts],
            "top_logprobs_num": 0,
        },
        timeout=timeout,
    )
    latency = time.perf_counter() - started
    http_response.raise_for_status()
    responses = http_response.json()
    if not isinstance(responses, list) or len(responses) != len(prompts):
        raise ValueError(
            f"expected {len(prompts)} batch responses, got {type(responses).__name__}"
        )

    results = []
    for case_id, (prompt, prefix_tokens, response) in enumerate(
        zip(prompts, prefix_lengths, responses, strict=True), start=1
    ):
        meta = response.get("meta_info") or {}
        results.append(
            CaseResult(
                case_id=case_id,
                prefix_tokens=prefix_tokens,
                prompt_tokens=len(prompt),
                latency_s=latency,
                status_code=http_response.status_code,
                text=str(response.get("text", "")),
                output_token_ids=extract_output_token_ids(response),
                finish_reason=meta.get("finish_reason"),
            )
        )
    return results


def main() -> None:
    args = parse_args()
    marker, shared = build_tokens(args.shared_tokens)

    info_response = requests.get(
        f"{args.base_url.rstrip('/')}/model_info", timeout=args.timeout
    )
    info_response.raise_for_status()
    model_info = info_response.json()

    warm_prefix_tokens = 96
    warm_prompt = build_prompt(
        0,
        prefix_tokens=warm_prefix_tokens,
        marker=marker,
        shared=shared,
    )
    warm = issue_request(
        base_url=args.base_url,
        case_id=0,
        prompt=warm_prompt,
        prefix_tokens=warm_prefix_tokens,
        max_new_tokens=args.max_new_tokens,
        timeout=args.timeout,
        barrier=None,
    )
    if warm.status_code != 200:
        raise RuntimeError(f"warm request failed: {warm}")

    prefix_lengths = [32 + case_id * 37 for case_id in range(1, args.concurrency + 1)]
    prompts = [
        build_prompt(
            case_id,
            prefix_tokens=prefix_tokens,
            marker=marker,
            shared=shared,
        )
        for case_id, prefix_tokens in enumerate(prefix_lengths, start=1)
    ]
    if args.request_mode == "batch":
        results = issue_batch_request(
            base_url=args.base_url,
            prompts=prompts,
            prefix_lengths=prefix_lengths,
            max_new_tokens=args.max_new_tokens,
            timeout=args.timeout,
        )
    elif args.request_mode == "concurrent":
        barrier = threading.Barrier(args.concurrency)
        futures = []
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            for case_id, (prompt, prefix_tokens) in enumerate(
                zip(prompts, prefix_lengths, strict=True), start=1
            ):
                futures.append(
                    executor.submit(
                        issue_request,
                        base_url=args.base_url,
                        case_id=case_id,
                        prompt=prompt,
                        prefix_tokens=prefix_tokens,
                        max_new_tokens=args.max_new_tokens,
                        timeout=args.timeout,
                        barrier=barrier,
                    )
                )
            results = sorted(
                (future.result() for future in as_completed(futures)),
                key=lambda result: result.case_id,
            )
    else:
        results = [
            issue_request(
                base_url=args.base_url,
                case_id=case_id,
                prompt=prompt,
                prefix_tokens=prefix_tokens,
                max_new_tokens=args.max_new_tokens,
                timeout=args.timeout,
                barrier=None,
            )
            for case_id, (prompt, prefix_tokens) in enumerate(
                zip(prompts, prefix_lengths, strict=True), start=1
            )
        ]

    latencies = [result.latency_s for result in results]
    report = {
        "label": args.label,
        "model_info": model_info,
        "workload": {
            "concurrency": args.concurrency,
            "shared_tokens": args.shared_tokens,
            "marker_tokens": len(marker),
            "max_new_tokens": args.max_new_tokens,
            "request_mode": args.request_mode,
        },
        "warmup": asdict(warm),
        "results": [asdict(result) for result in results],
        "summary": {
            "successful": sum(result.status_code == 200 for result in results),
            "failed": sum(result.status_code != 200 for result in results),
            "latency_mean_s": statistics.mean(latencies),
            "latency_median_s": statistics.median(latencies),
            "latency_max_s": max(latencies),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
