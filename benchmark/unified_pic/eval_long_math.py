"""Evaluate shifted long-context GSM8K or AIME prompts through SGLang HTTP."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from transformers import AutoTokenizer

_NUMBER_RE = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
_HASH_ANSWER_RE = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")
_GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
_AIME_2024_REVISION = "2fe88a2f1091d5048c0f36abc874fb997b3dd99a"


@dataclass(frozen=True, slots=True)
class MathCase:
    case_id: int
    problem: str
    label: str


@dataclass(frozen=True, slots=True)
class MathResult:
    case_id: int
    label: str
    prediction: str | None
    correct: bool
    parse_mode: str
    prefix_tokens: int
    prompt_tokens: int
    output_tokens: int
    latency_s: float
    status_code: int
    text: str
    output_token_ids: tuple[int, ...]
    finish_reason: Any
    error: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--model-path", default="/workspace/models/DeepSeek-V2-Lite-AWQ"
    )
    parser.add_argument("--task", choices=("gsm8k", "aime2024"), required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--case-ids",
        type=lambda value: tuple(int(item) for item in value.split(",")),
        help="Optional comma-separated case IDs selected after --limit.",
    )
    parser.add_argument("--num-shots", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--timeout", type=float, default=600.0)
    return parser.parse_args()


def normalize_number(value: str) -> str | None:
    try:
        decimal = Decimal(value.replace(",", ""))
    except InvalidOperation:
        return None
    if decimal == decimal.to_integral_value():
        return str(int(decimal))
    return format(decimal.normalize(), "f")


def extract_prediction(text: str) -> tuple[str | None, str]:
    hashed = _HASH_ANSWER_RE.findall(text)
    if hashed:
        return normalize_number(hashed[-1]), "hash"
    numbers = _NUMBER_RE.findall(text)
    if numbers:
        return normalize_number(numbers[-1]), "last_number"
    return None, "invalid"


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


def load_cases(task: str, limit: int | None) -> list[MathCase]:
    if task == "gsm8k":
        dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="test",
            revision=_GSM8K_REVISION,
        )
        default_limit = 100
        rows = dataset.select(range(min(limit or default_limit, len(dataset))))
        return [
            MathCase(
                case_id=index,
                problem=str(row["question"]),
                label=normalize_number(_HASH_ANSWER_RE.findall(row["answer"])[-1])
                or "",
            )
            for index, row in enumerate(rows)
        ]

    dataset = load_dataset(
        "HuggingFaceH4/aime_2024",
        split="train",
        revision=_AIME_2024_REVISION,
    )
    default_limit = len(dataset)
    rows = dataset.select(range(min(limit or default_limit, len(dataset))))
    return [
        MathCase(
            case_id=index,
            problem=str(row["problem"]),
            label=normalize_number(str(row["answer"])) or "",
        )
        for index, row in enumerate(rows)
    ]


def build_shared_context(num_shots: int) -> str:
    train = load_dataset(
        "openai/gsm8k",
        "main",
        split="train",
        revision=_GSM8K_REVISION,
    )
    if not 1 <= num_shots <= len(train):
        raise ValueError(f"num_shots must be in [1, {len(train)}]")
    marker_sentence = (
        "The following worked examples define the required reasoning and answer "
        "format. Treat all request metadata before this sentence as irrelevant "
        "bookkeeping. "
    )
    parts = [marker_sentence * 8, "\n\n"]
    parts.append(
        "Solve each problem carefully. End every final answer on a new line using "
        "exactly #### followed by one number.\n\n"
    )
    for row in train.select(range(num_shots)):
        parts.extend(
            (
                "Question: ",
                str(row["question"]),
                "\nSolution: ",
                str(row["answer"]),
                "\n\n",
            )
        )
    return "".join(parts)


def build_prefix(case_id: int) -> str:
    level = case_id % 8 + 1
    identifiers = " ".join(f"req{case_id:03d}-{i:03d}" for i in range(level * 12))
    return (
        f"Metadata-{case_id:03d}: Ignore these bookkeeping identifiers; they do "
        f"not alter the mathematics: {identifiers}\n\n"
    )


def build_prompt(
    tokenizer: Any, case: MathCase, shared_context: str
) -> tuple[tuple[int, ...], int]:
    prefix = build_prefix(case.case_id)
    text = prefix + shared_context + "Question: " + case.problem + "\nSolution:"
    return (
        tuple(tokenizer.encode(text, add_special_tokens=True)),
        len(tokenizer.encode(prefix, add_special_tokens=True)),
    )


def issue_request(
    *,
    base_url: str,
    case: MathCase,
    prompt: tuple[int, ...],
    prefix_tokens: int,
    max_new_tokens: int,
    timeout: float,
    barrier: threading.Barrier | None,
) -> MathResult:
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
                    "stop": ["Question", "Assistant:", "<|separator|>"],
                },
                "return_logprob": True,
                "logprob_start_len": len(prompt),
                "top_logprobs_num": 0,
            },
            timeout=timeout,
        )
        latency = time.perf_counter() - started
        response = http_response.json()
        text = str(response.get("text", ""))
        prediction, parse_mode = extract_prediction(text)
        output_token_ids = extract_output_token_ids(response)
        meta = response.get("meta_info") or {}
        return MathResult(
            case_id=case.case_id,
            label=case.label,
            prediction=prediction,
            correct=prediction == case.label,
            parse_mode=parse_mode,
            prefix_tokens=prefix_tokens,
            prompt_tokens=len(prompt),
            output_tokens=len(output_token_ids),
            latency_s=latency,
            status_code=http_response.status_code,
            text=text,
            output_token_ids=output_token_ids,
            finish_reason=meta.get("finish_reason"),
        )
    except (requests.RequestException, ValueError) as exc:
        return MathResult(
            case_id=case.case_id,
            label=case.label,
            prediction=None,
            correct=False,
            parse_mode="error",
            prefix_tokens=prefix_tokens,
            prompt_tokens=len(prompt),
            output_tokens=0,
            latency_s=time.perf_counter() - started,
            status_code=0,
            text="",
            output_token_ids=(),
            finish_reason=None,
            error=f"{type(exc).__name__}: {exc}",
        )


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    cases = load_cases(args.task, args.limit)
    if args.case_ids is not None:
        cases_by_id = {case.case_id: case for case in cases}
        missing = sorted(set(args.case_ids) - cases_by_id.keys())
        if missing:
            raise ValueError(f"case IDs outside the loaded range: {missing}")
        cases = [cases_by_id[case_id] for case_id in args.case_ids]
    shared_context = build_shared_context(args.num_shots)
    prompts = {
        case.case_id: build_prompt(tokenizer, case, shared_context) for case in cases
    }

    model_info_response = requests.get(
        f"{args.base_url.rstrip('/')}/model_info", timeout=args.timeout
    )
    model_info_response.raise_for_status()

    warm_case = MathCase(case_id=10_000, problem="What is 2 + 3?", label="5")
    warm_prompt, warm_prefix_tokens = build_prompt(tokenizer, warm_case, shared_context)
    warm = issue_request(
        base_url=args.base_url,
        case=warm_case,
        prompt=warm_prompt,
        prefix_tokens=warm_prefix_tokens,
        max_new_tokens=min(args.max_new_tokens, 64),
        timeout=args.timeout,
        barrier=None,
    )
    if warm.status_code != 200:
        raise RuntimeError(f"warm request failed: {warm}")

    started = time.perf_counter()
    results = []
    for batch_start in range(0, len(cases), args.concurrency):
        batch_cases = cases[batch_start : batch_start + args.concurrency]
        barrier = threading.Barrier(len(batch_cases))
        with ThreadPoolExecutor(max_workers=len(batch_cases)) as executor:
            futures = [
                executor.submit(
                    issue_request,
                    base_url=args.base_url,
                    case=case,
                    prompt=prompts[case.case_id][0],
                    prefix_tokens=prompts[case.case_id][1],
                    max_new_tokens=args.max_new_tokens,
                    timeout=args.timeout,
                    barrier=barrier,
                )
                for case in batch_cases
            ]
            results.extend(future.result() for future in as_completed(futures))
    wall_time = time.perf_counter() - started
    results.sort(key=lambda result: result.case_id)

    successful = [result for result in results if result.status_code == 200]
    prompt_lengths = [result.prompt_tokens for result in results]
    latencies = [result.latency_s for result in results]
    report = {
        "label": args.label,
        "task": args.task,
        "model_info": model_info_response.json(),
        "workload": {
            "num_cases": len(cases),
            "case_ids": [case.case_id for case in cases],
            "num_shots": args.num_shots,
            "concurrency": args.concurrency,
            "max_new_tokens": args.max_new_tokens,
            "shared_context_tokens": len(
                tokenizer.encode(shared_context, add_special_tokens=False)
            ),
            "prompt_tokens_min": min(prompt_lengths),
            "prompt_tokens_mean": statistics.mean(prompt_lengths),
            "prompt_tokens_max": max(prompt_lengths),
        },
        "warmup": asdict(warm),
        "results": [asdict(result) for result in results],
        "summary": {
            "accuracy": sum(result.correct for result in results) / len(results),
            "correct": sum(result.correct for result in results),
            "successful": len(successful),
            "failed": len(results) - len(successful),
            "invalid": sum(result.prediction is None for result in results),
            "truncated": sum(
                isinstance(result.finish_reason, dict)
                and result.finish_reason.get("type") == "length"
                for result in results
            ),
            "output_tokens": sum(result.output_tokens for result in results),
            "wall_time_s": wall_time,
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
