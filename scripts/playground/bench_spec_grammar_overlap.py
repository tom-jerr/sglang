"""
Scenario benchmark for speculative decoding + constrained decoding overlap.

It launches an SGLang server with speculative decoding enabled, runs a small set
of representative constrained-generation scenarios, then summarizes the
spec-grammar timing histograms collected from the Prometheus `/metrics` endpoint.

Example:
python3 scripts/playground/bench_spec_grammar_overlap.py \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --draft-model-path lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B \
  --speculative-algorithm EAGLE3 \
  --grammar-backend xgrammar
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

try:
    import jsonschema
except ImportError:
    jsonschema = None


PROM_METRIC_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)$"
)
LE_LABEL_RE = re.compile(r'le="([^"]+)"')

METRIC_NAMES = (
    "sglang:spec_grammar_cpu_copy_time_seconds",
    "sglang:spec_grammar_cpu_mask_time_seconds",
    "sglang:spec_grammar_h2d_time_seconds",
    "sglang:spec_grammar_verify_gpu_time_seconds",
    "sglang:spec_grammar_uncovered_cpu_tail_seconds",
)
TAIL_METRIC = "sglang:spec_grammar_uncovered_cpu_tail_seconds"

DEFAULT_TARGET_MODEL_EAGLE3 = "meta-llama/Llama-3.1-8B-Instruct"
DEFAULT_DRAFT_MODEL_EAGLE3 = "lmsys/sglang-EAGLE3-LLaMA3.1-Instruct-8B"
DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600

SIMPLE_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
        "population_millions": {"type": "integer"},
        "landmarks": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["city", "country", "population_millions", "landmarks"],
    "additionalProperties": False,
}

COMPLEX_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "trip_id": {"type": "string"},
        "traveler": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "membership": {"type": "string", "enum": ["silver", "gold", "platinum"]},
                "preferences": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 3,
                    "maxItems": 5,
                },
            },
            "required": ["name", "membership", "preferences"],
            "additionalProperties": False,
        },
        "legs": {
            "type": "array",
            "minItems": 2,
            "maxItems": 3,
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "day": {"type": "integer"},
                    "hotel": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "nights": {"type": "integer"},
                        },
                        "required": ["name", "nights"],
                        "additionalProperties": False,
                    },
                    "activities": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 2,
                        "maxItems": 4,
                    },
                },
                "required": ["from", "to", "day", "hotel", "activities"],
                "additionalProperties": False,
            },
        },
        "budget": {
            "type": "object",
            "properties": {
                "currency": {"type": "string", "enum": ["USD", "EUR", "JPY"]},
                "estimated_total": {"type": "integer"},
                "breakdown": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {"type": "string"},
                            "amount": {"type": "integer"},
                        },
                        "required": ["category", "amount"],
                        "additionalProperties": False,
                    },
                    "minItems": 3,
                    "maxItems": 4,
                },
            },
            "required": ["currency", "estimated_total", "breakdown"],
            "additionalProperties": False,
        },
    },
    "required": ["trip_id", "traveler", "legs", "budget"],
    "additionalProperties": False,
}

SIMPLE_REGEX = r"^[A-Z][a-z]{3,10}(, [A-Z][a-z]{3,10}){7}$"
COMPLEX_REGEX = (
    r'^\{"id":"[A-Z0-9]{6}","status":"(open|closed|pending)",'
    r'"scores":\[(100|[1-9]?\d),(100|[1-9]?\d),(100|[1-9]?\d)\],'
    r'"tags":\["[a-z]{3,8}"(,"[a-z]{3,8}"){2}\]\}$'
)

JSON_PROMPT = (
    "Return exactly one compact JSON object for a fictional travel or city profile. "
    "Do not add Markdown, comments, or prose."
)
REGEX_PROMPT_SIMPLE = (
    "Output only a comma-separated list of eight capitalized city names."
)
REGEX_PROMPT_COMPLEX = (
    "Output only a compact JSON-like record with id, status, scores, and tags."
)


@dataclass(frozen=True)
class Scenario:
    name: str
    batch_size: int
    measure_batches: int
    warmup_batches: int
    topk: int
    constraint_kind: str
    complexity: str
    prompt: str
    sampling_params: dict[str, Any]


@dataclass
class HistogramSnapshot:
    count: float = 0.0
    sum_value: float = 0.0
    buckets: dict[float, float] | None = None

    def __post_init__(self) -> None:
        if self.buckets is None:
            self.buckets = {}

    def delta(self, before: "HistogramSnapshot") -> "HistogramSnapshot":
        return HistogramSnapshot(
            count=max(0.0, self.count - before.count),
            sum_value=max(0.0, self.sum_value - before.sum_value),
            buckets={
                key: max(0.0, self.buckets.get(key, 0.0) - before.buckets.get(key, 0.0))
                for key in set(self.buckets) | set(before.buckets)
            },
        )


@dataclass
class MetricSummary:
    count: int
    total_seconds: float
    mean_ms: float
    p50_ms: float
    p95_ms: float
    nonzero_fraction: Optional[float] = None


def find_available_port(base_port: int) -> int:
    port = base_port
    while True:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1


def kill_process_tree(pid: int) -> None:
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return


def wait_for_server_health(
    process: subprocess.Popen[Any], base_url: str, timeout: float
) -> None:
    deadline = time.time() + timeout
    last_error = "server did not become healthy"
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited early with code {process.returncode}")
        try:
            response = requests.get(f"{base_url}/health_generate", timeout=5)
            if response.status_code == 200:
                return
            last_error = f"health endpoint returned {response.status_code}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"Timed out waiting for server health: {last_error}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_TARGET_MODEL_EAGLE3)
    parser.add_argument("--draft-model-path", type=str, default=DEFAULT_DRAFT_MODEL_EAGLE3)
    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        default="EAGLE3",
        choices=["EAGLE", "EAGLE3", "STANDALONE"],
    )
    parser.add_argument("--grammar-backend", type=str, default="xgrammar")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--mem-fraction-static", type=float, default=0.7)
    parser.add_argument("--speculative-num-steps", type=int, default=5)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=64)
    parser.add_argument("--small-batch-size", type=int, default=1)
    parser.add_argument("--large-batch-size", type=int, default=16)
    parser.add_argument("--compare-batch-size", type=int, default=8)
    parser.add_argument("--greedy-topk", type=int, default=1)
    parser.add_argument("--high-topk", type=int, default=8)
    parser.add_argument("--compare-topk", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--measure-batches", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--base-port", type=int, default=26000)
    parser.add_argument("--server-timeout", type=float, default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH)
    parser.add_argument("--log-timing", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--output-jsonl", type=str, default=None)
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    parser.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional list of scenario names to run.",
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="Additional raw server args, repeated as needed.",
    )
    return parser.parse_args()


def build_scenarios(args: argparse.Namespace) -> list[Scenario]:
    shared = {
        "measure_batches": args.measure_batches,
        "warmup_batches": args.warmup_batches,
    }
    scenarios = [
        Scenario(
            name="small_batch_greedy_json_simple",
            batch_size=args.small_batch_size,
            topk=args.greedy_topk,
            constraint_kind="json",
            complexity="simple",
            prompt=JSON_PROMPT,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "json_schema": SIMPLE_JSON_SCHEMA,
            },
            **shared,
        ),
        Scenario(
            name="large_batch_high_topk_json_simple",
            batch_size=args.large_batch_size,
            topk=args.high_topk,
            constraint_kind="json",
            complexity="simple",
            prompt=JSON_PROMPT,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "json_schema": SIMPLE_JSON_SCHEMA,
            },
            **shared,
        ),
        Scenario(
            name="json_simple",
            batch_size=args.compare_batch_size,
            topk=args.compare_topk,
            constraint_kind="json",
            complexity="simple",
            prompt=JSON_PROMPT,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "json_schema": SIMPLE_JSON_SCHEMA,
            },
            **shared,
        ),
        Scenario(
            name="json_complex",
            batch_size=args.compare_batch_size,
            topk=args.compare_topk,
            constraint_kind="json",
            complexity="complex",
            prompt=JSON_PROMPT,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "json_schema": COMPLEX_JSON_SCHEMA,
            },
            **shared,
        ),
        Scenario(
            name="regex_simple",
            batch_size=args.compare_batch_size,
            topk=args.compare_topk,
            constraint_kind="regex",
            complexity="simple",
            prompt=REGEX_PROMPT_SIMPLE,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "regex": SIMPLE_REGEX,
            },
            **shared,
        ),
        Scenario(
            name="regex_complex",
            batch_size=args.compare_batch_size,
            topk=args.compare_topk,
            constraint_kind="regex",
            complexity="complex",
            prompt=REGEX_PROMPT_COMPLEX,
            sampling_params={
                "temperature": 0,
                "max_new_tokens": args.max_new_tokens,
                "regex": COMPLEX_REGEX,
            },
            **shared,
        ),
    ]

    if args.scenarios is None:
        return scenarios

    scenario_map = {scenario.name: scenario for scenario in scenarios}
    unknown = sorted(set(args.scenarios) - set(scenario_map))
    if unknown:
        raise ValueError(f"Unknown scenarios: {', '.join(unknown)}")
    return [scenario_map[name] for name in args.scenarios]


def build_server_args(args: argparse.Namespace, scenario: Scenario) -> list[str]:
    server_args = [
        f"--speculative-algorithm={args.speculative_algorithm}",
        f"--speculative-draft-model-path={args.draft_model_path}",
        f"--speculative-num-steps={args.speculative_num_steps}",
        f"--speculative-eagle-topk={scenario.topk}",
        f"--speculative-num-draft-tokens={args.speculative_num_draft_tokens}",
        f"--mem-fraction-static={args.mem_fraction_static}",
        f"--cuda-graph-max-bs={max(1, scenario.batch_size)}",
        f"--max-running-requests={max(1, scenario.batch_size)}",
        f"--tp-size={args.tp_size}",
        f"--grammar-backend={args.grammar_backend}",
        "--enable-metrics",
    ]
    if args.trust_remote_code:
        server_args.append("--trust-remote-code")
    server_args.extend(args.server_arg)
    return server_args


def build_prompt_batch(prompt: str, batch_size: int, batch_index: int) -> str | list[str]:
    prompts = [
        f"Case {batch_index}-{sample_index}: {prompt}"
        for sample_index in range(batch_size)
    ]
    return prompts[0] if batch_size == 1 else prompts


def validate_output(scenario: Scenario, text: str) -> None:
    stripped = text.strip()
    if scenario.constraint_kind == "regex":
        pattern = SIMPLE_REGEX if scenario.complexity == "simple" else COMPLEX_REGEX
        if re.fullmatch(pattern, stripped) is None:
            raise ValueError(
                f"Scenario {scenario.name} produced text that does not match regex: {stripped!r}"
            )
        return

    obj = json.loads(stripped)
    if jsonschema is not None:
        schema = SIMPLE_JSON_SCHEMA if scenario.complexity == "simple" else COMPLEX_JSON_SCHEMA
        jsonschema.validate(obj, schema)


def run_generate_batch(
    base_url: str,
    scenario: Scenario,
    batch_index: int,
    request_timeout: float,
) -> dict[str, float]:
    payload = {
        "text": build_prompt_batch(scenario.prompt, scenario.batch_size, batch_index),
        "sampling_params": scenario.sampling_params,
        "stream": False,
    }
    response = requests.post(
        f"{base_url}/generate",
        json=payload,
        timeout=request_timeout,
    )
    response.raise_for_status()
    response_json = response.json()
    results = response_json if isinstance(response_json, list) else [response_json]

    completion_tokens = 0.0
    spec_verify_ct = 0.0
    e2e_latency = 0.0
    for result in results:
        validate_output(scenario, result["text"])
        meta_info = result.get("meta_info", {})
        completion_tokens += float(meta_info.get("completion_tokens", 0.0))
        spec_verify_ct += float(meta_info.get("spec_verify_ct", 0.0))
        e2e_latency += float(meta_info.get("e2e_latency", 0.0))

    return {
        "completion_tokens": completion_tokens,
        "spec_verify_ct": spec_verify_ct,
        "e2e_latency": e2e_latency,
    }


def fetch_metrics_text(base_url: str, request_timeout: float) -> str:
    response = requests.get(f"{base_url}/metrics", timeout=request_timeout)
    response.raise_for_status()
    return response.text


def parse_histogram_snapshot(metrics_text: str, metric_name: str) -> HistogramSnapshot:
    snapshot = HistogramSnapshot()
    for line in metrics_text.splitlines():
        if not line or line.startswith("#") or metric_name not in line:
            continue
        match = PROM_METRIC_RE.match(line)
        if match is None:
            continue

        name = match.group("name")
        value = float(match.group("value"))
        labels = match.group("labels") or ""

        if name == f"{metric_name}_count":
            snapshot.count += value
        elif name == f"{metric_name}_sum":
            snapshot.sum_value += value
        elif name == f"{metric_name}_bucket":
            le_match = LE_LABEL_RE.search(labels)
            if le_match is None:
                continue
            le_value = le_match.group(1)
            bucket_key = math.inf if le_value == "+Inf" else float(le_value)
            snapshot.buckets[bucket_key] = snapshot.buckets.get(bucket_key, 0.0) + value
    return snapshot


def summarize_histogram(snapshot: HistogramSnapshot, include_nonzero_fraction: bool = False) -> MetricSummary:
    count = int(round(snapshot.count))
    if count <= 0:
        return MetricSummary(
            count=0,
            total_seconds=0.0,
            mean_ms=0.0,
            p50_ms=0.0,
            p95_ms=0.0,
            nonzero_fraction=0.0 if include_nonzero_fraction else None,
        )

    sorted_buckets = sorted(snapshot.buckets.items(), key=lambda item: item[0])

    def estimate_quantile_ms(quantile: float) -> float:
        rank = snapshot.count * quantile
        for upper_bound, cumulative_count in sorted_buckets:
            if cumulative_count >= rank:
                return upper_bound * 1000.0 if math.isfinite(upper_bound) else math.inf
        return math.inf

    nonzero_fraction = None
    if include_nonzero_fraction:
        zero_count = snapshot.buckets.get(0.0, 0.0)
        nonzero_fraction = max(0.0, min(1.0, 1.0 - zero_count / snapshot.count))

    return MetricSummary(
        count=count,
        total_seconds=snapshot.sum_value,
        mean_ms=snapshot.sum_value * 1000.0 / snapshot.count,
        p50_ms=estimate_quantile_ms(0.50),
        p95_ms=estimate_quantile_ms(0.95),
        nonzero_fraction=nonzero_fraction,
    )


def collect_metric_summaries(
    base_url: str, request_timeout: float, before_text: str
) -> dict[str, MetricSummary]:
    after_text = fetch_metrics_text(base_url, request_timeout)
    summaries = {}
    for metric_name in METRIC_NAMES:
        before = parse_histogram_snapshot(before_text, metric_name)
        after = parse_histogram_snapshot(after_text, metric_name)
        delta = after.delta(before)
        summaries[metric_name] = summarize_histogram(
            delta, include_nonzero_fraction=(metric_name == TAIL_METRIC)
        )
    return summaries


def format_ms(value: float) -> str:
    if math.isinf(value):
        return "inf"
    return f"{value:.3f}"


def launch_server_for_scenario(
    args: argparse.Namespace, scenario: Scenario, base_url: str
):
    env = dict(os.environ)
    env["SGLANG_RECORD_SPEC_GRAMMAR_TIMING"] = "1"
    if args.log_timing:
        env["SGLANG_LOG_SPEC_GRAMMAR_TIMING"] = "1"
    host = "127.0.0.1"
    port = base_url.rsplit(":", 1)[-1]

    command = [
        "sglang",
        "serve",
        "--model-path",
        args.model_path,
        *build_server_args(args, scenario),
        "--host",
        host,
        "--port",
        port,
    ]
    if args.device != "auto":
        command.extend(["--device", args.device])

    process = subprocess.Popen(
        command,
        env=env,
        cwd=str(REPO_ROOT),
        start_new_session=True,
    )
    try:
        wait_for_server_health(process, base_url, args.server_timeout)
    except Exception:
        kill_process_tree(process.pid)
        raise
    return process


def run_scenario(args: argparse.Namespace, scenario: Scenario) -> dict[str, Any]:
    port = find_available_port(args.base_port)
    base_url = f"http://127.0.0.1:{port}"
    process = launch_server_for_scenario(args, scenario, base_url)

    try:
        for warmup_index in range(scenario.warmup_batches):
            run_generate_batch(
                base_url,
                scenario,
                batch_index=warmup_index,
                request_timeout=args.request_timeout,
            )

        before_text = fetch_metrics_text(base_url, args.request_timeout)

        total_completion_tokens = 0.0
        total_spec_verify_ct = 0.0
        total_e2e_latency = 0.0
        wall_start = time.perf_counter()
        for batch_index in range(scenario.measure_batches):
            batch_stats = run_generate_batch(
                base_url,
                scenario,
                batch_index=scenario.warmup_batches + batch_index,
                request_timeout=args.request_timeout,
            )
            total_completion_tokens += batch_stats["completion_tokens"]
            total_spec_verify_ct += batch_stats["spec_verify_ct"]
            total_e2e_latency += batch_stats["e2e_latency"]
        wall_time = time.perf_counter() - wall_start

        metric_summaries = collect_metric_summaries(
            base_url, args.request_timeout, before_text
        )

        tail_summary = metric_summaries[TAIL_METRIC]
        cpu_mask_summary = metric_summaries["sglang:spec_grammar_cpu_mask_time_seconds"]
        verify_gpu_summary = metric_summaries[
            "sglang:spec_grammar_verify_gpu_time_seconds"
        ]

        accept_length = (
            total_completion_tokens / total_spec_verify_ct
            if total_spec_verify_ct > 0
            else 1.0
        )
        throughput = total_completion_tokens / wall_time if wall_time > 0 else 0.0
        avg_request_latency = (
            total_e2e_latency / (scenario.batch_size * scenario.measure_batches)
            if scenario.batch_size * scenario.measure_batches > 0
            else 0.0
        )

        if tail_summary.count <= 0:
            raise RuntimeError(
                "No speculative grammar timing samples were recorded. "
                "Check that metrics are enabled and the request path hit speculative constrained decoding."
            )

        return {
            "scenario": scenario.name,
            "batch_size": scenario.batch_size,
            "topk": scenario.topk,
            "constraint_kind": scenario.constraint_kind,
            "complexity": scenario.complexity,
            "measure_batches": scenario.measure_batches,
            "accept_length": accept_length,
            "throughput_toks_per_s": throughput,
            "avg_request_latency_s": avg_request_latency,
            "tail_count": tail_summary.count,
            "tail_mean_ms": tail_summary.mean_ms,
            "tail_p50_ms": tail_summary.p50_ms,
            "tail_p95_ms": tail_summary.p95_ms,
            "tail_nonzero_fraction": tail_summary.nonzero_fraction,
            "cpu_mask_mean_ms": cpu_mask_summary.mean_ms,
            "verify_gpu_mean_ms": verify_gpu_summary.mean_ms,
            "metrics": {
                metric_name: {
                    "count": summary.count,
                    "total_seconds": summary.total_seconds,
                    "mean_ms": summary.mean_ms,
                    "p50_ms": summary.p50_ms,
                    "p95_ms": summary.p95_ms,
                    "nonzero_fraction": summary.nonzero_fraction,
                }
                for metric_name, summary in metric_summaries.items()
            },
        }
    finally:
        kill_process_tree(process.pid)


def print_result_table(results: list[dict[str, Any]]) -> None:
    header = (
        f"{'scenario':34} {'bs':>4} {'topk':>4} {'kind':>6} {'comp':>7} "
        f"{'acc_len':>8} {'tail_nz%':>9} {'tail_mean':>10} {'tail_p95':>9} "
        f"{'cpu_mask':>10} {'verify_gpu':>11}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        nonzero_fraction = result["tail_nonzero_fraction"]
        nonzero_pct = 100.0 * nonzero_fraction if nonzero_fraction is not None else 0.0
        print(
            f"{result['scenario'][:34]:34} "
            f"{result['batch_size']:>4} "
            f"{result['topk']:>4} "
            f"{result['constraint_kind']:>6} "
            f"{result['complexity']:>7} "
            f"{result['accept_length']:>8.3f} "
            f"{nonzero_pct:>8.1f}% "
            f"{format_ms(result['tail_mean_ms']):>10} "
            f"{format_ms(result['tail_p95_ms']):>9} "
            f"{format_ms(result['cpu_mask_mean_ms']):>10} "
            f"{format_ms(result['verify_gpu_mean_ms']):>11}"
        )


def print_pairwise_comparison(results: list[dict[str, Any]]) -> None:
    result_map = {result["scenario"]: result for result in results}
    pairs = [
        ("json_simple", "json_complex", "JSON complexity"),
        ("regex_simple", "regex_complex", "Regex complexity"),
        (
            "small_batch_greedy_json_simple",
            "large_batch_high_topk_json_simple",
            "Batch/topk sweep",
        ),
    ]
    for left_name, right_name, title in pairs:
        if left_name not in result_map or right_name not in result_map:
            continue
        left = result_map[left_name]
        right = result_map[right_name]
        tail_delta = right["tail_mean_ms"] - left["tail_mean_ms"]
        nz_delta = (
            (right["tail_nonzero_fraction"] or 0.0)
            - (left["tail_nonzero_fraction"] or 0.0)
        ) * 100.0
        print(
            f"{title}: {left_name} -> {right_name}, "
            f"tail_mean_delta={tail_delta:.3f} ms, "
            f"tail_nonzero_delta={nz_delta:.1f} pct"
        )


def maybe_append_jsonl(path: Optional[str], records: list[dict[str, Any]]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    scenarios = build_scenarios(args)

    if args.list_scenarios:
        for scenario in scenarios:
            print(scenario.name)
        return

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for scenario in scenarios:
        print(
            f"\n[Run] {scenario.name} "
            f"(batch={scenario.batch_size}, topk={scenario.topk}, "
            f"{scenario.constraint_kind}/{scenario.complexity})"
        )
        try:
            result = run_scenario(args, scenario)
            results.append(result)
            print(
                f"[Done] {scenario.name}: "
                f"tail_mean={result['tail_mean_ms']:.3f} ms, "
                f"tail_p95={format_ms(result['tail_p95_ms'])} ms, "
                f"tail_nonzero={(result['tail_nonzero_fraction'] or 0.0) * 100.0:.1f}%"
            )
        except Exception as exc:
            failure = {"scenario": scenario.name, "error": str(exc)}
            failures.append(failure)
            print(f"[Fail] {scenario.name}: {exc}")
            if args.fail_fast:
                raise

    if results:
        print("\nSummary")
        print_result_table(results)
        print()
        print_pairwise_comparison(results)
        maybe_append_jsonl(args.output_jsonl, results)

    if failures:
        print("\nFailures")
        for failure in failures:
            print(f"{failure['scenario']}: {failure['error']}")


if __name__ == "__main__":
    main()
