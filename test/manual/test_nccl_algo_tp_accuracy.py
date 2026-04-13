"""
Manual experiment for measuring TP inference drift between NCCL ring/tree.

This script reuses existing SGLang test utilities:
- server launch: `popen_launch_server`
- cache control: `flush_cache_with_retry`
- request helper: `send_single`
- built-in prompts: `PROMPT_1`, `PROMPT_2`, `LONG_PROMPT`

Example:
python3 test/manual/test_nccl_algo_tp_accuracy.py \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --tp-size 4 \
    --trust-remote-code \
    --max-new-tokens 16
"""

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH = 600
PROMPT_1 = "Tell me about Richard Feynman: "
PROMPT_2 = (
    "Generate 1000 random numbers. Go directly into it, don't say Sure and don't "
    "say here are numbers. Just start with a number."
)
LONG_PROMPT = (PYTHON_ROOT / "sglang/test/long_prompt.txt").read_text()

try:
    from sglang.benchmark.utils import get_tokenizer
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_deterministic import BenchArgs, send_single
    from sglang.test.test_utils import (
        DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        flush_cache_with_retry,
        popen_launch_server,
    )
except ModuleNotFoundError:
    import requests

    class BenchArgs:
        host: str = "127.0.0.1"
        port: int = 30000
        temperature: float = 0.0
        sampling_seed: int = 42
        max_new_tokens: int = 16
        frequency_penalty: float = 0.0
        presence_penalty: float = 0.0
        return_logprob: bool = True
        stream: bool = False

    def get_tokenizer(
        model_path: str,
        trust_remote_code: bool = False,
    ):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code
        )

    def kill_process_tree(pid: int) -> None:
        proc = subprocess.Popen(
            ["pkill", "-TERM", "-P", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait(timeout=5)
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            pass

    def flush_cache_with_retry(
        base_url: str, retries: int = 5, interval: float = 2.0
    ) -> bool:
        for _ in range(retries):
            try:
                response = requests.post(f"{base_url}/flush_cache", timeout=10)
                if response.status_code == 200:
                    return True
            except requests.RequestException:
                pass
            time.sleep(interval)
        return False

    def send_single(
        args,
        profile: bool = False,
        profile_steps: int = 3,
        profile_by_stage: bool = False,
        return_full_response: bool = False,
        input_ids: List[int] = None,
        prompt: List[str] = None,
        max_new_tokens: int = None,
        extra_params: Optional[Dict[str, Any]] = None,
        pick_first_result: bool = True,
    ):
        del profile, profile_steps, profile_by_stage
        base_url = f"http://{args.host}:{args.port}"

        if input_ids is not None:
            json_data = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": args.temperature,
                    "max_new_tokens": (
                        max_new_tokens
                        if max_new_tokens is not None
                        else args.max_new_tokens
                    ),
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                },
                "return_logprob": args.return_logprob,
                "stream": args.stream,
                **(extra_params or {}),
            }
        else:
            json_data = {
                "text": prompt,
                "sampling_params": {
                    "temperature": args.temperature,
                    "max_new_tokens": (
                        max_new_tokens
                        if max_new_tokens is not None
                        else args.max_new_tokens
                    ),
                    "frequency_penalty": args.frequency_penalty,
                    "presence_penalty": args.presence_penalty,
                },
                "return_logprob": args.return_logprob,
                "stream": args.stream,
                **(extra_params or {}),
            }

        if args.sampling_seed is not None:
            json_data["sampling_params"]["sampling_seed"] = args.sampling_seed

        response = requests.post(f"{base_url}/generate", json=json_data, timeout=600)
        response.raise_for_status()
        ret = response.json()
        if pick_first_result:
            ret = ret[0] if isinstance(ret, list) else ret
        return ret if return_full_response else ret["text"]

    def _wait_for_server_health(base_url: str, timeout_duration: float) -> None:
        start_time = time.perf_counter()
        while time.perf_counter() - start_time < timeout_duration:
            try:
                response = requests.get(f"{base_url}/health_generate", timeout=5)
                if response.status_code == 200:
                    return
            except requests.RequestException:
                pass
            time.sleep(5)
        raise TimeoutError(f"Server failed to start within {timeout_duration} seconds")


    def _select_server_python() -> str:
        candidates = [
            sys.executable,
            str(REPO_ROOT / ".venv/bin/python"),
            "/opt/conda/bin/python3",
            "python3",
        ]
        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate != "python3" and not Path(candidate).exists():
                continue
        return sys.executable

    def popen_launch_server(
        model: str,
        base_url: str,
        timeout: float,
        api_key: Optional[str] = None,
        other_args: Optional[list[str]] = None,
        env: Optional[dict] = None,
        return_stdout_stderr: Optional[tuple] = None,
        device: str = "auto",
        pd_separated: bool = False,
        num_replicas: Optional[int] = None,
    ):
        del api_key, return_stdout_stderr, device, pd_separated, num_replicas
        other_args = other_args or []
        env = dict(os.environ if env is None else env)
        existing_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = (
            f"{PYTHON_ROOT}:{existing_pythonpath}"
            if existing_pythonpath
            else str(PYTHON_ROOT)
        )
        python_executable = _select_server_python()
        parsed = urlparse(base_url)
        command = [
            python_executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            *[str(x) for x in other_args],
            "--host",
            parsed.hostname or "127.0.0.1",
            "--port",
            str(parsed.port),
        ]
        print(f"command={' '.join(command)}")
        process = subprocess.Popen(env=env, args=command)
        _wait_for_server_health(base_url, timeout)
        return process

NCCL_ALGOS = {
    "ring": "allreduce:ring",
    "tree": "allreduce:tree",
}


def _safe_mean(values: List[float]) -> Optional[float]:
    return statistics.fmean(values) if values else None


def _safe_max(values: List[float]) -> Optional[float]:
    return max(values) if values else None


def _json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _extract_logprob_and_token_id(item: Any) -> tuple[Optional[float], Optional[int]]:
    if item is None:
        return None, None
    if isinstance(item, (list, tuple)):
        logprob = float(item[0]) if len(item) > 0 and item[0] is not None else None
        token_id = int(item[1]) if len(item) > 1 and item[1] is not None else None
        return logprob, token_id
    raise TypeError(f"Unsupported logprob item type: {type(item).__name__}")


def _compare_logprob_sequences(
    lhs: List[Any],
    rhs: List[Any],
    require_same_token_id: bool = True,
) -> Dict[str, Any]:
    compare_len = min(len(lhs), len(rhs))
    diffs: List[float] = []
    token_id_matches = 0
    first_token_id_mismatch = None

    for idx in range(compare_len):
        lhs_lp, lhs_tid = _extract_logprob_and_token_id(lhs[idx])
        rhs_lp, rhs_tid = _extract_logprob_and_token_id(rhs[idx])
        same_tid = lhs_tid == rhs_tid
        if same_tid:
            token_id_matches += 1
        elif first_token_id_mismatch is None:
            first_token_id_mismatch = idx

        if require_same_token_id and not same_tid:
            continue
        if lhs_lp is None or rhs_lp is None:
            continue
        diffs.append(abs(lhs_lp - rhs_lp))

    return {
        "lhs_len": len(lhs),
        "rhs_len": len(rhs),
        "compared_positions": len(diffs),
        "token_id_match_rate": (
            token_id_matches / compare_len if compare_len > 0 else None
        ),
        "first_token_id_mismatch": first_token_id_mismatch,
        "max_abs_diff": _safe_max(diffs),
        "mean_abs_diff": _safe_mean(diffs),
    }


def _compare_output_token_logprobs(
    lhs_output_ids: List[int],
    rhs_output_ids: List[int],
    lhs_logprobs: List[Any],
    rhs_logprobs: List[Any],
) -> Dict[str, Any]:
    shared_prefix_len = 0
    for lhs_tid, rhs_tid in zip(lhs_output_ids, rhs_output_ids):
        if lhs_tid != rhs_tid:
            break
        shared_prefix_len += 1

    lhs_prefix = lhs_logprobs[:shared_prefix_len]
    rhs_prefix = rhs_logprobs[:shared_prefix_len]
    summary = _compare_logprob_sequences(
        lhs_prefix, rhs_prefix, require_same_token_id=False
    )
    summary["shared_output_prefix_len"] = shared_prefix_len
    summary["output_ids_match"] = lhs_output_ids == rhs_output_ids
    summary["lhs_output_len"] = len(lhs_output_ids)
    summary["rhs_output_len"] = len(rhs_output_ids)
    summary["first_output_id_mismatch"] = (
        None
        if lhs_output_ids == rhs_output_ids
        else shared_prefix_len if shared_prefix_len < min(len(lhs_output_ids), len(rhs_output_ids)) else min(len(lhs_output_ids), len(rhs_output_ids))
    )
    return summary


def _compare_top_logprobs(
    lhs_steps: List[List[Any]],
    rhs_steps: List[List[Any]],
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    compare_len = min(len(lhs_steps), len(rhs_steps))
    if max_steps is not None:
        compare_len = min(compare_len, max_steps)

    jaccards: List[float] = []
    shared_diffs: List[float] = []

    for idx in range(compare_len):
        lhs_map = {}
        rhs_map = {}
        for item in lhs_steps[idx]:
            logprob, token_id = _extract_logprob_and_token_id(item)
            if logprob is not None and token_id is not None:
                lhs_map[token_id] = logprob
        for item in rhs_steps[idx]:
            logprob, token_id = _extract_logprob_and_token_id(item)
            if logprob is not None and token_id is not None:
                rhs_map[token_id] = logprob

        lhs_ids = set(lhs_map)
        rhs_ids = set(rhs_map)
        union = lhs_ids | rhs_ids
        shared = lhs_ids & rhs_ids
        if union:
            jaccards.append(len(shared) / len(union))
        for token_id in shared:
            shared_diffs.append(abs(lhs_map[token_id] - rhs_map[token_id]))

    return {
        "compared_steps": compare_len,
        "mean_jaccard": _safe_mean(jaccards),
        "min_jaccard": min(jaccards) if jaccards else None,
        "max_abs_diff_on_shared_tokens": _safe_max(shared_diffs),
        "mean_abs_diff_on_shared_tokens": _safe_mean(shared_diffs),
    }


def _build_default_prompts(long_prompt_chars: int) -> List[Dict[str, str]]:
    return [
        {"name": "short_factual", "text": PROMPT_1},
        {"name": "medium_generation", "text": PROMPT_2},
        {"name": "long_summary", "text": LONG_PROMPT[:long_prompt_chars]},
    ]


def _build_random_token_ids(
    model_path: str,
    tokenizer_path: Optional[str],
    trust_remote_code: bool,
    seed: int,
    count: int,
) -> List[int]:
    tokenizer = get_tokenizer(
        tokenizer_path or model_path,
        trust_remote_code=trust_remote_code,
    )
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None:
        vocab_size = len(tokenizer)
    if count <= 0:
        return []
    count = min(count, vocab_size)
    rng = random.Random(seed)
    return sorted(rng.sample(range(vocab_size), count))


def _make_bench_args(port: int, max_new_tokens: int, seed: int) -> BenchArgs:
    args = BenchArgs()
    args.host = "127.0.0.1"
    args.port = port
    args.temperature = 0.0
    args.sampling_seed = seed
    args.max_new_tokens = max_new_tokens
    args.return_logprob = True
    args.stream = False
    return args


def _warmup_server(port: int, prompt: str, seed: int) -> None:
    warmup_args = _make_bench_args(port=port, max_new_tokens=1, seed=seed)
    send_single(
        warmup_args,
        prompt=[prompt],
        return_full_response=True,
        extra_params={"top_logprobs_num": 3},
    )


def _run_single_case(
    port: int,
    prompt: str,
    max_new_tokens: int,
    seed: int,
    top_logprobs_num: int,
    token_ids_logprob: List[int],
) -> Dict[str, Any]:
    req_args = _make_bench_args(port=port, max_new_tokens=max_new_tokens, seed=seed)
    response = send_single(
        req_args,
        prompt=[prompt],
        return_full_response=True,
        extra_params={
            "top_logprobs_num": top_logprobs_num,
            "token_ids_logprob": token_ids_logprob,
        },
    )
    if response is None:
        raise RuntimeError("Received empty response from server")
    return response


def _launch_and_collect(
    algo_key: str,
    args: argparse.Namespace,
    prompts: List[Dict[str, str]],
    token_ids_logprob: List[int],
    port: int,
) -> Dict[str, Any]:
    base_url = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env["NCCL_ALGO"] = NCCL_ALGOS[algo_key]
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    other_args = [
        "--tp-size",
        str(args.tp_size),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--disable-custom-all-reduce",
        "--disable-radix-cache",
        "--disable-cuda-graph",
    ]
    if args.trust_remote_code:
        other_args.append("--trust-remote-code")
    if args.tokenizer_path:
        other_args.extend(["--tokenizer-path", args.tokenizer_path])
    if args.dtype:
        other_args.extend(["--dtype", args.dtype])
    if args.attention_backend:
        other_args.extend(["--attention-backend", args.attention_backend])

    process = popen_launch_server(
        args.model_path,
        base_url,
        timeout=args.timeout,
        other_args=other_args,
        env=env,
    )

    try:
        _warmup_server(port, prompts[0]["text"], args.seed)
        results = []
        for prompt_case in prompts:
            if not flush_cache_with_retry(base_url):
                raise RuntimeError(f"Failed to flush cache for {algo_key} at {base_url}")
            response = _run_single_case(
                port=port,
                prompt=prompt_case["text"],
                max_new_tokens=args.max_new_tokens,
                seed=args.seed,
                top_logprobs_num=args.top_logprobs_num,
                token_ids_logprob=token_ids_logprob,
            )
            results.append(
                {
                    "prompt_name": prompt_case["name"],
                    "prompt_text": prompt_case["text"],
                    "response": response,
                }
            )
        return {
            "algo": algo_key,
            "nccl_algo_env": NCCL_ALGOS[algo_key],
            "base_url": base_url,
            "results": results,
        }
    finally:
        kill_process_tree(process.pid)
        time.sleep(2)


def _compare_algo_runs(
    ring_run: Dict[str, Any],
    tree_run: Dict[str, Any],
) -> Dict[str, Any]:
    ring_by_name = {item["prompt_name"]: item for item in ring_run["results"]}
    tree_by_name = {item["prompt_name"]: item for item in tree_run["results"]}

    per_prompt = []
    for prompt_name in ring_by_name:
        ring_resp = ring_by_name[prompt_name]["response"]
        tree_resp = tree_by_name[prompt_name]["response"]

        ring_meta = ring_resp["meta_info"]
        tree_meta = tree_resp["meta_info"]
        ring_output_ids = ring_resp.get("output_ids", [])
        tree_output_ids = tree_resp.get("output_ids", [])

        prompt_logprob_diff = _compare_logprob_sequences(
            ring_meta["input_token_logprobs"],
            tree_meta["input_token_logprobs"],
            require_same_token_id=True,
        )
        output_logprob_diff = _compare_output_token_logprobs(
            ring_output_ids,
            tree_output_ids,
            ring_meta["output_token_logprobs"],
            tree_meta["output_token_logprobs"],
        )
        fixed_token_diff_all = _compare_logprob_sequences(
            [
                item
                for step in ring_meta.get("output_token_ids_logprobs", [])
                for item in step
            ],
            [
                item
                for step in tree_meta.get("output_token_ids_logprobs", [])
                for item in step
            ],
            require_same_token_id=True,
        )
        fixed_token_diff_first = _compare_logprob_sequences(
            ring_meta.get("output_token_ids_logprobs", [[]])[0]
            if ring_meta.get("output_token_ids_logprobs")
            else [],
            tree_meta.get("output_token_ids_logprobs", [[]])[0]
            if tree_meta.get("output_token_ids_logprobs")
            else [],
            require_same_token_id=True,
        )
        top_logprob_first = _compare_top_logprobs(
            ring_meta["output_top_logprobs"],
            tree_meta["output_top_logprobs"],
            max_steps=1,
        )
        top_logprob_prefix = _compare_top_logprobs(
            ring_meta["output_top_logprobs"],
            tree_meta["output_top_logprobs"],
            max_steps=output_logprob_diff["shared_output_prefix_len"],
        )

        per_prompt.append(
            {
                "prompt_name": prompt_name,
                "prompt_tokens_ring": ring_meta["prompt_tokens"],
                "prompt_tokens_tree": tree_meta["prompt_tokens"],
                "output_text_match": ring_resp["text"] == tree_resp["text"],
                "output_ids_match": ring_output_ids == tree_output_ids,
                "ring_output_preview": ring_resp["text"][:160],
                "tree_output_preview": tree_resp["text"][:160],
                "prompt_logprob_diff": prompt_logprob_diff,
                "output_logprob_diff": output_logprob_diff,
                "fixed_token_logprob_diff_first_step": fixed_token_diff_first,
                "fixed_token_logprob_diff_all_steps": fixed_token_diff_all,
                "top_logprob_first_step": top_logprob_first,
                "top_logprob_shared_prefix": top_logprob_prefix,
            }
        )

    def _collect(metric_path: List[str]) -> List[float]:
        values = []
        for item in per_prompt:
            current = item
            for key in metric_path:
                current = current.get(key)
                if current is None:
                    break
            if isinstance(current, (int, float)):
                values.append(float(current))
        return values

    aggregate = {
        "num_prompts": len(per_prompt),
        "all_output_ids_match": all(x["output_ids_match"] for x in per_prompt),
        "all_output_text_match": all(x["output_text_match"] for x in per_prompt),
        "max_prompt_logprob_abs_diff": _safe_max(
            _collect(["prompt_logprob_diff", "max_abs_diff"])
        ),
        "max_output_logprob_abs_diff_on_shared_prefix": _safe_max(
            _collect(["output_logprob_diff", "max_abs_diff"])
        ),
        "max_fixed_token_abs_diff_first_step": _safe_max(
            _collect(["fixed_token_logprob_diff_first_step", "max_abs_diff"])
        ),
        "max_fixed_token_abs_diff_all_steps": _safe_max(
            _collect(["fixed_token_logprob_diff_all_steps", "max_abs_diff"])
        ),
        "min_topk_jaccard_first_step": min(
            _collect(["top_logprob_first_step", "min_jaccard"]) or [1.0]
        ),
        "min_topk_jaccard_shared_prefix": min(
            _collect(["top_logprob_shared_prefix", "min_jaccard"]) or [1.0]
        ),
    }

    return {"aggregate": aggregate, "per_prompt": per_prompt}


def _print_summary(summary: Dict[str, Any]) -> None:
    aggregate = summary["aggregate"]
    print("\n=== NCCL ring vs tree summary ===")
    print(
        f"all_output_ids_match={aggregate['all_output_ids_match']}  "
        f"all_output_text_match={aggregate['all_output_text_match']}"
    )
    print(
        f"max_prompt_logprob_abs_diff={aggregate['max_prompt_logprob_abs_diff']}  "
        f"max_output_logprob_abs_diff_on_shared_prefix={aggregate['max_output_logprob_abs_diff_on_shared_prefix']}"
    )
    print(
        f"max_fixed_token_abs_diff_first_step={aggregate['max_fixed_token_abs_diff_first_step']}  "
        f"max_fixed_token_abs_diff_all_steps={aggregate['max_fixed_token_abs_diff_all_steps']}"
    )
    print(
        f"min_topk_jaccard_first_step={aggregate['min_topk_jaccard_first_step']}  "
        f"min_topk_jaccard_shared_prefix={aggregate['min_topk_jaccard_shared_prefix']}"
    )

    print("\n=== Per prompt ===")
    for item in summary["per_prompt"]:
        print(
            f"[{item['prompt_name']}] "
            f"output_ids_match={item['output_ids_match']} "
            f"first_output_id_mismatch={item['output_logprob_diff']['first_output_id_mismatch']} "
            f"prompt_lp_max_abs={item['prompt_logprob_diff']['max_abs_diff']} "
            f"output_lp_max_abs={item['output_logprob_diff']['max_abs_diff']} "
            f"fixed_lp_first_max_abs={item['fixed_token_logprob_diff_first_step']['max_abs_diff']} "
            f"topk_first_jaccard={item['top_logprob_first_step']['mean_jaccard']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure TP inference drift between NCCL ring/tree in SGLang."
    )
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.72)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--dtype", type=str, default=None)
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--top-logprobs-num", type=int, default=20)
    parser.add_argument("--num-token-ids-logprob", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--base-port", type=int, default=31000)
    parser.add_argument(
        "--timeout", type=int, default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    )
    parser.add_argument("--long-prompt-chars", type=int, default=6000)
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Custom prompt. Can be specified multiple times.",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Optional CUDA_VISIBLE_DEVICES override, e.g. '0,1,2,3'.",
    )
    parser.add_argument(
        "--result-json",
        type=str,
        default=None,
        help="Optional output path. Defaults to /tmp/nccl_algo_tp_accuracy_<ts>.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = (
        [
            {"name": f"custom_{idx}", "text": prompt}
            for idx, prompt in enumerate(args.prompt)
        ]
        if args.prompt
        else _build_default_prompts(args.long_prompt_chars)
    )
    token_ids_logprob = _build_random_token_ids(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
        count=args.num_token_ids_logprob,
    )

    ring_port = args.base_port
    tree_port = args.base_port + 1

    ring_run = _launch_and_collect(
        algo_key="ring",
        args=args,
        prompts=prompts,
        token_ids_logprob=token_ids_logprob,
        port=ring_port,
    )
    tree_run = _launch_and_collect(
        algo_key="tree",
        args=args,
        prompts=prompts,
        token_ids_logprob=token_ids_logprob,
        port=tree_port,
    )

    summary = _compare_algo_runs(ring_run, tree_run)
    _print_summary(summary)

    result_json = args.result_json
    if result_json is None:
        timestamp = int(time.time())
        result_json = f"/tmp/nccl_algo_tp_accuracy_{timestamp}.json"

    payload = {
        "config": {
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path,
            "tp_size": args.tp_size,
            "max_new_tokens": args.max_new_tokens,
            "top_logprobs_num": args.top_logprobs_num,
            "num_token_ids_logprob": args.num_token_ids_logprob,
            "seed": args.seed,
            "prompts": [x["name"] for x in prompts],
            "token_ids_logprob": token_ids_logprob,
        },
        "ring_run": ring_run,
        "tree_run": tree_run,
        "summary": summary,
    }
    with open(result_json, "w") as fout:
        json.dump(payload, fout, indent=2, default=_json_default)
    print(f"\nSaved detailed results to: {result_json}")


if __name__ == "__main__":
    main()
