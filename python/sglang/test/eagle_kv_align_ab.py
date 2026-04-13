import argparse
import json
import random
import re
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.srt.utils.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)
from sglang.utils import download_and_cache_file, read_jsonl


def _post_generate(
    base_url: str,
    prompt: str,
    *,
    temperature: float,
    max_new_tokens: int,
) -> Dict[str, Any]:
    response = requests.post(
        f"{base_url}/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=600,
    )
    response.raise_for_status()
    return response.json()


def _generate_many(
    base_url: str,
    prompts: List[str],
    *,
    temperature: float,
    max_new_tokens: int,
) -> List[Dict[str, Any]]:
    return [
        _post_generate(
            base_url,
            prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        for prompt in prompts
    ]


def _first_divergence(lhs: List[int], rhs: List[int]) -> Optional[int]:
    for idx, (lhs_token, rhs_token) in enumerate(zip(lhs, rhs)):
        if lhs_token != rhs_token:
            return idx
    if len(lhs) != len(rhs):
        return min(len(lhs), len(rhs))
    return None


def _distribution(values: List[Optional[int]]) -> Dict[str, int]:
    counter = Counter("match" if value is None else str(value) for value in values)
    return dict(sorted(counter.items(), key=lambda item: item[0]))


def _throughput_from_outputs(outputs: List[Dict[str, Any]]) -> float:
    total_completion_tokens = 0
    total_latency = 0.0
    for output in outputs:
        meta_info = output.get("meta_info", {})
        total_completion_tokens += meta_info.get(
            "completion_tokens", len(output.get("output_ids", []))
        )
        total_latency += meta_info.get("e2e_latency", 0.0)
    if total_latency <= 0:
        return 0.0
    return total_completion_tokens / total_latency


def _collect_server_info(base_url: str) -> Dict[str, Any]:
    response = requests.get(f"{base_url}/get_server_info", timeout=30)
    response.raise_for_status()
    return response.json()


def _launch_mode_server(
    *,
    model: str,
    draft_model: str,
    tp_size: int,
    mode: str,
    compare: bool,
    collect_trace: bool,
    base_url: str,
    timeout: float,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    extra_server_args: Optional[List[str]] = None,
):
    env = {
        "SGLANG_ENABLE_SPEC_V2": "1",
        "SGLANG_EAGLE_KV_ALIGN_MODE": mode,
        "SGLANG_EAGLE_KV_ALIGN_COMPARE": "1" if compare else "0",
        "SGLANG_COLLECT_SPEC_ACCEPT_LENGTH_TRACE": "1" if collect_trace else "0",
    }
    other_args = [
        "--tp-size",
        str(tp_size),
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-draft-model-path",
        draft_model,
        "--speculative-num-steps",
        str(speculative_num_steps),
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        str(speculative_num_draft_tokens),
        "--mem-fraction-static",
        "0.7",
        "--cuda-graph-max-bs",
        "4",
        "--trust-remote-code",
    ]
    if extra_server_args:
        other_args.extend(extra_server_args)
    return popen_launch_server(
        model=model,
        base_url=base_url,
        timeout=timeout,
        other_args=other_args,
        env=env,
    )


def _shutdown_process(process) -> None:
    if process is not None and process.poll() is None:
        kill_process_tree(process.pid)


def _build_chat_prompt(tokenizer, content: str) -> str:
    messages = [{"role": "user", "content": content}]
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return content


def _run_short_prompt_compare(
    *,
    model: str,
    draft_model: str,
    tp_size: int,
    base_url: str,
    timeout: float,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    ) -> Dict[str, Any]:
    prompts = [
        "Today is a sunny day and I like",
        "The capital of France is",
    ]
    process = None
    try:
        process = _launch_mode_server(
            model=model,
            draft_model=draft_model,
            tp_size=tp_size,
            mode="rerun_k",
            compare=True,
            collect_trace=True,
            base_url=base_url,
            timeout=timeout,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        outputs = _generate_many(
            base_url,
            prompts,
            temperature=0.0,
            max_new_tokens=1,
        )
        server_info = _collect_server_info(base_url)
        internal_state = server_info["internal_states"][0]
        return {
            "mode": "rerun_k",
            "max_new_tokens": 1,
            "throughput": _throughput_from_outputs(outputs),
            "avg_spec_accept_length": internal_state.get("avg_spec_accept_length"),
            "kv_align_debug": internal_state.get("eagle_kv_align_debug"),
            "samples": [
                {
                    "text": output["text"],
                    "output_ids": output.get("output_ids", []),
                    "spec_accept_length_trace": output.get("meta_info", {}).get(
                        "spec_accept_length_trace", []
                    ),
                }
                for output in outputs
            ],
        }
    finally:
        _shutdown_process(process)


def _run_short_prompt_accept_trace_pair(
    *,
    model: str,
    draft_model: str,
    tp_size: int,
    base_url: str,
    timeout: float,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
) -> Dict[str, Any]:
    prompts = [
        "Today is a sunny day and I like",
        "The capital of France is",
    ]
    rerun = _run_generation_mode(
        model=model,
        draft_model=draft_model,
        tp_size=tp_size,
        mode="rerun_k",
        prompts=prompts,
        max_new_tokens=16,
        base_url=base_url,
        timeout=timeout,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        collect_trace=True,
    )
    append_last = _run_generation_mode(
        model=model,
        draft_model=draft_model,
        tp_size=tp_size,
        mode="append_last",
        prompts=prompts,
        max_new_tokens=16,
        base_url=base_url,
        timeout=timeout,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        collect_trace=True,
    )

    next_round_matches = []
    examples = []
    for prompt, rerun_output, append_output in zip(
        prompts, rerun["outputs"], append_last["outputs"]
    ):
        rerun_trace = rerun_output.get("meta_info", {}).get("spec_accept_length_trace", [])
        append_trace = append_output.get("meta_info", {}).get(
            "spec_accept_length_trace", []
        )
        rerun_next = rerun_trace[1] if len(rerun_trace) > 1 else None
        append_next = append_trace[1] if len(append_trace) > 1 else None
        next_round_matches.append(
            rerun_next is not None and append_next is not None and rerun_next == append_next
        )
        examples.append(
            {
                "prompt": prompt,
                "rerun_accept_trace": rerun_trace,
                "append_last_accept_trace": append_trace,
                "rerun_t_plus_1_accept_length": rerun_next,
                "append_last_t_plus_1_accept_length": append_next,
                "t_plus_1_accept_length_match": next_round_matches[-1],
            }
        )

    return {
        "rerun_k": {
            "throughput": rerun["throughput"],
            "avg_spec_accept_length": rerun["avg_spec_accept_length"],
        },
        "append_last": {
            "throughput": append_last["throughput"],
            "avg_spec_accept_length": append_last["avg_spec_accept_length"],
        },
        "t_plus_1_accept_length_match_rate": sum(next_round_matches)
        / max(len(next_round_matches), 1),
        "examples": examples,
    }


def _run_generation_mode(
    *,
    model: str,
    draft_model: str,
    tp_size: int,
    mode: str,
    prompts: List[str],
    max_new_tokens: int,
    base_url: str,
    timeout: float,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    collect_trace: bool,
) -> Dict[str, Any]:
    process = None
    try:
        process = _launch_mode_server(
            model=model,
            draft_model=draft_model,
            tp_size=tp_size,
            mode=mode,
            compare=False,
            collect_trace=collect_trace,
            base_url=base_url,
            timeout=timeout,
            speculative_num_steps=speculative_num_steps,
            speculative_num_draft_tokens=speculative_num_draft_tokens,
        )
        outputs = _generate_many(
            base_url,
            prompts,
            temperature=0.0,
            max_new_tokens=max_new_tokens,
        )
        server_info = _collect_server_info(base_url)
        return {
            "mode": mode,
            "outputs": outputs,
            "throughput": _throughput_from_outputs(outputs),
            "avg_spec_accept_length": server_info["internal_states"][0].get(
                "avg_spec_accept_length"
            ),
            "server_info": server_info,
        }
    finally:
        _shutdown_process(process)


def _summarize_pairs(
    pairs: List[Dict[str, Any]],
    *,
    include_examples: bool = True,
    max_examples: int = 10,
) -> Dict[str, Any]:
    first_divergence_positions = [
        pair["first_divergence_token_pos"] for pair in pairs
    ]
    exact_match_count = sum(
        1 for pair in pairs if pair["first_divergence_token_pos"] is None
    )
    ret = {
        "first_divergence_token_pos_distribution": _distribution(
            first_divergence_positions
        ),
        "exact_match_rate": exact_match_count / max(len(pairs), 1),
    }
    if any("accept_trace_divergence_pos" in pair for pair in pairs):
        ret["accept_trace_divergence_pos_distribution"] = _distribution(
            [pair.get("accept_trace_divergence_pos") for pair in pairs]
        )
    if include_examples:
        ret["examples"] = pairs[:max_examples]
    return ret


def _run_long_drift(
    *,
    model: str,
    draft_model: str,
    tp_size: int,
    base_url: str,
    timeout: float,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
) -> Dict[str, Any]:
    prompts = [
        "Today is a sunny day and I like",
        "The future of AI is",
        "A good unit test should",
        "Write a short paragraph about speculative decoding:",
    ]
    rerun = _run_generation_mode(
        model=model,
        draft_model=draft_model,
        tp_size=tp_size,
        mode="rerun_k",
        prompts=prompts,
        max_new_tokens=512,
        base_url=base_url,
        timeout=timeout,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        collect_trace=True,
    )
    append_last = _run_generation_mode(
        model=model,
        draft_model=draft_model,
        tp_size=tp_size,
        mode="append_last",
        prompts=prompts,
        max_new_tokens=512,
        base_url=base_url,
        timeout=timeout,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        collect_trace=True,
    )

    pairs = []
    for prompt, rerun_output, append_output in zip(
        prompts, rerun["outputs"], append_last["outputs"]
    ):
        rerun_ids = rerun_output.get("output_ids", [])
        append_ids = append_output.get("output_ids", [])
        rerun_trace = rerun_output.get("meta_info", {}).get("spec_accept_length_trace", [])
        append_trace = append_output.get("meta_info", {}).get(
            "spec_accept_length_trace", []
        )
        pairs.append(
            {
                "prompt": prompt,
                "first_divergence_token_pos": _first_divergence(rerun_ids, append_ids),
                "accept_trace_divergence_pos": _first_divergence(
                    rerun_trace, append_trace
                ),
                "rerun_accept_trace": rerun_trace,
                "append_last_accept_trace": append_trace,
            }
        )

    return {
        "rerun_k": {
            "throughput": rerun["throughput"],
            "avg_spec_accept_length": rerun["avg_spec_accept_length"],
        },
        "append_last": {
            "throughput": append_last["throughput"],
            "avg_spec_accept_length": append_last["avg_spec_accept_length"],
        },
        **_summarize_pairs(pairs),
    }


def _load_gsm8k_cases(
    *, num_examples: Optional[int], num_shots: int, data_path: Optional[str]
) -> List[Dict[str, Any]]:
    from sglang.test.simple_eval_gsm8k import (
        GSM8K_URL,
        get_answer_value,
        get_few_shot_examples,
        get_one_example,
    )

    filename = data_path or download_and_cache_file(GSM8K_URL)
    lines = list(read_jsonl(filename))
    few_shot_prompt = get_few_shot_examples(lines, num_shots)
    eval_lines = lines[num_shots:]
    if num_examples is not None:
        eval_lines = eval_lines[:num_examples]

    return [
        {
            "case_id": idx,
            "prompt_content": few_shot_prompt
            + get_one_example(eval_lines, idx, include_answer=False),
            "reference_answer": eval_lines[idx]["answer"],
            "reference_value": get_answer_value(eval_lines[idx]["answer"]),
        }
        for idx in range(len(eval_lines))
    ]


def _extract_humaneval_code(completion: str) -> str:
    completion = completion or ""
    matches = re.findall(r"```python\n(.*?)```", completion, flags=re.DOTALL)
    extracted = matches[0] if matches else completion
    signature_offset = extracted.find(":\n    ")
    if signature_offset >= 0:
        extracted = extracted[signature_offset + 2 :]
    return extracted


def _load_humaneval_cases(*, num_examples: Optional[int]) -> List[Dict[str, Any]]:
    from human_eval.data import read_problems

    instruction = (
        "Read the following function signature and docstring, and fully implement "
        "the function described. Your response should only contain the code for "
        "this function.\n"
    )
    samples = list(read_problems().values())
    if num_examples:
        samples = random.Random(0).sample(samples, num_examples)
    return [
        {
            "case_id": sample["task_id"],
            "prompt_content": instruction + sample["prompt"],
            "sample": sample,
        }
        for sample in samples
    ]


def _score_gsm8k_output(case: Dict[str, Any], output: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
    from sglang.test.simple_eval_gsm8k import get_answer_value

    extracted_value = get_answer_value(output.get("text", ""))
    score = float(extracted_value == case["reference_value"])
    return score, {
        "extracted_value": extracted_value,
        "reference_value": case["reference_value"],
    }


def _score_humaneval_output(
    case: Dict[str, Any], output: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    from sglang.test.simple_eval_humaneval import evaluate_functional_correctness

    extracted_code = _extract_humaneval_code(output.get("text", ""))
    passed = evaluate_functional_correctness(case["sample"], [extracted_code])
    score = float(any(passed))
    return score, {
        "passed": bool(score),
        "extracted_code": extracted_code,
    }


def _run_paired_dataset(
    *,
    model: str,
    draft_model: str,
    tp_size: int,
    base_url: str,
    timeout: float,
    speculative_num_steps: int,
    speculative_num_draft_tokens: int,
    cases: List[Dict[str, Any]],
    max_new_tokens: int,
    score_name: str,
    score_fn: Callable[[Dict[str, Any], Dict[str, Any]], Tuple[float, Dict[str, Any]]],
) -> Dict[str, Any]:
    tokenizer = get_tokenizer(model, trust_remote_code=True)
    prompts = [
        _build_chat_prompt(tokenizer, case["prompt_content"]) for case in cases
    ]

    rerun = _run_generation_mode(
        model=model,
        draft_model=draft_model,
        tp_size=tp_size,
        mode="rerun_k",
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        base_url=base_url,
        timeout=timeout,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        collect_trace=False,
    )
    append_last = _run_generation_mode(
        model=model,
        draft_model=draft_model,
        tp_size=tp_size,
        mode="append_last",
        prompts=prompts,
        max_new_tokens=max_new_tokens,
        base_url=base_url,
        timeout=timeout,
        speculative_num_steps=speculative_num_steps,
        speculative_num_draft_tokens=speculative_num_draft_tokens,
        collect_trace=False,
    )

    rerun_score_sum = 0.0
    append_score_sum = 0.0
    pairs = []
    for case, rerun_output, append_output in zip(
        cases, rerun["outputs"], append_last["outputs"]
    ):
        rerun_score, rerun_details = score_fn(case, rerun_output)
        append_score, append_details = score_fn(case, append_output)
        rerun_score_sum += rerun_score
        append_score_sum += append_score
        pairs.append(
            {
                "case_id": case["case_id"],
                "first_divergence_token_pos": _first_divergence(
                    rerun_output.get("output_ids", []),
                    append_output.get("output_ids", []),
                ),
                "rerun_score": rerun_score,
                "append_last_score": append_score,
                "rerun_text": rerun_output.get("text", ""),
                "append_last_text": append_output.get("text", ""),
                "rerun_details": rerun_details,
                "append_last_details": append_details,
            }
        )

    return {
        "num_examples": len(cases),
        "score_name": score_name,
        "rerun_k": {
            score_name: rerun_score_sum / max(len(cases), 1),
            "output_throughput": rerun["throughput"],
            "avg_spec_accept_length": rerun["avg_spec_accept_length"],
        },
        "append_last": {
            score_name: append_score_sum / max(len(cases), 1),
            "output_throughput": append_last["throughput"],
            "avg_spec_accept_length": append_last["avg_spec_accept_length"],
        },
        **_summarize_pairs(pairs),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run EAGLE rerun_k vs append_last KV-alignment A/B experiments."
    )
    parser.add_argument("--model", type=str, default=DEFAULT_TARGET_MODEL_EAGLE)
    parser.add_argument(
        "--draft-model", type=str, default=DEFAULT_DRAFT_MODEL_EAGLE
    )
    parser.add_argument("--base-url", type=str, default=DEFAULT_URL_FOR_TEST)
    parser.add_argument(
        "--timeout", type=float, default=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
    )
    parser.add_argument("--speculative-num-steps", type=int, default=4)
    parser.add_argument("--speculative-num-draft-tokens", type=int, default=5)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--gsm8k-num-examples", type=int, default=50)
    parser.add_argument("--gsm8k-num-shots", type=int, default=5)
    parser.add_argument("--gsm8k-data-path", type=str, default=None)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=512)
    parser.add_argument("--humaneval-num-examples", type=int, default=None)
    parser.add_argument("--humaneval-max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--skip-gsm8k",
        action="store_true",
        help="Skip the GSM8K paired A/B evaluation.",
    )
    parser.add_argument(
        "--skip-humaneval",
        action="store_true",
        help="Skip the HumanEval paired A/B evaluation.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    results = {
        "config": {
            "model": args.model,
            "draft_model": args.draft_model,
            "base_url": args.base_url,
            "tp_size": args.tp_size,
            "speculative_num_steps": args.speculative_num_steps,
            "speculative_num_draft_tokens": args.speculative_num_draft_tokens,
            "spec_v2_enabled": envs.SGLANG_ENABLE_SPEC_V2.get(),
            "kv_align_mode_env_default": envs.SGLANG_EAGLE_KV_ALIGN_MODE.get(),
        },
        "short_prompt_compare": _run_short_prompt_compare(
            model=args.model,
            draft_model=args.draft_model,
            tp_size=args.tp_size,
            base_url=args.base_url,
            timeout=args.timeout,
            speculative_num_steps=args.speculative_num_steps,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        ),
        "short_prompt_t_plus_1_accept_trace": _run_short_prompt_accept_trace_pair(
            model=args.model,
            draft_model=args.draft_model,
            tp_size=args.tp_size,
            base_url=args.base_url,
            timeout=args.timeout,
            speculative_num_steps=args.speculative_num_steps,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        ),
        "long_drift": _run_long_drift(
            model=args.model,
            draft_model=args.draft_model,
            tp_size=args.tp_size,
            base_url=args.base_url,
            timeout=args.timeout,
            speculative_num_steps=args.speculative_num_steps,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
        ),
    }

    if not args.skip_gsm8k:
        results["gsm8k"] = _run_paired_dataset(
            model=args.model,
            draft_model=args.draft_model,
            tp_size=args.tp_size,
            base_url=args.base_url,
            timeout=args.timeout,
            speculative_num_steps=args.speculative_num_steps,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
            cases=_load_gsm8k_cases(
                num_examples=args.gsm8k_num_examples,
                num_shots=args.gsm8k_num_shots,
                data_path=args.gsm8k_data_path,
            ),
            max_new_tokens=args.gsm8k_max_new_tokens,
            score_name="accuracy",
            score_fn=_score_gsm8k_output,
        )

    if not args.skip_humaneval:
        results["humaneval"] = _run_paired_dataset(
            model=args.model,
            draft_model=args.draft_model,
            tp_size=args.tp_size,
            base_url=args.base_url,
            timeout=args.timeout,
            speculative_num_steps=args.speculative_num_steps,
            speculative_num_draft_tokens=args.speculative_num_draft_tokens,
            cases=_load_humaneval_cases(num_examples=args.humaneval_num_examples),
            max_new_tokens=args.humaneval_max_new_tokens,
            score_name="score",
            score_fn=_score_humaneval_output,
        )

    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
