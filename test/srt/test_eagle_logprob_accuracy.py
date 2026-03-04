import unittest

import numpy as np
import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=400, suite="stage-b-test-small-1-gpu")


class TestEagleLogprobAccuracyBase(CustomTestCase):
    """Base class for logprob accuracy tests."""

    spec_v2_enabled = True  # Override in subclasses
    max_running_requests = 32
    attention_backend = "triton"
    spec_steps = 5
    spec_topk = 1
    spec_draft_tokens = 6

    # Tolerance for logprob comparison
    # Small differences are expected due to numerical precision
    LOGPROB_TOLERANCE = 0.3

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = [
            "--trust-remote-code",
            "--attention-backend",
            cls.attention_backend,
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-draft-model",
            DEFAULT_DRAFT_MODEL_EAGLE,
            "--speculative-num-steps",
            str(cls.spec_steps),
            "--speculative-eagle-topk",
            str(cls.spec_topk),
            "--speculative-num-draft-tokens",
            str(cls.spec_draft_tokens),
            "--mem-fraction-static",
            "0.75",
            "--max-running-requests",
            str(cls.max_running_requests),
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(cls.spec_v2_enabled):
            cls.process = popen_launch_server(
                DEFAULT_TARGET_MODEL_EAGLE,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=launch_args,
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_logprob_spec_v2_match(self):
        """Verify spec v2 decode logprobs match prefill scoring logprobs.

        Generate tokens with spec v2, then score the same sequence via
        prefill-only (no speculation). The two sets of logprobs should be
        close, validating that spec v2 computes logprobs correctly.
        """
        top_k = 5
        probe_token_ids = [1, 2, 10, 100, 1000]
        prompt = "The capital of France is"

        gen_res = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 32,
                    "ignore_eos": True,
                },
                "return_logprob": True,
                "top_logprobs_num": top_k,
                "token_ids_logprob": probe_token_ids,
                "logprob_start_len": 0,
            },
        ).json()

        decode_logprobs = gen_res["meta_info"]["output_token_logprobs"]
        decode_top_logprobs = gen_res["meta_info"]["output_top_logprobs"]
        decode_tid_logprobs = gen_res["meta_info"]["output_token_ids_logprobs"]
        input_token_ids = [t[1] for t in gen_res["meta_info"]["input_token_logprobs"]]
        output_token_ids = [t[1] for t in decode_logprobs]
        num_prompt_tokens = gen_res["meta_info"]["prompt_tokens"]

        score_res = requests.post(
            self.base_url + "/generate",
            json={
                "input_ids": input_token_ids + output_token_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 0,
                },
                "return_logprob": True,
                "top_logprobs_num": top_k,
                "token_ids_logprob": probe_token_ids,
                "logprob_start_len": 0,
            },
        ).json()

        score_logprobs = score_res["meta_info"]["input_token_logprobs"][
            num_prompt_tokens:
        ]
        score_top_logprobs = score_res["meta_info"]["input_top_logprobs"][
            num_prompt_tokens:
        ]
        score_tid_logprobs = score_res["meta_info"]["input_token_ids_logprobs"][
            num_prompt_tokens:
        ]

        self.assertEqual(len(decode_logprobs), len(score_logprobs))

        # Check per-token logprobs
        decode_vals = np.array([t[0] for t in decode_logprobs])
        score_vals = np.array([t[0] for t in score_logprobs])
        max_diff = np.max(np.abs(decode_vals - score_vals))
        print(f"logprob max_diff={max_diff:.6f}")
        print(f"decode_vals[-5:]={decode_vals[-5:]}")
        print(f"score_vals[-5:]={score_vals[-5:]}")
        self.assertLess(max_diff, 0.255)

        # Check top-k logprobs
        for pos in range(len(decode_logprobs)):
            dec_top = {t[1]: t[0] for t in decode_top_logprobs[pos]}
            scr_top = {t[1]: t[0] for t in score_top_logprobs[pos]}
            common_ids = set(dec_top.keys()) & set(scr_top.keys())
            self.assertGreater(len(common_ids), 0)
            for tid in common_ids:
                self.assertAlmostEqual(dec_top[tid], scr_top[tid], delta=0.255)

        # Check token_ids_logprob
        self.assertEqual(len(decode_tid_logprobs), len(score_tid_logprobs))
        for pos in range(len(decode_tid_logprobs)):
            dec_tid = {t[1]: t[0] for t in decode_tid_logprobs[pos]}
            scr_tid = {t[1]: t[0] for t in score_tid_logprobs[pos]}
            self.assertEqual(set(dec_tid.keys()), set(scr_tid.keys()))
            for tid in dec_tid:
                self.assertAlmostEqual(dec_tid[tid], scr_tid[tid], delta=0.255)


class TestEagleLogprobSpecV2(TestEagleLogprobAccuracyBase):
    """Test logprob accuracy for speculative decoding v2 (overlap scheduling)."""

    spec_v2_enabled = True


if __name__ == "__main__":
    unittest.main()
