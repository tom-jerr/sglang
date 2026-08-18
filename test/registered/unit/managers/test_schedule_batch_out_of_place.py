import dataclasses
import types
import unittest
from array import array
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import ScheduleBatch  # noqa: E402
from sglang.srt.managers.scheduler import Scheduler  # noqa: E402
from sglang.srt.managers.overlap_utils import resolve_forward_inputs  # noqa: E402
from sglang.srt.model_executor.forward_batch_info import ForwardMode  # noqa: E402
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm  # noqa: E402
from sglang.srt.utils.common import Range  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

AUTO_FILL_EXCLUDED_FIELDS = ["reqs"]


def make_schedule_batch(bs: int, **overrides) -> ScheduleBatch:
    batch = ScheduleBatch(reqs=overrides.pop("reqs"))
    for field in dataclasses.fields(ScheduleBatch):
        name = field.name
        if name in overrides or name in AUTO_FILL_EXCLUDED_FIELDS:
            continue
        annotation = str(field.type)
        if "List" in annotation or "list[" in annotation:
            setattr(batch, name, [f"{name}-{i}" for i in range(bs)])
        elif "Tensor" in annotation:
            setattr(batch, name, torch.arange(bs, dtype=torch.int64))
    for name, value in overrides.items():
        setattr(batch, name, value)
    return batch


def _snapshot_mutable_fields(batch):
    snapshot = []
    for field in dataclasses.fields(batch):
        value = getattr(batch, field.name, None)
        if isinstance(value, MagicMock):
            continue
        if isinstance(value, list):
            snapshot.append((field.name, value, list(value)))
        elif isinstance(value, torch.Tensor):
            snapshot.append((field.name, value, value.clone()))
    return snapshot


def _assert_snapshot_not_mutated(test_case, snapshot):
    for name, obj, value_copy in snapshot:
        if isinstance(obj, list):
            test_case.assertEqual(obj, value_copy, f"field {name} was mutated in place")
        else:
            test_case.assertTrue(
                torch.equal(obj, value_copy), f"field {name} was mutated in place"
            )


class _FakeReq:
    def __init__(self, rid, origin_len, output_len):
        self.rid = rid
        self.origin_input_ids = list(range(origin_len))
        self.output_ids = list(range(output_len))
        self.full_untruncated_fill_ids = list(range(origin_len + output_len))
        self.extend_range = None

    def _refresh_fill_ids(self):
        self.full_untruncated_fill_ids = self.origin_input_ids + self.output_ids

    def set_extend_range(self, start, end):
        self.extend_range = Range(start, end)


class TestMergeBatchOutOfPlace(unittest.TestCase):
    def test_merge_batch_rebinds_lists_without_mutating_either_side(self):
        """merge_batch must build new list objects; no field of either side may be mutated in place."""
        self_batch = make_schedule_batch(
            2,
            reqs=[types.SimpleNamespace(rid="a"), types.SimpleNamespace(rid="b")],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=True,
            top_logprobs_nums=[1, 2],
            token_ids_logprobs=[[10], [20]],
        )
        other_batch = make_schedule_batch(
            1,
            reqs=[types.SimpleNamespace(rid="c")],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=True,
            top_logprobs_nums=[3],
            token_ids_logprobs=[[30]],
        )

        self_reqs_before = self_batch.reqs
        self_top_before = self_batch.top_logprobs_nums
        self_snapshot = _snapshot_mutable_fields(self_batch)
        other_snapshot = _snapshot_mutable_fields(other_batch)

        self_batch.merge_batch(other_batch)

        self.assertEqual([r.rid for r in self_batch.reqs], ["a", "b", "c"])
        self.assertEqual(self_batch.top_logprobs_nums, [1, 2, 3])
        self.assertEqual(self_batch.token_ids_logprobs, [[10], [20], [30]])
        self.assertIsNot(self_batch.reqs, self_reqs_before)
        self.assertIsNot(self_batch.top_logprobs_nums, self_top_before)
        _assert_snapshot_not_mutated(self, self_snapshot)
        _assert_snapshot_not_mutated(self, other_snapshot)


class TestMixWithRunningOutOfPlace(unittest.TestCase):
    def test_spec_decode_reserve_handles_unpublished_overlap_kv(self):
        batch = ScheduleBatch(
            reqs=[types.SimpleNamespace(kv=None, kv_committed_len=17)],
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            token_to_kv_pool_allocator=types.SimpleNamespace(page_size=4),
        )

        with patch(
            "sglang.srt.managers.schedule_batch.get_alloc_reserve_per_decode",
            return_value=7,
        ):
            self.assertEqual(batch.new_tokens_required_next_decode(), 8)

    def test_spec_mixed_resolves_future_relay_only_for_verify_child(self):
        prefill = ScheduleBatch(
            reqs=[],
            device="cpu",
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            prefill_input_ids_cpu=torch.tensor([1, 2]),
            spec_info=types.SimpleNamespace(kind="draft_extend_without_future_indices"),
        )
        verify = ScheduleBatch(
            reqs=[],
            device="cpu",
            enable_overlap=True,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            input_ids=torch.tensor([3]),
            spec_info=types.SimpleNamespace(future_indices=torch.tensor([0])),
        )
        parent = ScheduleBatch(
            reqs=[],
            spec_mixed_prefill_batch=prefill,
            spec_mixed_verify_batch=verify,
        )
        future_map = types.SimpleNamespace(
            spec_algo=SpeculativeAlgorithm.EAGLE3,
            _resolve_spec_extras=MagicMock(),
        )

        resolve_forward_inputs(parent, future_map)

        self.assertTrue(torch.equal(prefill.input_ids, torch.tensor([1, 2])))
        future_map._resolve_spec_extras.assert_called_once_with(verify)

    def test_spec_mixed_child_snapshots_do_not_inherit_previous_children(self):
        common = dict(
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=False,
            spec_algorithm=SpeculativeAlgorithm.EAGLE3,
            spec_info=None,
        )
        prefill = make_schedule_batch(
            1,
            reqs=[types.SimpleNamespace(rid="p")],
            forward_mode=ForwardMode.EXTEND,
            **common,
        )
        running = make_schedule_batch(
            1,
            reqs=[types.SimpleNamespace(rid="r")],
            forward_mode=ForwardMode.TARGET_VERIFY,
            **common,
        )
        running.spec_mixed_prefill_batch = types.SimpleNamespace(stale=True)
        running.spec_mixed_verify_batch = types.SimpleNamespace(stale=True)

        prefill.mix_with_spec_running(running)

        self.assertIsNone(prefill.spec_mixed_prefill_batch.spec_mixed_prefill_batch)
        self.assertIsNone(prefill.spec_mixed_prefill_batch.spec_mixed_verify_batch)
        self.assertIsNone(prefill.spec_mixed_verify_batch.spec_mixed_prefill_batch)
        self.assertIsNone(prefill.spec_mixed_verify_batch.spec_mixed_verify_batch)

    def test_mix_with_running_rebinds_extend_fields_without_mutating_either_side(self):
        """mix_with_running must append via rebound lists; no field of either side may be mutated in place."""
        extend_batch = make_schedule_batch(
            2,
            reqs=[types.SimpleNamespace(rid="e1"), types.SimpleNamespace(rid="e2")],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=False,
            forward_mode=ForwardMode.EXTEND,
            enable_overlap=False,
            is_prefill_only=True,
            out_cache_loc=torch.arange(6, dtype=torch.int64),
            prefix_lens=[0, 0],
            extend_lens=[3, 3],
            extend_num_tokens=6,
            extend_logprob_start_lens=[0, 0],
        )
        running_batch = make_schedule_batch(
            1,
            reqs=[_FakeReq("r1", origin_len=4, output_len=2)],
            model_config=types.SimpleNamespace(is_encoder_decoder=False),
            sampling_info=MagicMock(),
            return_logprob=False,
            forward_mode=ForwardMode.DECODE,
            out_cache_loc=torch.arange(6, 7, dtype=torch.int64),
        )

        extend_prefix_before = extend_batch.prefix_lens
        extend_lens_before = extend_batch.extend_lens
        extend_snapshot = _snapshot_mutable_fields(extend_batch)
        running_snapshot = _snapshot_mutable_fields(running_batch)

        extend_batch.mix_with_running(running_batch)

        self.assertEqual(extend_batch.forward_mode, ForwardMode.MIXED)
        self.assertIs(extend_batch.mix_running_indices, running_batch.req_pool_indices)
        self.assertEqual([r.rid for r in extend_batch.reqs], ["e1", "e2", "r1"])
        self.assertTrue(
            torch.equal(extend_batch.out_cache_loc, torch.arange(7, dtype=torch.int64))
        )
        # delta is -1 without overlap: 4 origin + 2 output - 1
        self.assertEqual(extend_batch.prefix_lens, [0, 0, 5])
        self.assertEqual(extend_batch.extend_lens, [3, 3, 1])
        self.assertEqual(extend_batch.extend_num_tokens, 7)
        self.assertEqual(extend_batch.extend_logprob_start_lens, [0, 0, 0])
        self.assertFalse(extend_batch.is_prefill_only)
        self.assertIsNot(extend_batch.prefix_lens, extend_prefix_before)
        self.assertIsNot(extend_batch.extend_lens, extend_lens_before)
        _assert_snapshot_not_mutated(self, extend_snapshot)
        _assert_snapshot_not_mutated(self, running_snapshot)


class TestPrepareEncoderInfoExtendOutOfPlace(unittest.TestCase):
    def test_prepare_encoder_info_extend_rebinds_lens_without_mutating_old_lists(self):
        """prepare_encoder_info_extend must strip encoder tokens via rebound lists; old list objects stay intact."""
        req_with_image = types.SimpleNamespace(
            rid="img",
            multimodal_inputs=types.SimpleNamespace(num_image_tokens=2),
            prefix_indices=[],
            extend_range=Range(0, 5),
            logprob_start_len=0,
        )
        req_text_only = types.SimpleNamespace(
            rid="txt",
            multimodal_inputs=None,
            prefix_indices=[],
            extend_range=Range(0, 4),
            logprob_start_len=0,
        )
        batch = make_schedule_batch(
            2,
            reqs=[req_with_image, req_text_only],
            device="cpu",
            forward_mode=ForwardMode.EXTEND,
            out_cache_loc=torch.arange(9, dtype=torch.int64),
            prefix_lens=[0, 0],
            extend_lens=[5, 4],
            extend_num_tokens=9,
            extend_logprob_start_lens=[0, 0],
            extend_input_logprob_token_ids=torch.arange(9, dtype=torch.int64),
        )

        prefix_before = batch.prefix_lens
        extend_before = batch.extend_lens
        logprob_start_before = batch.extend_logprob_start_lens
        snapshot = _snapshot_mutable_fields(batch)

        batch.prepare_encoder_info_extend(
            input_ids=[array("q", range(5)), array("q", range(4))],
            seq_lens=[5, 4],
        )

        self.assertEqual(batch.encoder_lens_cpu, [2, 0])
        self.assertEqual(batch.encoder_cached, [False, True])
        self.assertEqual(batch.extend_lens, [3, 4])
        self.assertEqual(batch.prefix_lens, [0, 0])
        self.assertEqual(batch.extend_num_tokens, 7)
        self.assertTrue(
            torch.equal(batch.out_cache_loc, torch.arange(2, 9, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(batch.encoder_out_cache_loc, torch.arange(2, dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(batch.seq_lens_cpu, torch.tensor([3, 4], dtype=torch.int64))
        )
        self.assertTrue(
            torch.equal(
                batch.extend_input_logprob_token_ids,
                torch.arange(2, 9, dtype=torch.int64),
            )
        )
        self.assertEqual(batch.extend_logprob_start_lens, [0, 0])
        self.assertIsNot(batch.prefix_lens, prefix_before)
        self.assertIsNot(batch.extend_lens, extend_before)
        self.assertIsNot(batch.extend_logprob_start_lens, logprob_start_before)
        _assert_snapshot_not_mutated(self, snapshot)


class TestSpecMixedAdmission(unittest.TestCase):
    @patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0)
    @patch("sglang.srt.managers.scheduler.get_spec")
    def test_tiny_prefix_hit_uses_pure_prefill_while_decode_has_slack(
        self, get_spec_mock, _monotonic_mock
    ):
        get_spec_mock.return_value = types.SimpleNamespace(
            speculative_num_draft_tokens=4
        )
        scheduler = object.__new__(Scheduler)
        scheduler._spec_mixed_decode_service_ewma_s = 0.010
        scheduler._spec_mixed_last_decode_completion_s = 99.995
        new_batch = types.SimpleNamespace(
            reqs=[
                types.SimpleNamespace(
                    extend_range=Range(0, 2), prefix_indices=[0] * 1024
                )
            ]
        )
        running_batch = types.SimpleNamespace(reqs=[object()] * 4)

        self.assertFalse(
            scheduler._should_mix_spec_prefill(new_batch, running_batch)
        )

    @patch("sglang.srt.managers.scheduler.time.monotonic", return_value=100.0)
    @patch("sglang.srt.managers.scheduler.get_spec")
    def test_tiny_prefix_hit_mixes_when_decode_deadline_is_exhausted(
        self, get_spec_mock, _monotonic_mock
    ):
        get_spec_mock.return_value = types.SimpleNamespace(
            speculative_num_draft_tokens=4
        )
        scheduler = object.__new__(Scheduler)
        scheduler._spec_mixed_decode_service_ewma_s = 0.010
        scheduler._spec_mixed_last_decode_completion_s = 99.950
        new_batch = types.SimpleNamespace(
            reqs=[
                types.SimpleNamespace(
                    extend_range=Range(0, 2), prefix_indices=[0] * 1024
                )
            ]
        )
        running_batch = types.SimpleNamespace(reqs=[object()] * 4)

        self.assertTrue(scheduler._should_mix_spec_prefill(new_batch, running_batch))

    def test_non_tiny_suffix_keeps_existing_mixed_path(self):
        scheduler = object.__new__(Scheduler)
        scheduler._spec_mixed_decode_service_ewma_s = None
        scheduler._spec_mixed_last_decode_completion_s = 0.0
        new_batch = types.SimpleNamespace(
            reqs=[
                types.SimpleNamespace(
                    extend_range=Range(0, 8), prefix_indices=[0] * 1024
                )
            ]
        )
        running_batch = types.SimpleNamespace(reqs=[object()] * 4)

        self.assertTrue(scheduler._should_mix_spec_prefill(new_batch, running_batch))

    def test_tiny_uncached_prompt_keeps_existing_mixed_path(self):
        scheduler = object.__new__(Scheduler)
        scheduler._spec_mixed_decode_service_ewma_s = None
        scheduler._spec_mixed_last_decode_completion_s = 0.0
        new_batch = types.SimpleNamespace(
            reqs=[
                types.SimpleNamespace(extend_range=Range(0, 2), prefix_indices=[])
            ]
        )
        running_batch = types.SimpleNamespace(reqs=[object()] * 4)

        self.assertTrue(scheduler._should_mix_spec_prefill(new_batch, running_batch))


if __name__ == "__main__":
    unittest.main()
