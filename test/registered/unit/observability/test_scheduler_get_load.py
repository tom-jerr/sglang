import time
import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.scheduler_metrics_mixin import SchedulerMetricsMixin


class _FakeBatch:
    def __init__(
        self,
        reqs,
        forward_mode=None,
        extend_num_tokens=0,
        is_prefill_only=False,
    ):
        self.reqs = reqs
        self.forward_mode = forward_mode or ForwardMode.IDLE
        self.extend_num_tokens = extend_num_tokens
        self.is_prefill_only = is_prefill_only

    def is_empty(self):
        return len(self.reqs) == 0


class _DummyScheduler(SchedulerMetricsMixin):
    def __init__(self):
        self.is_hybrid_swa = False
        self.is_hybrid_ssm = False
        self.dp_rank = 0
        self.disaggregation_mode = DisaggregationMode.NULL
        self.waiting_queue = []
        self.running_batch = _FakeBatch([])
        self.cur_batch = None
        self.last_batch = None
        self.prefill_ewma_ms = 7.5

    def _get_token_info(self):
        return (128, 0.0, 0, 0)


class TestSchedulerGetLoad(unittest.TestCase):
    def _make_req(self, seqlen: int, extend_input_len: int, age_ms: float):
        return SimpleNamespace(
            seqlen=seqlen,
            extend_input_len=extend_input_len,
            fill_ids=list(range(extend_input_len)),
            origin_input_ids=list(range(extend_input_len)),
            time_stats=SimpleNamespace(
                wait_queue_entry_time=time.perf_counter() - age_ms / 1000.0
            ),
        )

    def test_prefill_state_reports_stage_aware_fields(self):
        scheduler = _DummyScheduler()
        scheduler.waiting_queue = [
            self._make_req(100, 12, 80),
            self._make_req(200, 20, 30),
        ]
        scheduler.running_batch = _FakeBatch([object()], is_prefill_only=True)
        scheduler.cur_batch = _FakeBatch(
            [object()],
            forward_mode=ForwardMode.EXTEND,
            extend_num_tokens=64,
            is_prefill_only=True,
        )

        load = scheduler.get_load()

        self.assertEqual(load.forward_mode, "prefill")
        self.assertTrue(load.prefill_locked)
        self.assertEqual(load.inflight_prefill_tokens, 64)
        self.assertEqual(load.queued_prefill_reqs, 2)
        self.assertEqual(load.queued_prefill_tokens, 32)
        self.assertGreater(load.oldest_wait_ms, 50.0)
        self.assertEqual(load.prefill_ewma_ms, 7.5)

    def test_decode_state_reports_decode_mode(self):
        scheduler = _DummyScheduler()
        scheduler.running_batch = _FakeBatch([object(), object(), object()])
        scheduler.cur_batch = _FakeBatch(
            [object(), object()],
            forward_mode=ForwardMode.DECODE,
        )

        load = scheduler.get_load()

        self.assertEqual(load.forward_mode, "decode")
        self.assertFalse(load.prefill_locked)
        self.assertEqual(load.running_batch_size, 3)

    def test_idle_state_reports_idle_mode(self):
        scheduler = _DummyScheduler()

        load = scheduler.get_load()

        self.assertEqual(load.forward_mode, "idle")
        self.assertFalse(load.prefill_locked)
        self.assertEqual(load.queued_prefill_reqs, 0)
        self.assertEqual(load.oldest_wait_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
