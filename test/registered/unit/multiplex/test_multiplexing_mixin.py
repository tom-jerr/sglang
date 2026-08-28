from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.multiplex.multiplexing_mixin import SchedulerMultiplexMixin
from sglang.srt.multiplex.pdmux_context import PDMuxConfig


def test_init_pdmux_uses_parallel_state_gpu_id():
    scheduler = SimpleNamespace(
        ps=SimpleNamespace(gpu_id=3),
        tp_worker=SimpleNamespace(),
    )
    config = PDMuxConfig()
    stream_groups = [(object(), object())]

    with (
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.get_disagg",
            return_value=SimpleNamespace(pdmux_config_path=None),
        ),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.load_pdmux_config",
            return_value=config,
        ),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.initialize_stream_groups"
        ) as initialize,
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.get_stream_groups",
            return_value=stream_groups,
        ),
        patch(
            "sglang.srt.multiplex.multiplexing_mixin.get_sm_counts",
            return_value=[(128, 0)],
        ),
    ):
        SchedulerMultiplexMixin.init_pdmux(scheduler)

    initialize.assert_called_once_with(3, config)
    assert scheduler.stream_groups == stream_groups


def test_pdmux_rank_fields_live_in_parallel_state():
    # Scheduler rank fields were consolidated into ParallelState. Keep the two
    # PDMux call sites on that API so startup and split-prefill completion do
    # not regress independently.
    import inspect

    source = inspect.getsource(SchedulerMultiplexMixin)
    assert "self.ps.gpu_id" in source
    assert "self.ps.tp_size" in source
    assert "self.gpu_id" not in source
    assert "self.tp_size" not in source
