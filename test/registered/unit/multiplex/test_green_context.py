from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sglang.srt.multiplex.green_context import create_green_context_streams
from sglang.srt.multiplex.pdmux_context import _create_green_context_stream_pair


class FakeCudaRuntime:
    cudaStreamNonBlocking = 1
    cudaDevResourceType = SimpleNamespace(cudaDevResourceTypeSm=1)

    class cudaDevSmResourceGroupParams:
        pass

    def __init__(self):
        self.destroyed_streams = []
        self.destroyed_contexts = []

    def cudaSetDevice(self, device):
        return (0,)

    def cudaDeviceGetDevResource(self, device, resource_type):
        return (
            0,
            SimpleNamespace(sm=SimpleNamespace(smCount=128, smCoscheduledAlignment=2)),
        )

    def cudaDevSmResourceSplit(self, count, resource, flags, params):
        assert [param.smCount for param in params] == [112, 16]
        groups = [
            SimpleNamespace(sm=SimpleNamespace(smCount=112)),
            SimpleNamespace(sm=SimpleNamespace(smCount=16)),
        ]
        return 0, groups, SimpleNamespace()

    def cudaDevResourceGenerateDesc(self, resources, count):
        return 0, f"descriptor-{resources[0].sm.smCount}"

    def cudaGreenCtxCreate(self, descriptor, device, flags):
        return 0, int(descriptor.rsplit("-", 1)[1])

    def cudaExecutionCtxStreamCreate(self, context, flags, priority):
        return 0, context + 1000

    def cudaStreamDestroy(self, stream):
        self.destroyed_streams.append(stream)
        return (0,)

    def cudaExecutionCtxDestroy(self, context):
        self.destroyed_contexts.append(context)
        return (0,)


def test_create_and_close_green_context_streams():
    runtime = FakeCudaRuntime()
    with (
        patch(
            "sglang.srt.multiplex.green_context._load_runtime",
            return_value=runtime,
        ),
        patch(
            "sglang.srt.multiplex.green_context.torch.cuda.ExternalStream",
            side_effect=lambda stream, device: (stream, device),
        ),
    ):
        owner = create_green_context_streams((112, 16), device=0)

    assert owner.sm_counts == (112, 16)
    assert [stream[0] for stream in owner.streams] == [1112, 1016]

    owner.close()
    owner.close()
    assert runtime.destroyed_streams == [1016, 1112]
    assert runtime.destroyed_contexts == [16, 112]


def test_green_context_partitions_must_cover_device():
    runtime = FakeCudaRuntime()
    with patch(
        "sglang.srt.multiplex.green_context._load_runtime", return_value=runtime
    ):
        with pytest.raises(ValueError, match="cover the device exactly"):
            create_green_context_streams((100, 16), device=0)


def test_green_context_partitions_must_be_aligned():
    runtime = FakeCudaRuntime()
    with patch(
        "sglang.srt.multiplex.green_context._load_runtime", return_value=runtime
    ):
        with pytest.raises(ValueError, match="not aligned"):
            create_green_context_streams((111, 17), device=0)


def test_torch_stream_backend_is_a_scheduler_only_control():
    with patch(
        "sglang.srt.multiplex.pdmux_context.torch.cuda.Stream",
        side_effect=lambda device: f"stream-{device}",
    ) as stream:
        pair = _create_green_context_stream_pair(112, 16, 3, "torch")

    assert pair == ("stream-3", "stream-3")
    assert stream.call_count == 2
