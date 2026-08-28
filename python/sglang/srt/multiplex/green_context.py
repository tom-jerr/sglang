"""CUDA Runtime API helpers for green-context streams.

The public CUDA Runtime exposure of green contexts was added in CUDA 13.1.
This module intentionally uses ``cuda.bindings.runtime`` instead of the CUDA
Driver API so callers do not need a compiled extension or manage a current
driver context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

import torch

logger = logging.getLogger(__name__)

_MIN_GREEN_CONTEXT_RUNTIME_VERSION = 13010


class GreenContextRuntimeError(RuntimeError):
    """Raised when CUDA Runtime green-context setup or teardown fails."""


def _error_name(error: Any) -> str:
    return getattr(error, "name", str(error))


def _check(result: tuple, operation: str) -> tuple:
    error = result[0]
    if int(error) != 0:
        raise GreenContextRuntimeError(f"{operation} failed: {_error_name(error)}")
    return result[1:]


def _load_runtime():
    try:
        from cuda.bindings import runtime as cudart
    except (ImportError, AttributeError) as exc:
        raise GreenContextRuntimeError(
            "CUDA Runtime green contexts require cuda-python with CUDA 13.1+ "
            "bindings. Install/upgrade with `pip install -U cuda-python`."
        ) from exc

    required = (
        "cudaDeviceGetDevResource",
        "cudaDevSmResourceSplit",
        "cudaDevResourceGenerateDesc",
        "cudaGreenCtxCreate",
        "cudaExecutionCtxStreamCreate",
        "cudaExecutionCtxDestroy",
    )
    missing = [name for name in required if not hasattr(cudart, name)]
    if missing:
        raise GreenContextRuntimeError(
            "CUDA Runtime green-context bindings are unavailable; missing: "
            + ", ".join(missing)
        )

    (version,) = _check(cudart.cudaRuntimeGetVersion(), "cudaRuntimeGetVersion")
    if version < _MIN_GREEN_CONTEXT_RUNTIME_VERSION:
        raise GreenContextRuntimeError(
            f"CUDA Runtime 13.1+ is required for green contexts; found "
            f"{version // 1000}.{(version % 1000) // 10}."
        )
    return cudart


@dataclass
class GreenContextStreams:
    """Own green contexts and expose their streams as PyTorch streams.

    Keep this object alive for at least as long as any returned stream is in
    use. It can be used as a context manager by standalone tools; SGLang keeps
    the owners for the scheduler process lifetime.
    """

    streams: tuple[torch.cuda.ExternalStream, ...]
    sm_counts: tuple[int, ...]
    device: int
    _contexts: list[Any]
    _raw_streams: list[Any]
    _runtime: Any
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return

        failures = []
        for stream in reversed(self._raw_streams):
            try:
                _check(self._runtime.cudaStreamDestroy(stream), "cudaStreamDestroy")
            except Exception as exc:  # best-effort teardown during shutdown
                failures.append(exc)
        for context in reversed(self._contexts):
            try:
                _check(
                    self._runtime.cudaExecutionCtxDestroy(context),
                    "cudaExecutionCtxDestroy",
                )
            except Exception as exc:  # best-effort teardown during shutdown
                failures.append(exc)

        self._closed = True
        self._raw_streams.clear()
        self._contexts.clear()
        if failures:
            raise GreenContextRuntimeError(
                "Failed to completely destroy CUDA green contexts: "
                + "; ".join(str(exc) for exc in failures)
            )

    def __enter__(self) -> "GreenContextStreams":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def create_green_context_streams(
    sm_counts: Sequence[int],
    device: int | None = None,
    priority: int = 0,
) -> GreenContextStreams:
    """Partition SMs and create one green-context stream per partition.

    Args:
        sm_counts: Exact SM count for every non-overlapping partition. The sum
            must equal the device's available SM count.
        device: CUDA device ordinal. Defaults to PyTorch's current device.
        priority: CUDA stream priority (lower values have higher priority).

    Returns:
        An owner whose ``streams`` can be passed directly to
        ``torch.cuda.stream``.
    """

    if len(sm_counts) < 2:
        raise ValueError("At least two SM partitions are required")
    if any(not isinstance(count, int) or count <= 0 for count in sm_counts):
        raise ValueError(f"SM partition counts must be positive integers: {sm_counts}")

    cudart = _load_runtime()
    if device is None:
        device = torch.cuda.current_device()

    _check(cudart.cudaSetDevice(device), "cudaSetDevice")
    (initial_resource,) = _check(
        cudart.cudaDeviceGetDevResource(
            device, cudart.cudaDevResourceType.cudaDevResourceTypeSm
        ),
        "cudaDeviceGetDevResource",
    )

    available_sms = initial_resource.sm.smCount
    if sum(sm_counts) != available_sms:
        raise ValueError(
            f"SM partitions must cover the device exactly: requested "
            f"{sum(sm_counts)}, available {available_sms}"
        )

    alignment = initial_resource.sm.smCoscheduledAlignment
    invalid = [count for count in sm_counts if count % alignment != 0]
    if invalid:
        raise ValueError(
            f"SM partition counts {invalid} are not aligned to the device "
            f"requirement ({alignment})"
        )

    group_params = []
    for count in sm_counts:
        params = cudart.cudaDevSmResourceGroupParams()
        params.smCount = count
        params.coscheduledSmCount = 0
        params.preferredCoscheduledSmCount = 0
        params.flags = 0
        group_params.append(params)

    split_result, _remainder = _check(
        cudart.cudaDevSmResourceSplit(
            len(group_params), initial_resource, 0, group_params
        ),
        "cudaDevSmResourceSplit",
    )

    contexts: list[Any] = []
    raw_streams: list[Any] = []
    torch_streams: list[torch.cuda.ExternalStream] = []
    try:
        for resource in split_result:
            (descriptor,) = _check(
                cudart.cudaDevResourceGenerateDesc([resource], 1),
                "cudaDevResourceGenerateDesc",
            )
            (context,) = _check(
                cudart.cudaGreenCtxCreate(descriptor, device, 0),
                "cudaGreenCtxCreate",
            )
            contexts.append(context)
            (raw_stream,) = _check(
                cudart.cudaExecutionCtxStreamCreate(
                    context, cudart.cudaStreamNonBlocking, priority
                ),
                "cudaExecutionCtxStreamCreate",
            )
            raw_streams.append(raw_stream)
            torch_streams.append(
                torch.cuda.ExternalStream(
                    int(raw_stream), device=torch.device(f"cuda:{device}")
                )
            )
    except Exception:
        for raw_stream in reversed(raw_streams):
            cudart.cudaStreamDestroy(raw_stream)
        for context in reversed(contexts):
            cudart.cudaExecutionCtxDestroy(context)
        raise

    logger.info(
        "Created CUDA Runtime green contexts on cuda:%d with SM partitions %s",
        device,
        tuple(sm_counts),
    )
    return GreenContextStreams(
        streams=tuple(torch_streams),
        sm_counts=tuple(sm_counts),
        device=device,
        _contexts=contexts,
        _raw_streams=raw_streams,
        _runtime=cudart,
    )


def create_green_context_stream_pair(
    first_sm_count: int,
    second_sm_count: int,
    device: int | None = None,
    priority: int = 0,
) -> GreenContextStreams:
    """Convenience wrapper for the two-stream P/D multiplexing case."""

    return create_green_context_streams(
        (first_sm_count, second_sm_count),
        device=device,
        priority=priority,
    )
