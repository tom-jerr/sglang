# SPDX-License-Identifier: Apache-2.0
"""
CUDA Stream and Event management for async pipeline execution.

This module provides utilities for managing CUDA streams and events
to enable CPU-GPU overlap within the denoising loop.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch

from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class StreamManager:
    """
    Manages CUDA streams for async pipeline execution.

    Provides three streams:
    - default_stream: The default CUDA stream (for compute)
    - copy_stream: Dedicated stream for H2D/D2H copies
    - compute_stream: Explicit compute stream (same as default in most cases)

    Usage:
        stream_mgr = StreamManager.create()

        # Async copy on copy_stream
        with stream_mgr.copy_stream_context():
            tensor_gpu = tensor_cpu.to(device, non_blocking=True)
            stream_mgr.record_copy_done()

        # Wait for copy before compute
        stream_mgr.wait_copy_done()

        # Compute on default stream
        output = model(tensor_gpu)
    """

    device: torch.device
    copy_stream: Optional[torch.cuda.Stream] = None
    _copy_done_event: Optional[torch.cuda.Event] = None
    _step_events: Dict[int, torch.cuda.Event] = field(default_factory=dict)
    _enabled: bool = True

    @classmethod
    def create(
        cls, device: Optional[torch.device] = None, enabled: bool = True
    ) -> "StreamManager":
        """Create a StreamManager for the given device."""
        if device is None:
            device = torch.device(
                current_platform.device_type,
                torch.cuda.current_device() if torch.cuda.is_available() else 0,
            )

        if not enabled or not torch.cuda.is_available():
            return cls(device=device, _enabled=False)

        copy_stream = torch.cuda.Stream(device=device)

        logger.debug(f"StreamManager created with copy_stream on {device}")
        return cls(
            device=device,
            copy_stream=copy_stream,
            _enabled=True,
        )

    @property
    def enabled(self) -> bool:
        return self._enabled and self.copy_stream is not None

    @property
    def default_stream(self) -> Optional[torch.cuda.Stream]:
        """Get the default (compute) stream."""
        if not self._enabled:
            return None
        return torch.cuda.current_stream(self.device)

    @contextlib.contextmanager
    def copy_stream_context(self):
        """Context manager for executing on the copy stream."""
        if not self.enabled:
            yield
            return

        with torch.cuda.stream(self.copy_stream):
            yield

    def record_copy_done(self) -> Optional[torch.cuda.Event]:
        """Record an event on the copy stream to mark copy completion."""
        if not self.enabled:
            return None

        event = torch.cuda.Event()
        event.record(self.copy_stream)
        self._copy_done_event = event
        return event

    def wait_copy_done(self, stream: Optional[torch.cuda.Stream] = None):
        """Make a stream wait for the copy to complete."""
        if not self.enabled or self._copy_done_event is None:
            return

        target_stream = stream or torch.cuda.current_stream(self.device)
        target_stream.wait_event(self._copy_done_event)

    def record_step_done(self, step_idx: int) -> Optional[torch.cuda.Event]:
        """Record an event marking the completion of a denoising step."""
        if not self.enabled:
            return None

        event = torch.cuda.Event()
        event.record()
        self._step_events[step_idx] = event
        return event

    def wait_step_done(self, step_idx: int, stream: Optional[torch.cuda.Stream] = None):
        """Make a stream wait for a specific step to complete."""
        if not self.enabled or step_idx not in self._step_events:
            return

        target_stream = stream or torch.cuda.current_stream(self.device)
        target_stream.wait_event(self._step_events[step_idx])

    def get_step_event(self, step_idx: int) -> Optional[torch.cuda.Event]:
        """Get the event for a specific step."""
        return self._step_events.get(step_idx)

    def sync_copy_stream(self):
        """Synchronize the copy stream (blocking)."""
        if self.enabled and self.copy_stream is not None:
            self.copy_stream.synchronize()

    def sync_all(self):
        """Synchronize all streams (blocking)."""
        if self.enabled:
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
            torch.cuda.current_stream(self.device).synchronize()

    def clear_events(self):
        """Clear all recorded events."""
        self._step_events.clear()
        self._copy_done_event = None


# Global singleton for easy access
_global_stream_manager: Optional[StreamManager] = None


def get_stream_manager(
    device: Optional[torch.device] = None, enabled: bool = True, force_new: bool = False
) -> StreamManager:
    """
    Get or create the global StreamManager.

    Args:
        device: Target device. If None, uses current CUDA device.
        enabled: Whether to enable async streams.
        force_new: If True, create a new instance even if one exists.

    Returns:
        The global StreamManager instance.
    """
    global _global_stream_manager

    if _global_stream_manager is None or force_new:
        _global_stream_manager = StreamManager.create(device, enabled)

    return _global_stream_manager


def reset_stream_manager():
    """Reset the global StreamManager."""
    global _global_stream_manager
    if _global_stream_manager is not None:
        _global_stream_manager.clear_events()
    _global_stream_manager = None
