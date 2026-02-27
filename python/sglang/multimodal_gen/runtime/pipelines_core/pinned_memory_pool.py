# SPDX-License-Identifier: Apache-2.0
"""
Pinned Memory Pool for efficient CPU-GPU data transfer.

This module provides a pool of pinned (page-locked) CPU memory buffers
for fast async H2D transfers. Pinned memory enables DMA transfers that
don't require CPU involvement, significantly improving copy throughput.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import torch

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


@dataclass
class BufferKey:
    """Key for identifying a buffer in the pool."""

    shape: Tuple[int, ...]
    dtype: torch.dtype

    def __hash__(self):
        return hash((self.shape, self.dtype))

    def __eq__(self, other):
        if not isinstance(other, BufferKey):
            return False
        return self.shape == other.shape and self.dtype == other.dtype

    @property
    def numel(self) -> int:
        result = 1
        for s in self.shape:
            result *= s
        return result

    @property
    def nbytes(self) -> int:
        return self.numel * self.dtype.itemsize


class PinnedMemoryPool:
    """
    A pool of pinned CPU memory buffers for efficient GPU transfers.

    Features:
    - Lazy allocation: Buffers are created on first request
    - LRU eviction: Old buffers are evicted when pool size limit is reached
    - Thread-safe: Can be used from multiple threads
    - Shape-aware: Maintains separate buffers for different shapes/dtypes

    Usage:
        pool = PinnedMemoryPool(max_size_mb=256)

        # Get a pinned buffer (may reuse existing or allocate new)
        pinned_buf = pool.get_buffer((1, 16, 64, 64), torch.float32)

        # Fill the buffer with data
        pinned_buf.copy_(cpu_tensor)

        # Async copy to GPU (fast DMA transfer)
        gpu_tensor = pinned_buf.to(device, non_blocking=True)

        # Return buffer to pool when done
        pool.return_buffer(pinned_buf)
    """

    def __init__(
        self,
        max_size_mb: float = 256.0,
        enabled: bool = True,
    ):
        """
        Initialize the pinned memory pool.

        Args:
            max_size_mb: Maximum total size of pinned buffers in MB.
            enabled: If False, get_buffer returns regular CPU tensors.
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.enabled = enabled and torch.cuda.is_available()

        # Buffer storage: key -> list of available buffers
        self._buffers: Dict[BufferKey, list[torch.Tensor]] = {}
        # LRU tracking: key -> last access order
        self._lru: OrderedDict[BufferKey, int] = OrderedDict()
        self._access_counter = 0

        # Current allocated size
        self._current_size_bytes = 0

        # Thread safety
        self._lock = threading.Lock()

        # Statistics
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_allocated_bytes": 0,
        }

    def get_buffer(
        self,
        shape: Tuple[int, ...],
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Get a pinned memory buffer of the specified shape and dtype.

        Args:
            shape: Desired tensor shape.
            dtype: Desired tensor dtype.

        Returns:
            A pinned CPU tensor. If pool is disabled, returns regular tensor.
        """
        if not self.enabled:
            return torch.empty(shape, dtype=dtype)

        key = BufferKey(shape=shape, dtype=dtype)

        with self._lock:
            # Try to get from pool
            if key in self._buffers and self._buffers[key]:
                buffer = self._buffers[key].pop()
                self._update_lru(key)
                self._stats["hits"] += 1
                return buffer

            # Need to allocate new buffer
            self._stats["misses"] += 1

            # Check if we need to evict
            needed_bytes = key.nbytes
            while (
                self._current_size_bytes + needed_bytes > self.max_size_bytes
                and self._lru
            ):
                self._evict_one()

            # Allocate new pinned buffer
            try:
                buffer = torch.empty(shape, dtype=dtype, pin_memory=True)
                self._current_size_bytes += needed_bytes
                self._stats["total_allocated_bytes"] += needed_bytes
                self._update_lru(key)
                return buffer
            except RuntimeError as e:
                # Fall back to regular memory if pinned allocation fails
                logger.warning(f"Failed to allocate pinned memory: {e}")
                return torch.empty(shape, dtype=dtype)

    def return_buffer(self, buffer: torch.Tensor):
        """
        Return a buffer to the pool for reuse.

        Args:
            buffer: The tensor to return. Must be a pinned CPU tensor.
        """
        if not self.enabled or not buffer.is_pinned():
            return

        key = BufferKey(shape=tuple(buffer.shape), dtype=buffer.dtype)

        with self._lock:
            if key not in self._buffers:
                self._buffers[key] = []
            self._buffers[key].append(buffer)
            self._update_lru(key)

    def async_copy(
        self,
        src: torch.Tensor,
        target_device: torch.device,
        non_blocking: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Copy tensor across devices with pinned-memory staging when applicable.

        Returns:
            (copied_tensor, staged_pinned_buffer_or_none)
        """
        if src.device == target_device:
            return src, None

        # CPU -> CUDA: stage through pinned CPU for async H2D DMA.
        if src.device.type == "cpu" and target_device.type == "cuda":
            if self.enabled:
                pinned_buf = self.get_buffer(tuple(src.shape), src.dtype)
                pinned_buf.copy_(src)
                dst = torch.empty_like(src, device=target_device)
                dst.copy_(pinned_buf, non_blocking=non_blocking)
                if non_blocking:
                    return dst, pinned_buf
                self.return_buffer(pinned_buf)
                return dst, None
            return src.to(target_device, non_blocking=non_blocking), None

        # CUDA -> CPU: use pinned destination for async D2H.
        if src.device.type == "cuda" and target_device.type == "cpu":
            if self.enabled:
                pinned_buf = self.get_buffer(tuple(src.shape), src.dtype)
                pinned_buf.copy_(src, non_blocking=non_blocking)
                if non_blocking:
                    return pinned_buf, pinned_buf
                return pinned_buf, None
            return src.to(target_device, non_blocking=non_blocking), None

        return src.to(target_device, non_blocking=non_blocking), None

    def _update_lru(self, key: BufferKey):
        """Update LRU tracking for a key."""
        self._access_counter += 1
        if key in self._lru:
            self._lru.move_to_end(key)
        else:
            self._lru[key] = self._access_counter

    def _evict_one(self):
        """Evict the least recently used buffer."""
        if not self._lru:
            return

        # Get the oldest key
        oldest_key = next(iter(self._lru))

        # Remove buffers for this key
        if oldest_key in self._buffers and self._buffers[oldest_key]:
            buffer = self._buffers[oldest_key].pop()
            self._current_size_bytes -= oldest_key.nbytes
            self._stats["evictions"] += 1

            # Clean up empty lists
            if not self._buffers[oldest_key]:
                del self._buffers[oldest_key]
                del self._lru[oldest_key]

    def clear(self):
        """Clear all buffers from the pool."""
        with self._lock:
            self._buffers.clear()
            self._lru.clear()
            self._current_size_bytes = 0

    def get_stats(self) -> Dict:
        """Get pool statistics."""
        with self._lock:
            return {
                **self._stats,
                "current_size_mb": self._current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "num_buffer_types": len(self._buffers),
                "hit_rate": (
                    self._stats["hits"] / (self._stats["hits"] + self._stats["misses"])
                    if (self._stats["hits"] + self._stats["misses"]) > 0
                    else 0.0
                ),
            }

    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"PinnedMemoryPool("
            f"current={stats['current_size_mb']:.1f}MB, "
            f"max={stats['max_size_mb']:.1f}MB, "
            f"hit_rate={stats['hit_rate']:.1%})"
        )


# Global singleton
_global_pinned_pool: Optional[PinnedMemoryPool] = None


def get_pinned_memory_pool(
    max_size_mb: float = 256.0,
    enabled: bool = True,
    force_new: bool = False,
) -> PinnedMemoryPool:
    """
    Get or create the global pinned memory pool.

    Args:
        max_size_mb: Maximum pool size in MB (only used on first creation).
        enabled: Whether pinned memory is enabled.
        force_new: If True, create a new pool even if one exists.

    Returns:
        The global PinnedMemoryPool instance.
    """
    global _global_pinned_pool

    if _global_pinned_pool is None or force_new:
        _global_pinned_pool = PinnedMemoryPool(max_size_mb, enabled)
        logger.debug(f"Created PinnedMemoryPool with max_size={max_size_mb}MB")

    return _global_pinned_pool


def reset_pinned_memory_pool():
    """Reset the global pinned memory pool."""
    global _global_pinned_pool
    if _global_pinned_pool is not None:
        _global_pinned_pool.clear()
    _global_pinned_pool = None
