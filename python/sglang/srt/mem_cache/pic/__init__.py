"""Position-independent cache primitives for UnifiedRadixCache.

The first implementation stage is intentionally observer/reference-only.  It
builds and validates content-addressed MLA span plans without changing the
request's live KV mapping.  The registry and lease protocol are usable by the
future fused-attention path without changing the public data model.
"""

from sglang.srt.mem_cache.pic.cdc import ContentDefinedChunker
from sglang.srt.mem_cache.pic.component import PicSpanComponent
from sglang.srt.mem_cache.pic.config import PicConfig
from sglang.srt.mem_cache.pic.mla import MLARequestLocalView, delta_rotate_kr
from sglang.srt.mem_cache.pic.registry import PicSpanLease, PicSpanRegistry
from sglang.srt.mem_cache.pic.types import (
    DependencyMode,
    PicNamespace,
    PicObserverPlan,
    PicResidency,
    PicSpanIdentity,
    PicSpanPayload,
    SeamMetadata,
    ShareScope,
)

__all__ = [
    "ContentDefinedChunker",
    "DependencyMode",
    "MLARequestLocalView",
    "PicConfig",
    "PicNamespace",
    "PicObserverPlan",
    "PicResidency",
    "PicSpanComponent",
    "PicSpanIdentity",
    "PicSpanLease",
    "PicSpanPayload",
    "PicSpanRegistry",
    "SeamMetadata",
    "ShareScope",
    "delta_rotate_kr",
]
