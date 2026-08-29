from __future__ import annotations

import hashlib
import struct
from collections.abc import Hashable
from dataclasses import dataclass, field
from enum import Enum


class ShareScope(str, Enum):
    TENANT = "tenant"
    SESSION = "session"


class DependencyMode(str, Enum):
    """How a span depends on its absolute/logical position."""

    MLA_POSITION_FREE = "mla_position_free"
    ABSOLUTE_POSITION = "absolute_position"
    GDN_TRANSITION = "gdn_transition"


class PicResidency(str, Enum):
    DEVICE = "device"
    HOST = "host"
    STORAGE = "storage"
    OBSERVER = "observer"


@dataclass(frozen=True, slots=True, kw_only=True)
class PicNamespace:
    """Isolation boundary for a content-addressed span.

    Model/tokenizer/cache-format fingerprints are part of the key so identical
    token integers produced by incompatible runtimes can never alias.  Tenant
    isolation is mandatory; session isolation can be selected for untrusted or
    private traffic.
    """

    tenant_id: str
    model_fingerprint: str
    tokenizer_fingerprint: str
    cache_format: str
    share_scope: ShareScope = ShareScope.TENANT
    session_id: str | None = None

    def __post_init__(self) -> None:
        for name in (
            "tenant_id",
            "model_fingerprint",
            "tokenizer_fingerprint",
            "cache_format",
        ):
            if not getattr(self, name):
                raise ValueError(f"PIC namespace {name} must be non-empty")
        if self.share_scope is ShareScope.SESSION and not self.session_id:
            raise ValueError("session-scoped PIC namespace requires session_id")

    def fingerprint(self) -> bytes:
        session = self.session_id if self.share_scope is ShareScope.SESSION else ""
        digest = hashlib.blake2b(digest_size=16, person=b"sglang-pic-ns")
        for value in (
            self.tenant_id,
            session or "",
            self.model_fingerprint,
            self.tokenizer_fingerprint,
            self.cache_format,
        ):
            encoded = value.encode("utf-8")
            digest.update(struct.pack("<I", len(encoded)))
            digest.update(encoded)
        return digest.digest()


@dataclass(frozen=True, slots=True, kw_only=True)
class PicSpanIdentity:
    namespace_digest: bytes
    content_digest: bytes
    token_count: int

    def __post_init__(self) -> None:
        if len(self.namespace_digest) != 16:
            raise ValueError("PIC namespace digest must be 128 bits")
        if len(self.content_digest) != 16:
            raise ValueError("PIC content digest must be 128 bits")
        if self.token_count <= 0:
            raise ValueError("PIC span token_count must be positive")


@dataclass(frozen=True, slots=True, kw_only=True)
class SeamMetadata:
    left_recompute_tokens: int = 0
    right_recompute_tokens: int = 0
    first_chunk_carveout: bool = False
    marker_stabilized: bool = False


@dataclass(frozen=True, slots=True, kw_only=True)
class PicSpanPayload:
    """Opaque handles owned by the physical MLA storage adapter.

    `c_kv_handle` is canonical and shareable.  `k_r_base_handle` identifies the
    BF16 RoPE slice at `source_position`; consumers create a request-local view
    or rotate on load.  The registry never interprets either handle.
    """

    c_kv_handle: Hashable
    k_r_base_handle: Hashable | None
    source_position: int
    residency: PicResidency = PicResidency.DEVICE
    storage_generation: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class PicSpanRecord:
    handle: int
    identity: PicSpanIdentity
    token_ids: tuple[int, ...]
    dependency_mode: DependencyMode
    seam: SeamMetadata
    payload: PicSpanPayload


@dataclass(frozen=True, slots=True, kw_only=True)
class PicObserverHit:
    source_start: int
    target_start: int
    token_count: int
    delta: int
    content_digest_hex: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PicObserverMiss:
    target_start: int
    token_count: int
    reason: str


@dataclass(frozen=True, slots=True, kw_only=True)
class PicObserverPlan:
    prefix_tokens: int
    hits: tuple[PicObserverHit, ...] = ()
    misses: tuple[PicObserverMiss, ...] = ()
    carveout_tokens: int = 0
    metadata: dict[str, int | str] = field(default_factory=dict)

    @property
    def hit_tokens(self) -> int:
        return sum(hit.token_count for hit in self.hits)
