from __future__ import annotations

from collections.abc import Callable, Sequence

from sglang.srt.mem_cache.pic.cdc import ContentChunk, ContentDefinedChunker
from sglang.srt.mem_cache.pic.config import PicConfig
from sglang.srt.mem_cache.pic.registry import PicSpanRegistry, RegistrationResult
from sglang.srt.mem_cache.pic.types import (
    DependencyMode,
    PicNamespace,
    PicObserverHit,
    PicObserverMiss,
    PicObserverPlan,
    PicResidency,
    PicSpanIdentity,
    PicSpanPayload,
    SeamMetadata,
)


class PicSpanComponent:
    """Content-addressed PIC-SPAN component composed with UnifiedRadixCache."""

    def __init__(self, config: PicConfig):
        self.config = config
        self.chunker = ContentDefinedChunker(config)
        self.registry = PicSpanRegistry(max_entries=config.max_registry_entries)

    @staticmethod
    def _slice_tokens(
        token_ids: Sequence[int], chunk: ContentChunk, absolute_start: int
    ) -> tuple[int, ...]:
        local_start = chunk.start - absolute_start
        local_end = chunk.end - absolute_start
        return tuple(int(token) for token in token_ids[local_start:local_end])

    @staticmethod
    def _identity(namespace: PicNamespace, chunk: ContentChunk) -> PicSpanIdentity:
        return PicSpanIdentity(
            namespace_digest=namespace.fingerprint(),
            content_digest=chunk.digest,
            token_count=chunk.token_count,
        )

    def publish(
        self,
        token_ids: Sequence[int],
        *,
        namespace: PicNamespace,
        absolute_start: int = 0,
        payload_factory: Callable[[ContentChunk], PicSpanPayload],
        on_retire: Callable[[PicSpanPayload], None] | None = None,
        dependency_mode: DependencyMode = DependencyMode.MLA_POSITION_FREE,
    ) -> tuple[RegistrationResult, ...]:
        if dependency_mode is not DependencyMode.MLA_POSITION_FREE:
            raise ValueError("phase-1 PIC publishing supports MLA_POSITION_FREE only")

        results = []
        for chunk in self.chunker.chunks(token_ids, absolute_start=absolute_start):
            if chunk.start < self.config.first_chunk_carveout_tokens:
                continue
            tokens = self._slice_tokens(token_ids, chunk, absolute_start)
            results.append(
                self.registry.register(
                    identity=self._identity(namespace, chunk),
                    token_ids=tokens,
                    dependency_mode=dependency_mode,
                    seam=SeamMetadata(
                        first_chunk_carveout=False,
                        marker_stabilized=(
                            chunk.start - absolute_start
                            >= self.config.rolling_window_tokens
                        ),
                    ),
                    payload=payload_factory(chunk),
                    on_retire=on_retire,
                )
            )
        return tuple(results)

    def observe_publish(
        self,
        token_ids: Sequence[int],
        *,
        namespace: PicNamespace,
        absolute_start: int = 0,
    ) -> tuple[RegistrationResult, ...]:
        return self.publish(
            token_ids,
            namespace=namespace,
            absolute_start=absolute_start,
            payload_factory=lambda chunk: PicSpanPayload(
                c_kv_handle=(chunk.digest, chunk.start),
                k_r_base_handle=None,
                source_position=chunk.start,
                residency=PicResidency.OBSERVER,
            ),
        )

    def observe_match(
        self,
        token_ids: Sequence[int],
        *,
        namespace: PicNamespace,
        prefix_tokens: int,
        absolute_start: int = 0,
    ) -> PicObserverPlan:
        hits = []
        misses = []
        carveout = 0
        for chunk in self.chunker.chunks(token_ids, absolute_start=absolute_start):
            if chunk.end <= prefix_tokens:
                continue
            if chunk.start < prefix_tokens:
                misses.append(
                    PicObserverMiss(
                        target_start=chunk.start,
                        token_count=chunk.token_count,
                        reason="prefix_seam",
                    )
                )
                continue
            if chunk.start < self.config.first_chunk_carveout_tokens:
                carveout += chunk.token_count
                misses.append(
                    PicObserverMiss(
                        target_start=chunk.start,
                        token_count=chunk.token_count,
                        reason="first_chunk_carveout",
                    )
                )
                continue

            tokens = self._slice_tokens(token_ids, chunk, absolute_start)
            record = self.registry.probe(self._identity(namespace, chunk), tokens)
            if record is None:
                misses.append(
                    PicObserverMiss(
                        target_start=chunk.start,
                        token_count=chunk.token_count,
                        reason="content_miss",
                    )
                )
                continue
            hits.append(
                PicObserverHit(
                    source_start=record.payload.source_position,
                    target_start=chunk.start,
                    token_count=chunk.token_count,
                    delta=chunk.start - record.payload.source_position,
                    content_digest_hex=chunk.digest.hex(),
                )
            )

        return PicObserverPlan(
            prefix_tokens=prefix_tokens,
            hits=tuple(hits),
            misses=tuple(misses),
            carveout_tokens=carveout,
            metadata={"mode": "observer", **self.registry.stats()},
        )
