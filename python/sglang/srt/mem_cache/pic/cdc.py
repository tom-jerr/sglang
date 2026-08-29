from __future__ import annotations

import hashlib
import struct
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from sglang.srt.mem_cache.pic.config import PicConfig

_U64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _U64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return value ^ (value >> 31)


def content_digest(token_ids: Iterable[int]) -> bytes:
    """Strong digest used for registry lookup; exact tokens are rechecked."""

    digest = hashlib.blake2b(digest_size=16, person=b"sglang-pic-span")
    for token_id in token_ids:
        digest.update(struct.pack("<q", int(token_id)))
    return digest.digest()


@dataclass(frozen=True, slots=True, kw_only=True)
class ContentChunk:
    start: int
    end: int
    digest: bytes

    @property
    def token_count(self) -> int:
        return self.end - self.start


class ContentDefinedChunker:
    """Token Gear-hash CDC with deterministic min/target/max clamps.

    The 64-bit recurrence forgets prefix state after at most 64 left shifts,
    matching the marker-stabilisation property used by Irminsul.  Chunk
    fingerprints use a separate strong hash and registry lookup always verifies
    exact tokens, so a digest collision cannot return incorrect KV.
    """

    def __init__(self, config: PicConfig):
        self.config = config
        self._boundary_mask = config.target_chunk_tokens - 1

    @staticmethod
    def _gear_value(token_id: int) -> int:
        return _splitmix64(int(token_id) & _U64_MASK)

    def chunks(
        self, token_ids: Sequence[int], *, absolute_start: int = 0
    ) -> tuple[ContentChunk, ...]:
        if not token_ids:
            return ()

        chunks: list[ContentChunk] = []
        chunk_start = 0
        rolling = 0
        for index, token_id in enumerate(token_ids):
            rolling = ((rolling << 1) + self._gear_value(token_id)) & _U64_MASK
            size = index + 1 - chunk_start
            boundary = (
                size >= self.config.min_chunk_tokens
                and (rolling & self._boundary_mask) == 0
            ) or size >= self.config.max_chunk_tokens
            if boundary:
                token_slice = token_ids[chunk_start : index + 1]
                chunks.append(
                    ContentChunk(
                        start=absolute_start + chunk_start,
                        end=absolute_start + index + 1,
                        digest=content_digest(token_slice),
                    )
                )
                chunk_start = index + 1

        if chunk_start < len(token_ids):
            token_slice = token_ids[chunk_start:]
            chunks.append(
                ContentChunk(
                    start=absolute_start + chunk_start,
                    end=absolute_start + len(token_ids),
                    digest=content_digest(token_slice),
                )
            )
        return tuple(chunks)
