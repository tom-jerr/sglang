from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class PicConfig:
    """Configuration for the phase-1 MLA PIC observer/reference path."""

    min_chunk_tokens: int = 32
    target_chunk_tokens: int = 128
    max_chunk_tokens: int = 512
    rolling_window_tokens: int = 64
    first_chunk_carveout_tokens: int = 32
    max_registry_entries: int = 100_000
    observer_only: bool = True
    model_fingerprint: str = "unknown-model"
    tokenizer_fingerprint: str = "unknown-tokenizer"
    cache_format: str = "mla-bf16-ckv-kr-v1"

    def __post_init__(self) -> None:
        if self.min_chunk_tokens < self.first_chunk_carveout_tokens:
            raise ValueError(
                "PIC min_chunk_tokens must cover the first-chunk carve-out"
            )
        if not (
            self.min_chunk_tokens <= self.target_chunk_tokens <= self.max_chunk_tokens
        ):
            raise ValueError(
                "PIC chunk sizes must satisfy min <= target <= max, got "
                f"{self.min_chunk_tokens} <= {self.target_chunk_tokens} <= "
                f"{self.max_chunk_tokens}"
            )
        if self.target_chunk_tokens & (self.target_chunk_tokens - 1):
            raise ValueError("PIC target_chunk_tokens must be a power of two")
        if self.rolling_window_tokens != 64:
            raise ValueError(
                "phase-1 PIC rolling_window_tokens must be 64 to match the "
                "64-bit Gear recurrence and marker contract"
            )
        if self.max_registry_entries <= 0:
            raise ValueError("PIC max_registry_entries must be positive")
        if not self.observer_only:
            raise ValueError(
                "The phase-1 implementation supports observer_only=True; "
                "live KV remapping requires the split MLA pool and fused attention path"
            )
