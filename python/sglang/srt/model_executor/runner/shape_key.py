# Copyright 2023-2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Typed identifiers and selection helpers for captured CUDA-graph shapes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True, order=True)
class GraphShape:
    """A finite graph envelope for packed-token workloads.

    ``token_capacity`` bounds the packed token axis and
    ``request_capacity`` bounds the request/segment axis.  Runtime metadata
    (sequence lengths, starts, slot ids, masks, ...) deliberately stays out of
    this object: it belongs in stable-address buffers refreshed before replay.
    """

    token_capacity: int
    request_capacity: int

    def __post_init__(self) -> None:
        if self.token_capacity <= 0 or self.request_capacity <= 0:
            raise ValueError(
                "graph shape capacities must be positive, got "
                f"({self.token_capacity}, {self.request_capacity})"
            )
        if self.request_capacity > self.token_capacity:
            raise ValueError(
                "request_capacity cannot exceed token_capacity for a packed "
                f"prefill/verify graph, got ({self.token_capacity}, "
                f"{self.request_capacity})"
            )

    def covers(self, *, num_tokens: int, num_requests: int) -> bool:
        return (
            num_tokens <= self.token_capacity and num_requests <= self.request_capacity
        )


class GraphShapePlanner:
    """Select from a sparse two-dimensional graph bucket table.

    Selection is deterministic and token-first because packed transformer
    work is primarily proportional to the token axis.  The request axis then
    chooses the smallest metadata/kernel geometry for that token tier.  The
    sparse table lets callers omit combinations that are not worth capturing.
    """

    def __init__(
        self,
        shapes: Iterable[GraphShape],
        *,
        max_token_padding_factor: Optional[float] = None,
    ) -> None:
        self.shapes = tuple(sorted(set(shapes)))
        if not self.shapes:
            raise ValueError("GraphShapePlanner requires at least one shape")
        if max_token_padding_factor is not None and max_token_padding_factor < 1:
            raise ValueError("max_token_padding_factor must be >= 1")
        self.max_token_padding_factor = max_token_padding_factor

    @property
    def token_capacities(self) -> tuple[int, ...]:
        return tuple(sorted({shape.token_capacity for shape in self.shapes}))

    def select(
        self,
        *,
        num_tokens: int,
        num_requests: int,
        token_capacity: Optional[int] = None,
    ) -> Optional[GraphShape]:
        if num_tokens <= 0 or num_requests <= 0:
            return None
        candidates = (
            shape
            for shape in self.shapes
            if shape.covers(num_tokens=num_tokens, num_requests=num_requests)
            and (token_capacity is None or shape.token_capacity == token_capacity)
        )
        selected = min(
            candidates,
            key=lambda shape: (shape.token_capacity, shape.request_capacity),
            default=None,
        )
        if selected is None:
            return None
        if (
            self.max_token_padding_factor is not None
            and selected.token_capacity > num_tokens * self.max_token_padding_factor
        ):
            return None
        return selected


@dataclass(frozen=True)
class ShapeKey:
    """Identifies one captured CUDA-graph shape across all runners.

    size: the per-phase capture size — what the runner iterates over.
        - prefill: num_tokens
        - decode:  bs
    stream_idx:   pdmux stream index, or None for single-stream runners.
    variant_label: optional execution variant (for example, "lora",
        "nolora", or "chunked_prefix"), or None for runners that don't
        record per-variant graphs.
    dsa_variant: DSA decode dual-graph variant ("dense" / "sparse"), or None
        when DSA dual-graph capture is not enabled. Composes with variant_label
        so LoRA and DSA variants can be captured independently.
    request_capacity: optional request/segment-axis capacity for a packed-token
        graph. Together with prefill/compact-verify ``size`` this forms the
        two-dimensional graph geometry. None preserves legacy 1-D keys.
    """

    size: int
    stream_idx: Optional[int] = None
    variant_label: Optional[str] = None
    dsa_variant: Optional[str] = None
    # Keep the new field last so positional construction of legacy keys retains
    # its established meaning.
    request_capacity: Optional[int] = None
