from __future__ import annotations

from dataclasses import dataclass

import torch


def delta_rotate_kr(
    k_r: torch.Tensor,
    delta: int | torch.Tensor,
    inv_freq: torch.Tensor,
    *,
    is_neox_style: bool = True,
) -> torch.Tensor:
    """Apply the closed-form MLA RoPE correction ``R(delta) k_r``.

    Frequencies come from the model rotary module's ``inv_freq`` buffer.  This
    deliberately avoids reconstructing theta, which is unsafe for scaled/custom
    DeepSeek rotary implementations.
    """

    rotary_dim = inv_freq.numel() * 2
    if k_r.shape[-1] != rotary_dim:
        raise ValueError(
            f"k_r width {k_r.shape[-1]} does not match inv_freq rotary width "
            f"{rotary_dim}"
        )
    work = k_r.float()
    delta_tensor = torch.as_tensor(delta, dtype=torch.float32, device=k_r.device)
    angles = delta_tensor[..., None] * inv_freq.to(
        device=k_r.device, dtype=torch.float32
    )
    while angles.ndim < work.ndim:
        angles = angles.unsqueeze(-2)
    cos = angles.cos()
    sin = angles.sin()

    if is_neox_style:
        left, right = work.chunk(2, dim=-1)
    else:
        left, right = work[..., 0::2], work[..., 1::2]
    rotated_left = left * cos - right * sin
    rotated_right = right * cos + left * sin
    if is_neox_style:
        rotated = torch.cat((rotated_left, rotated_right), dim=-1)
    else:
        rotated = torch.stack((rotated_left, rotated_right), dim=-1).flatten(-2)
    return rotated.to(dtype=k_r.dtype)


@dataclass(frozen=True, slots=True, kw_only=True)
class MLARequestLocalView:
    """Logical request view over one canonical c_KV span."""

    canonical_c_kv_handle: object
    source_position: int
    target_position: int
    token_count: int

    @property
    def delta(self) -> int:
        return self.target_position - self.source_position

    def rotate_k_r(
        self,
        k_r_base: torch.Tensor,
        inv_freq: torch.Tensor,
        *,
        is_neox_style: bool = True,
    ) -> torch.Tensor:
        return delta_rotate_kr(
            k_r_base, self.delta, inv_freq, is_neox_style=is_neox_style
        )
