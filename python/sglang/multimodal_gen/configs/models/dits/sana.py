# SPDX-License-Identifier: Apache-2.0
"""
Sana DiT architecture configuration.

Sana uses Linear Attention DiT with GLUMBConv FFN for efficient high-resolution image synthesis.
Reference: https://huggingface.co/papers/2410.10629
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.dits.base import DiTArchConfig, DiTConfig
from sglang.multimodal_gen.runtime.platforms import AttentionBackendEnum


@dataclass
class SanaArchConfig(DiTArchConfig):
    """Architecture configuration for SanaTransformer2DModel."""

    # Patch embedding
    patch_size: int = 1
    in_channels: int = 32  # DC-AE latent channels
    out_channels: int = 32

    # Attention dimensions
    num_attention_heads: int = 70  # 1600M model
    attention_head_dim: int = 32

    # Cross-attention dimensions
    num_cross_attention_heads: int = 20
    cross_attention_head_dim: int = 112
    cross_attention_dim: int = (
        2240  # num_cross_attention_heads * cross_attention_head_dim
    )

    # Transformer blocks
    num_layers: int = 20  # 1600M model: 20, 600M model: 28

    # Caption/text encoder projection
    caption_channels: int = 2304  # Gemma2-2b hidden size

    # MLP
    mlp_ratio: float = 2.5

    # Normalization
    norm_elementwise_affine: bool = False
    norm_eps: float = 1e-6

    # Misc
    attention_bias: bool = False
    sample_size: int = 32  # For 1024px with 32x compression
    dropout: float = 0.0
    interpolation_scale: int | None = None
    guidance_embeds: bool = False
    guidance_embeds_scale: float = 0.1
    qk_norm: str | None = None
    timestep_scale: float = 1.0

    # Weight mapping for loading diffusers weights
    param_names_mapping: dict = field(
        default_factory=lambda: {
            # HF diffusers format: transformer.X.Y -> X.Y
            r"^transformer\.(\w*)\.(.*)$": r"\1.\2",
        }
    )

    # FSDP sharding conditions
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [
            "transformer_blocks",
        ]
    )

    # Supported attention backends - Sana uses Linear Attention for self-attn
    # and standard attention for cross-attn
    _supported_attention_backends: set[AttentionBackendEnum] = field(
        default_factory=lambda: {
            AttentionBackendEnum.TORCH_SDPA,
            AttentionBackendEnum.FA,
        }
    )

    def __post_init__(self):
        super().__post_init__()
        self.out_channels = self.out_channels or self.in_channels
        self.hidden_size = self.num_attention_heads * self.attention_head_dim
        self.num_channels_latents = self.in_channels
        self.cross_attention_dim = (
            self.num_cross_attention_heads * self.cross_attention_head_dim
        )


@dataclass
class Sana1600MArchConfig(SanaArchConfig):
    """1600M parameter Sana model (20 layers)."""

    num_attention_heads: int = 70
    attention_head_dim: int = 32
    num_cross_attention_heads: int = 20
    cross_attention_head_dim: int = 112
    num_layers: int = 20


@dataclass
class Sana600MArchConfig(SanaArchConfig):
    """600M parameter Sana model (28 layers)."""

    num_attention_heads: int = 36
    attention_head_dim: int = 32
    num_cross_attention_heads: int = 16
    cross_attention_head_dim: int = 72
    num_layers: int = 28


@dataclass
class SanaConfig(DiTConfig):
    """Configuration for Sana DiT model."""

    arch_config: DiTArchConfig = field(default_factory=SanaArchConfig)
    prefix: str = "Sana"


@dataclass
class Sana1600MConfig(SanaConfig):
    """Configuration for 1600M Sana model."""

    arch_config: DiTArchConfig = field(default_factory=Sana1600MArchConfig)


@dataclass
class Sana600MConfig(SanaConfig):
    """Configuration for 600M Sana model."""

    arch_config: DiTArchConfig = field(default_factory=Sana600MArchConfig)
