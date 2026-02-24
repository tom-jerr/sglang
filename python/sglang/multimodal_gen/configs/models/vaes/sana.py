# SPDX-License-Identifier: Apache-2.0
"""
Sana DC-AE (Deep Compression Autoencoder) configuration.

Sana uses a 32x compression autoencoder (vs typical 8x), reducing latent tokens significantly.
Reference: https://huggingface.co/papers/2410.10629
"""

from dataclasses import dataclass, field
from typing import Tuple

from sglang.multimodal_gen.configs.models.vaes.base import VAEArchConfig, VAEConfig


@dataclass
class SanaVAEArchConfig(VAEArchConfig):
    """Architecture configuration for AutoencoderDC (DC-AE)."""

    # Core dimensions
    in_channels: int = 3
    latent_channels: int = 32  # Much higher than typical 4 or 16
    out_channels: int = 3

    # Compression ratio - Sana uses 32x spatial compression
    spatial_compression_ratio: int = 32
    temporal_compression_ratio: int = 1  # Image model, no temporal

    # Encoder/Decoder block configuration
    encoder_block_types: Tuple[str, ...] = (
        "ResBlock",
        "ResBlock",
        "ResBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
    )
    decoder_block_types: Tuple[str, ...] = (
        "ResBlock",
        "ResBlock",
        "ResBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
        "EfficientViTBlock",
    )

    # Channel configuration
    encoder_block_out_channels: Tuple[int, ...] = (128, 256, 512, 512, 1024, 1024)
    decoder_block_out_channels: Tuple[int, ...] = (1024, 1024, 512, 512, 256, 128)

    # Layers per block
    encoder_layers_per_block: Tuple[int, ...] = (2, 2, 2, 3, 3, 3)
    decoder_layers_per_block: Tuple[int, ...] = (3, 3, 3, 2, 2, 2)

    # Multi-scale attention for EfficientViT blocks
    encoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = (
        (),
        (),
        (),
        (5,),
        (5,),
        (5,),
    )
    decoder_qkv_multiscales: Tuple[Tuple[int, ...], ...] = (
        (5,),
        (5,),
        (5,),
        (),
        (),
        (),
    )

    # Attention configuration
    attention_head_dim: int = 32

    # Downsample/Upsample configuration
    downsample_block_type: str = "conv"
    upsample_block_type: str = "interpolate"

    # Normalization and activation
    decoder_norm_types: str = "rms_norm"
    decoder_act_fns: str = "silu"

    # Scaling factor for latent space
    scaling_factor: float = 0.41407

    # VAE scale factor (same as spatial_compression_ratio)
    vae_scale_factor: int = 32


@dataclass
class SanaVAEConfig(VAEConfig):
    """Configuration for Sana's DC-AE VAE."""

    arch_config: SanaVAEArchConfig = field(default_factory=SanaVAEArchConfig)

    # Disable tiling by default for image generation
    use_tiling: bool = False
    use_temporal_tiling: bool = False
    use_parallel_tiling: bool = False

    def __post_init__(self):
        super().__post_init__()

    def post_init(self):
        # Ensure vae_scale_factor is set correctly
        self.arch_config.vae_scale_factor = self.arch_config.spatial_compression_ratio
