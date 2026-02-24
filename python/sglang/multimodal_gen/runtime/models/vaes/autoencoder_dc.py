# SPDX-License-Identifier: Apache-2.0
"""
AutoencoderDC (DC-AE) wrapper for Sana.

This module provides a wrapper around diffusers' AutoencoderDC which uses
Deep Compression (32x spatial compression) instead of standard KL-VAE (8x).

Architecture:
- Uses EfficientViT blocks for efficient feature extraction
- Residual blocks for downsampling/upsampling
- 32x spatial compression (vs 8x in standard VAE)
- 32 latent channels

Reference: https://arxiv.org/abs/2410.10629
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.vaes.sana import SanaVAEConfig


class AutoencoderDC(nn.Module):
    """Deep Compression Autoencoder for Sana.

    This is a wrapper that provides the interface expected by sglang pipelines
    while delegating to diffusers' AutoencoderDC implementation.

    DC-AE achieves 32x spatial compression compared to 8x in standard VAEs,
    enabling much higher resolution generation with the same compute budget.
    """

    _supports_gradient_checkpointing = True

    def __init__(self, config: SanaVAEConfig):
        super().__init__()
        self.config = config
        self.arch_config = config.arch_config

        # Try to import diffusers' AutoencoderDC
        try:
            from diffusers import AutoencoderDC as DiffusersAutoencoderDC
        except ImportError:
            raise ImportError(
                "AutoencoderDC requires diffusers >= 0.32.0. "
                "Please upgrade: pip install -U diffusers"
            )

        # Build the diffusers model
        self._model = DiffusersAutoencoderDC(
            in_channels=self.arch_config.in_channels,
            out_channels=self.arch_config.out_channels,
            latent_channels=self.arch_config.latent_channels,
            encoder_block_types=self.arch_config.encoder_block_types,
            decoder_block_types=self.arch_config.decoder_block_types,
            encoder_block_out_channels=self.arch_config.encoder_block_out_channels,
            decoder_block_out_channels=self.arch_config.decoder_block_out_channels,
            encoder_layers_per_block=self.arch_config.encoder_layers_per_block,
            decoder_layers_per_block=self.arch_config.decoder_layers_per_block,
            encoder_qkv_multiscales=self.arch_config.encoder_qkv_multiscales,
            decoder_qkv_multiscales=self.arch_config.decoder_qkv_multiscales,
            scaling_factor=self.arch_config.scaling_factor,
        )

        # Settings for pipeline compatibility
        self.use_slicing = False
        self.use_tiling = False

        # Scale factors
        self.scaling_factor = self.arch_config.scaling_factor
        self.latent_channels = self.arch_config.latent_channels

    @property
    def device(self) -> torch.device:
        """Return the device of the model."""
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the dtype of the model."""
        return next(self.parameters()).dtype

    def enable_tiling(self, use_tiling: bool = True):
        """Enable tiled VAE decoding."""
        self.use_tiling = use_tiling
        if hasattr(self._model, "enable_tiling"):
            if use_tiling:
                self._model.enable_tiling()
            else:
                self._model.disable_tiling()

    def disable_tiling(self):
        """Disable tiled VAE decoding."""
        self.enable_tiling(False)

    def enable_slicing(self):
        """Enable sliced VAE decoding."""
        self.use_slicing = True
        if hasattr(self._model, "enable_slicing"):
            self._model.enable_slicing()

    def disable_slicing(self):
        """Disable sliced VAE decoding."""
        self.use_slicing = False
        if hasattr(self._model, "disable_slicing"):
            self._model.disable_slicing()

    def encode(
        self, x: torch.Tensor, return_dict: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Encode images into latent representations.

        Args:
            x: Input images with shape [B, C, H, W]
            return_dict: Whether to return a dict (for compatibility)

        Returns:
            Latent representation with shape [B, latent_channels, H/32, W/32]
        """
        # DC-AE outputs latents directly (no distribution sampling needed)
        latents = self._model.encode(x).latent

        # Apply scaling factor
        latents = latents * self.scaling_factor

        if return_dict:
            return {"latent": latents}
        return (latents,)

    def decode(
        self, z: torch.Tensor, return_dict: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Decode latent representations into images.

        Args:
            z: Latent tensor with shape [B, latent_channels, H, W]
            return_dict: Whether to return a dict (for compatibility)

        Returns:
            Decoded images with shape [B, 3, H*32, W*32]
        """
        # Unscale latents
        z = z / self.scaling_factor

        # Decode using diffusers model
        decoded = self._model.decode(z).sample

        if return_dict:
            return {"sample": decoded}
        return decoded

    def forward(
        self,
        x: torch.Tensor,
        sample_posterior: bool = False,  # Unused, kept for API compatibility
        return_dict: bool = True,
        generator: Optional[
            torch.Generator
        ] = None,  # Unused, kept for API compatibility
    ) -> Union[torch.Tensor, Tuple[torch.Tensor]]:
        """
        Forward pass: encode then decode.

        Args:
            x: Input images
            sample_posterior: Unused (DC-AE doesn't have stochastic encoding)
            return_dict: Whether to return a dict
            generator: Unused

        Returns:
            Reconstructed images
        """
        # Encode
        encoded = self.encode(x, return_dict=False)
        if isinstance(encoded, tuple):
            z = encoded[0]
        else:
            z = encoded

        # Decode
        return self.decode(z, return_dict=return_dict)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[SanaVAEConfig] = None,
        **kwargs
    ):
        """
        Load a pretrained AutoencoderDC model.

        Args:
            pretrained_model_name_or_path: HuggingFace repo ID or local path
            config: Optional config override
            **kwargs: Additional arguments for from_pretrained

        Returns:
            AutoencoderDC instance with loaded weights
        """
        from diffusers import AutoencoderDC as DiffusersAutoencoderDC

        if config is None:
            config = SanaVAEConfig()

        # Create instance
        model = cls(config)

        # Load weights from diffusers
        diffusers_model = DiffusersAutoencoderDC.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )

        # Replace internal model
        model._model = diffusers_model

        return model
