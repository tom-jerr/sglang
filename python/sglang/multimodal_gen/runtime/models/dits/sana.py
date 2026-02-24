# SPDX-License-Identifier: Apache-2.0
"""
SanaTransformer2DModel for Sana image generation.

Sana uses a Linear Attention DiT architecture with:
- Linear attention (O(n) complexity instead of O(n²))
- GLUMBConv FFN (gated linear unit with MBConv)
- AdaLN-single timestep conditioning
- Cross-attention for text conditioning

Reference: https://arxiv.org/abs/2410.10629
"""

from typing import Any, Optional

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.runtime.models.dits.base import CachableDiT


class SanaTransformer2DModel(CachableDiT):
    """
    Sana Transformer using Linear Attention for efficient image generation.

    This is a wrapper around diffusers' SanaTransformer2DModel that provides
    the interface expected by sglang pipelines. For native performance,
    a full sglang implementation can be added later.

    Key features:
    - Linear attention with O(n) complexity
    - GLUMBConv (Gated Linear Unit + Mobile Block Conv) FFN
    - AdaLN-single for efficient timestep conditioning
    - Cross-attention with text embeddings from Gemma2
    """

    # Required class attributes
    _fsdp_shard_conditions = []
    _compile_conditions = []
    param_names_mapping = {}
    reverse_param_names_mapping = {}

    def __init__(
        self,
        config: SanaConfig,
        hf_config: dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__(config, hf_config, **kwargs)

        arch = config.arch_config

        # Required instance attributes
        self.hidden_size = arch.num_attention_heads * arch.attention_head_dim
        self.num_attention_heads = arch.num_attention_heads
        self.num_channels_latents = arch.in_channels

        # Store config
        self.patch_size = arch.patch_size
        self.in_channels = arch.in_channels
        self.caption_channels = arch.caption_channels
        self.num_layers = arch.num_layers

        # Try to import diffusers' SanaTransformer2DModel
        try:
            from diffusers.models.transformers.sana_transformer import (
                SanaTransformer2DModel as DiffusersSanaTransformer,
            )
        except ImportError:
            raise ImportError(
                "SanaTransformer2DModel requires diffusers >= 0.32.0. "
                "Please upgrade: pip install -U diffusers"
            )

        # Build the diffusers model
        self._model = DiffusersSanaTransformer(
            in_channels=arch.in_channels,
            out_channels=arch.in_channels,
            num_attention_heads=arch.num_attention_heads,
            attention_head_dim=arch.attention_head_dim,
            num_layers=arch.num_layers,
            num_cross_attention_heads=arch.num_cross_attention_heads,
            cross_attention_head_dim=arch.cross_attention_head_dim,
            cross_attention_dim=arch.cross_attention_dim,
            caption_channels=arch.caption_channels,
            mlp_ratio=arch.mlp_ratio,
            patch_size=arch.patch_size,
            sample_size=arch.sample_size,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        guidance: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Forward pass of Sana transformer.

        Args:
            hidden_states: Latent tensor of shape [B, C, H, W] or [B, S, C]
            encoder_hidden_states: Text embeddings from Gemma2 [B, L, D]
            timestep: Timestep tensor [B]
            encoder_attention_mask: Attention mask for text [B, L]
            guidance: Optional guidance scale (for CFG)

        Returns:
            Predicted noise tensor
        """
        # Handle sequence format input
        input_is_sequence = hidden_states.dim() == 3
        if input_is_sequence:
            # [B, S, C] -> [B, C, H, W]
            batch_size, seq_len, channels = hidden_states.shape
            height = width = int(seq_len**0.5)
            hidden_states = hidden_states.permute(0, 2, 1)  # [B, C, S]
            hidden_states = hidden_states.reshape(batch_size, channels, height, width)

        # Normalize timestep to [0, 1] if needed
        if timestep.max() > 1.0:
            timestep = timestep / 1000.0

        # Call diffusers model
        output = self._model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            encoder_attention_mask=encoder_attention_mask,
            return_dict=False,
        )

        if isinstance(output, tuple):
            output = output[0]

        # Convert back to sequence format if input was sequence
        if input_is_sequence:
            batch_size, channels, height, width = output.shape
            output = output.reshape(batch_size, channels, -1)  # [B, C, S]
            output = output.permute(0, 2, 1)  # [B, S, C]

        return output

    def load_weights(self, weights) -> set[str]:
        """
        Load weights from a checkpoint.

        This wrapper delegates to the diffusers model's state dict loading.
        """
        loaded_params = set()

        # Convert weights iterator to state dict
        if hasattr(weights, "__iter__") and not isinstance(weights, dict):
            state_dict = {}
            for name, tensor in weights:
                state_dict[name] = tensor
        else:
            state_dict = weights

        # Load into the diffusers model
        missing, unexpected = self._model.load_state_dict(state_dict, strict=False)

        # Track loaded parameters
        loaded_params = set(state_dict.keys()) - set(missing)

        return loaded_params

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[SanaConfig] = None,
        **kwargs,
    ):
        """
        Load a pretrained SanaTransformer2DModel.

        Args:
            pretrained_model_name_or_path: HuggingFace repo ID or local path
            config: Optional config override
            **kwargs: Additional arguments for from_pretrained

        Returns:
            SanaTransformer2DModel instance with loaded weights
        """
        from diffusers.models.transformers.sana_transformer import (
            SanaTransformer2DModel as DiffusersSanaTransformer,
        )

        if config is None:
            config = SanaConfig()

        # Create instance
        hf_config = {}
        model = cls(config, hf_config)

        # Load weights from diffusers
        diffusers_model = DiffusersSanaTransformer.from_pretrained(
            pretrained_model_name_or_path,
            **kwargs,
        )

        # Replace internal model
        model._model = diffusers_model

        return model


# Entry point for model loading
EntryClass = SanaTransformer2DModel
