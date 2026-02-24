# SPDX-License-Identifier: Apache-2.0
"""
Sana pipeline configuration.

Sana is a text-to-image model using DC-AE (32x compression), Gemma2 text encoder,
and Linear Attention DiT with flow matching.
Reference: https://huggingface.co/papers/2410.10629
"""

from dataclasses import dataclass, field
from typing import Callable

import torch

from sglang.multimodal_gen.configs.models import DiTConfig, EncoderConfig, VAEConfig
from sglang.multimodal_gen.configs.models.dits.sana import SanaConfig
from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma_2 import Gemma2Config
from sglang.multimodal_gen.configs.models.vaes.sana import SanaVAEConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import (
    ImagePipelineConfig,
    ModelTaskType,
)


def sana_preprocess_text(prompt: str) -> str:
    """Preprocess text prompt for Sana.

    Sana uses a complex human instruction format with in-context learning
    to enhance image-text alignment.
    """
    # Sana's default complex human instruction template
    # This is the simplified version; full version includes more detailed instructions
    return prompt


def sana_postprocess_text(outputs: BaseEncoderOutput, _text_inputs) -> torch.Tensor:
    """Extract text embeddings from Gemma2 decoder outputs.

    Sana uses the last hidden state from the decoder-only Gemma2 model.
    """
    # Return hidden states (last layer)
    # Shape: [batch, seq_len, hidden_size]
    return outputs.last_hidden_state


@dataclass
class SanaPipelineConfig(ImagePipelineConfig):
    """Configuration for Sana text-to-image pipeline."""

    # Pipeline task type
    task_type: ModelTaskType = ModelTaskType.T2I

    # Guidance configuration
    should_use_guidance: bool = True
    embedded_cfg_scale: float = 4.5  # Sana default guidance scale

    # Flow matching parameters
    flow_shift: float = 3.0  # Sana uses flow_shift=3.0

    # Model configurations
    dit_config: DiTConfig = field(default_factory=SanaConfig)
    vae_config: VAEConfig = field(default_factory=SanaVAEConfig)

    # Single text encoder (Gemma2)
    text_encoder_configs: tuple[EncoderConfig, ...] = field(
        default_factory=lambda: (Gemma2Config(),)
    )
    text_encoder_precisions: tuple[str, ...] = field(default_factory=lambda: ("bf16",))

    # Text preprocessing/postprocessing
    preprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (sana_preprocess_text,)
    )
    postprocess_text_funcs: tuple[Callable, ...] = field(
        default_factory=lambda: (sana_postprocess_text,)
    )

    # VAE settings - disable tiling for image generation
    vae_tiling: bool = False
    vae_sp: bool = False

    # Autocast disabled for Sana (uses explicit dtypes)
    enable_autocast: bool = False

    def prepare_sigmas(self, sigmas, num_inference_steps):
        """Prepare sigmas for flow matching with shift."""
        import numpy as np

        # Flow matching sigmas with shift
        alphas = np.linspace(1.0, 1.0 / 1000, num_inference_steps + 1)
        sigmas = 1.0 - alphas
        # Apply flow shift
        sigmas = self.flow_shift * sigmas / (1 + (self.flow_shift - 1) * sigmas)
        sigmas = np.flip(sigmas)[:-1]
        return sigmas

    def prepare_latent_shape(self, batch, batch_size, num_frames):
        """Prepare latent tensor shape for Sana.

        Sana uses 32x spatial compression (DC-AE).
        """
        vae_scale_factor = self.vae_config.arch_config.vae_scale_factor
        height = batch.height // vae_scale_factor
        width = batch.width // vae_scale_factor
        num_channels_latents = self.dit_config.arch_config.in_channels

        # Sana uses spatial format: [B, C, H, W]
        shape = (batch_size, num_channels_latents, height, width)
        return shape

    def maybe_pack_latents(self, latents, batch_size, batch):
        """Pack latents for transformer input.

        Sana uses patch_size=1, so we just flatten spatial dims to sequence.
        """
        # latents: [B, C, H, W] -> [B, H*W, C] (sequence format)
        batch_size, channels, height, width = latents.shape
        latents = latents.permute(0, 2, 3, 1)  # [B, H, W, C]
        latents = latents.reshape(batch_size, height * width, channels)  # [B, S, C]

        # Store raw shape for unpacking
        batch.raw_latent_shape = (batch_size, channels, 1, height, width)

        return latents

    def get_pos_prompt_embeds(self, batch):
        """Get positive prompt embeddings."""
        return batch.prompt_embeds

    def get_neg_prompt_embeds(self, batch):
        """Get negative prompt embeddings."""
        if (
            hasattr(batch, "negative_prompt_embeds")
            and batch.negative_prompt_embeds is not None
        ):
            return batch.negative_prompt_embeds
        return None

    def tokenize_prompt(self, prompts: list[str], tokenizer, tok_kwargs) -> dict:
        """Tokenize prompts for Gemma2.

        Sana uses right-padding for decoder-only models.
        """
        # Set padding side to right for decoder-only model
        if hasattr(tokenizer, "padding_side"):
            original_padding_side = tokenizer.padding_side
            tokenizer.padding_side = "right"

        max_length = tok_kwargs.get("max_length", 300)

        inputs = tokenizer(
            prompts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
            return_attention_mask=True,
        )

        # Restore original padding side
        if hasattr(tokenizer, "padding_side"):
            tokenizer.padding_side = original_padding_side

        return inputs

    def post_denoising_loop(self, latents, batch):
        """Post-process latents after denoising loop.

        Unpack from sequence format back to spatial format.
        """
        # latents: [B, S, C] -> [B, C, H, W]
        raw_shape = getattr(batch, "raw_latent_shape", None)
        if raw_shape is not None:
            batch_size, channels, _, height, width = raw_shape
            latents = latents.reshape(batch_size, height, width, channels)
            latents = latents.permute(0, 3, 1, 2)  # [B, C, H, W]
        return latents

    def prepare_pos_cond_kwargs(self, batch):
        """Prepare positive conditioning kwargs for DiT forward."""
        kwargs = {}

        # Prompt embeddings
        prompt_embeds = self.get_pos_prompt_embeds(batch)
        if prompt_embeds is not None:
            kwargs["encoder_hidden_states"] = prompt_embeds

        # Attention mask for cross-attention
        if hasattr(batch, "prompt_attention_mask"):
            kwargs["encoder_attention_mask"] = batch.prompt_attention_mask

        return kwargs

    def prepare_neg_cond_kwargs(self, batch):
        """Prepare negative conditioning kwargs for DiT forward."""
        kwargs = {}

        # Negative prompt embeddings
        neg_embeds = self.get_neg_prompt_embeds(batch)
        if neg_embeds is not None:
            kwargs["encoder_hidden_states"] = neg_embeds

        if hasattr(batch, "negative_prompt_attention_mask"):
            kwargs["encoder_attention_mask"] = batch.negative_prompt_attention_mask

        return kwargs
