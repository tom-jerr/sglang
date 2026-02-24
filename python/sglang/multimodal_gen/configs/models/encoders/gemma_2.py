# SPDX-License-Identifier: Apache-2.0
"""
Gemma2 text encoder configuration for Sana.

Sana uses Gemma2-2b-it as a decoder-only text encoder, extracting hidden states
for conditioning the diffusion model.
Reference: https://huggingface.co/papers/2410.10629
"""

from dataclasses import dataclass, field

from sglang.multimodal_gen.configs.models.encoders.base import (
    TextEncoderArchConfig,
    TextEncoderConfig,
)


def _is_transformer_layer(n: str, m) -> bool:
    """Check if module is a transformer layer for FSDP sharding."""
    return "layers" in n and n.split(".")[-1].isdigit()


def _is_embeddings(n: str, m) -> bool:
    """Check if module is embedding layer."""
    return "embed_tokens" in n


def _is_final_layernorm(n: str, m) -> bool:
    """Check if module is final layer norm."""
    return n.endswith("norm")


@dataclass
class Gemma2ArchConfig(TextEncoderArchConfig):
    """Architecture configuration for Gemma2 text encoder."""

    # Model dimensions (Gemma2-2b-it)
    vocab_size: int = 256000
    hidden_size: int = 2304
    intermediate_size: int = 9216
    num_hidden_layers: int = 26
    num_attention_heads: int = 8
    num_key_value_heads: int = 4
    head_dim: int = 256

    # Attention configuration
    attention_bias: bool = False
    attention_dropout: float = 0.0
    max_position_embeddings: int = 8192
    sliding_window: int = 4096
    query_pre_attn_scalar: int = 224  # sqrt(head_dim) for stable training

    # RoPE configuration
    rope_theta: float = 10000.0

    # MLP configuration
    hidden_act: str = "gelu_pytorch_tanh"
    hidden_activation: str = "gelu_pytorch_tanh"

    # Normalization
    rms_norm_eps: float = 1e-6
    attn_logit_softcapping: float = 50.0
    final_logit_softcapping: float = 30.0

    # Token IDs
    pad_token_id: int = 0
    eos_token_id: int = 1
    bos_token_id: int = 2

    # Text length
    text_len: int = 300  # Sana's default max_sequence_length

    # FSDP sharding conditions
    _fsdp_shard_conditions: list = field(
        default_factory=lambda: [
            _is_transformer_layer,
            _is_embeddings,
            _is_final_layernorm,
        ]
    )

    # Weight stacking for efficient loading
    stacked_params_mapping: list[tuple[str, str, str]] = field(
        default_factory=lambda: [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", "gate"),
            (".gate_up_proj", ".up_proj", "up"),
        ]
    )

    def __post_init__(self):
        super().__post_init__()
        self.tokenizer_kwargs = {
            "padding": "max_length",
            "truncation": True,
            "max_length": self.text_len,
            "add_special_tokens": True,
            "return_attention_mask": True,
            "return_tensors": "pt",
        }


@dataclass
class Gemma2Config(TextEncoderConfig):
    """Configuration for Gemma2 text encoder."""

    arch_config: TextEncoderArchConfig = field(default_factory=Gemma2ArchConfig)
    prefix: str = "gemma_2"
