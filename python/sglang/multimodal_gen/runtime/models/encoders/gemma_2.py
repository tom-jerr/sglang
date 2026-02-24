# SPDX-License-Identifier: Apache-2.0
# Adapted from Gemma3 encoder and Llama encoder in sglang
"""
Gemma2 text encoder for Sana.

Gemma2 is a decoder-only transformer used by Sana as a text encoder.
Unlike CLIP/T5 which are encoder models, Gemma2 produces text embeddings
from the last hidden states of a causal language model.

Key features:
- GeGLU activation (gated GeLU)
- Sliding window attention
- Pre-normalization with RMS norm
- Soft capping for attention logits

Reference: https://arxiv.org/abs/2408.00118
"""

from collections.abc import Iterable
from typing import Any, Optional

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.models.encoders.gemma_2 import Gemma2Config
from sglang.multimodal_gen.runtime.distributed import get_tp_world_size
from sglang.multimodal_gen.runtime.layers.activation import GeluAndMul
from sglang.multimodal_gen.runtime.layers.attention import LocalAttention
from sglang.multimodal_gen.runtime.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig
from sglang.multimodal_gen.runtime.layers.rotary_embedding import get_rope
from sglang.multimodal_gen.runtime.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from sglang.multimodal_gen.runtime.loader.weight_utils import default_weight_loader
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder


class Gemma2RMSNorm(nn.Module):
    """RMSNorm with +1 weight initialization like Gemma models."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        output = self._norm(x.float())
        output = output * (1.0 + self.weight.float())
        return output.type_as(x), residual


class Gemma2MLP(nn.Module):
    """Gemma2 MLP with GeGLU activation."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        # Gemma2 uses gelu_pytorch_tanh
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class Gemma2Attention(nn.Module):
    """Gemma2 attention with sliding window support."""

    def __init__(
        self,
        layer_id: int,
        config: Gemma2Config,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        rope_theta: float,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.hidden_size = hidden_size
        self.head_dim = head_dim

        tp_size = get_tp_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size

        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.arch_config.query_pre_attn_scalar**-0.5

        # Attention softcapping
        self.attn_logit_softcapping = config.arch_config.attn_logit_softcapping

        # Sliding window (every other layer in Gemma2)
        sliding_window = config.arch_config.sliding_window
        self.is_sliding = layer_id % 2 == 0 and sliding_window is not None
        self.effective_window = sliding_window - 1 if self.is_sliding else None

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        # Rotary embeddings
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            is_neox_style=True,
        )

        # Local attention implementation
        self.attn = LocalAttention(
            num_heads=self.num_heads,
            head_size=self.head_dim,
            num_kv_heads=self.num_kv_heads,
            scaling=self.scaling,
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply rotary embeddings
        q, k = self.rotary_emb(position_ids, q, k)

        # Apply attention
        attn_output = self.attn(
            q,
            k,
            v,
            attention_mask=attention_mask,
            softcap=self.attn_logit_softcapping,
        )

        output, _ = self.o_proj(attn_output)
        return output


class Gemma2DecoderLayer(nn.Module):
    """Gemma2 decoder layer with pre/post attention and FFN layer norms."""

    def __init__(
        self,
        layer_id: int,
        config: Gemma2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        arch = config.arch_config

        self.self_attn = Gemma2Attention(
            layer_id=layer_id,
            config=config,
            hidden_size=arch.hidden_size,
            num_heads=arch.num_attention_heads,
            num_kv_heads=arch.num_key_value_heads,
            head_dim=arch.head_dim,
            max_position_embeddings=arch.max_position_embeddings,
            rope_theta=arch.rope_theta,
            quant_config=quant_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = Gemma2MLP(
            hidden_size=arch.hidden_size,
            intermediate_size=arch.intermediate_size,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )

        # Pre and post layer norms for attention
        self.input_layernorm = Gemma2RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)
        self.post_attention_layernorm = Gemma2RMSNorm(
            arch.hidden_size, eps=arch.rms_norm_eps
        )

        # Pre and post layer norms for FFN
        self.pre_feedforward_layernorm = Gemma2RMSNorm(
            arch.hidden_size, eps=arch.rms_norm_eps
        )
        self.post_feedforward_layernorm = Gemma2RMSNorm(
            arch.hidden_size, eps=arch.rms_norm_eps
        )

    def forward(
        self,
        position_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Pre-attention layernorm
        hidden_states, residual = self.input_layernorm(hidden_states, residual)

        # Self attention
        attn_output = self.self_attn(position_ids, hidden_states, attention_mask)

        # Post-attention layernorm
        attn_output, _ = self.post_attention_layernorm(attn_output)
        hidden_states = residual + attn_output
        residual = hidden_states

        # Pre-FFN layernorm
        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )

        # FFN
        mlp_output = self.mlp(hidden_states)

        # Post-FFN layernorm
        mlp_output, _ = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states, hidden_states


class Gemma2Model(TextEncoder):
    """Gemma2 decoder-only model used as text encoder for Sana."""

    def __init__(
        self,
        config: Gemma2Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config)
        self.config = config
        arch = config.arch_config

        self.padding_idx = arch.pad_token_id

        # Token embeddings
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=arch.vocab_size,
            embedding_dim=arch.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [
                Gemma2DecoderLayer(
                    layer_id=i,
                    config=config,
                    quant_config=quant_config,
                    prefix=f"{prefix}.layers.{i}",
                )
                for i in range(arch.num_hidden_layers)
            ]
        )

        # Final layer norm
        self.norm = Gemma2RMSNorm(arch.hidden_size, eps=arch.rms_norm_eps)

        # Normalizer for embeddings
        self.normalizer = arch.hidden_size**0.5

    def get_input_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings with Gemma2 scaling."""
        embeds = self.embed_tokens(input_ids)
        # Gemma2 scales embeddings by sqrt(hidden_size)
        return embeds * self.normalizer

    def forward(
        self,
        input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs,
    ) -> BaseEncoderOutput:
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )

        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.get_input_embeddings(input_ids)

        residual = None

        if position_ids is None:
            position_ids = torch.arange(
                0, hidden_states.shape[1], device=hidden_states.device
            ).unsqueeze(0)

        all_hidden_states: tuple[Any, ...] | None = () if output_hidden_states else None

        for layer in self.layers:
            if all_hidden_states is not None:
                all_hidden_states += (
                    (hidden_states,)
                    if residual is None
                    else (hidden_states + residual,)
                )

            hidden_states, residual = layer(
                position_ids,
                hidden_states,
                residual,
                attention_mask,
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if all_hidden_states is not None:
            all_hidden_states += (hidden_states,)

        return BaseEncoderOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """Load weights from pretrained checkpoint."""
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        stacked_params_mapping = self.config.arch_config.stacked_params_mapping

        for name, loaded_weight in weights:
            # Skip rotary embeddings
            if "rotary_emb" in name:
                continue

            # Handle stacked parameters (qkv_proj, gate_up_proj)
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)

                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)

            loaded_params.add(name)

        return loaded_params


# Entry point for model loading
EntryClass = Gemma2Model
