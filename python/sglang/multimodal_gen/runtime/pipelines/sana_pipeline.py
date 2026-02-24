# SPDX-License-Identifier: Apache-2.0
"""
Sana Pipeline for text-to-image generation.

Sana uses:
- DC-AE (32x spatial compression)
- Gemma2 text encoder
- Linear Attention DiT
- Flow matching with DPM-solver

Reference: https://arxiv.org/abs/2410.10629
"""

from sglang.multimodal_gen.runtime.pipelines_core import Req
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.server_args import ServerArgs
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger

logger = init_logger(__name__)


def calculate_sana_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    """Calculate flow shift based on image sequence length.

    Sana uses a linear interpolation of shift values based on sequence length.
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def prepare_mu(batch: Req, server_args: ServerArgs):
    """Prepare mu (flow shift) parameter for Sana.

    Sana uses 32x spatial compression (DC-AE), so we calculate
    sequence length based on that compression ratio.
    """
    height = batch.height
    width = batch.width

    # Sana uses 32x compression + patch_size=1
    vae_scale_factor = (
        server_args.pipeline_config.vae_config.arch_config.vae_scale_factor
    )
    patch_size = server_args.pipeline_config.dit_config.arch_config.patch_size

    # Calculate image sequence length
    # With 32x compression and patch_size=1: seq_len = (H/32) * (W/32)
    h_tokens = height // vae_scale_factor // patch_size
    w_tokens = width // vae_scale_factor // patch_size
    image_seq_len = h_tokens * w_tokens

    # Calculate shift
    mu = calculate_sana_shift(image_seq_len)

    return "mu", mu


class SanaPipeline(ComposedPipelineBase):
    """Pipeline for Sana text-to-image generation.

    This pipeline implements the standard T2I flow:
    1. Text encoding with Gemma2
    2. Latent preparation
    3. Timestep preparation with flow shift
    4. Denoising with Linear Attention DiT
    5. Decoding with DC-AE
    """

    pipeline_name = "SanaPipeline"

    _required_config_modules = [
        "text_encoder",
        "tokenizer",
        "vae",
        "transformer",
        "scheduler",
    ]

    def create_pipeline_stages(self, server_args: ServerArgs):
        """Create the pipeline stages for Sana.

        Uses the standard T2I stages with Sana-specific flow shift calculation.
        """
        self.add_standard_t2i_stages(prepare_extra_timestep_kwargs=[prepare_mu])


# Entry point for pipeline loading
EntryClass = SanaPipeline
