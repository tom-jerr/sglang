# SPDX-License-Identifier: Apache-2.0
"""
Sana sampling parameters.

Sana uses flow matching with DPM-solver for efficient sampling.
Default config follows the paper recommendations.
"""

from dataclasses import dataclass

from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams


@dataclass
class SanaSamplingParams(SamplingParams):
    """Sampling parameters for Sana text-to-image generation."""

    # Default 20 steps for flow matching with DPM-solver
    num_inference_steps: int = 20

    # Single frame for T2I
    num_frames: int = 1

    # Negative prompt (optional, empty by default)
    negative_prompt: str = ""

    # Classifier-free guidance scale (Sana default: 4.5)
    guidance_scale: float = 4.5

    # CFG normalization disabled by default for Sana
    cfg_normalization: float | bool = False
