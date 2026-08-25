from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
from sglang.kernels.ops.attention.fla.fused_gdn_gating import fused_gdn_gating
from sglang.kernels.ops.mamba.causal_conv1d_triton import (
    causal_conv1d_fn,
    causal_conv1d_update,
)
from sglang.srt.configs.hybrid_arch import hybrid_gdn_config
from sglang.srt.layers.attention.hybrid_linear_attn_backend import MambaAttnBackendBase
from sglang.srt.layers.attention.linear.kernels.gdn_triton import TritonGDNKernel
from sglang.srt.layers.attention.linear.utils import (
    LinearAttnBackends,
    LinearAttnKernelBackend,
    build_verify_intermediate_state_indices,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.mem_cache.memory_pool import MambaPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.runtime_context import (
    get_exec,
    get_memory,
    get_schedule,
    mamba_cache_chunk_size,
)
from sglang.srt.utils import is_cpu, is_cuda, is_hip, is_npu, is_xpu
from sglang.srt.utils.common import rank0_log

_is_hip = is_hip()

if not is_cpu():
    from sglang.kernels.ops.attention.fla.chunk_delta_h import (
        CHUNK_SIZE as FLA_CHUNK_SIZE,
    )
else:
    FLA_CHUNK_SIZE = 64

if is_cuda() or is_hip() or is_xpu():
    from sglang.kernels.ops.attention.triton_gdn_fused_proj import (
        fused_qkv_split_gdn_prefill,
    )

MAX_FUSED_QKV_SPLIT_DIM = 8192


def _gdn_bcg_chunk_plan_capacity(num_tokens: int, max_real_requests: int) -> int:
    """Maximum number of FLA chunks for ``num_tokens`` over bounded requests."""
    if num_tokens <= 0:
        return 1
    max_real_requests = max(1, min(max_real_requests, num_tokens))
    return min(
        num_tokens,
        (num_tokens + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE + max_real_requests - 1,
    )


def _build_gdn_bcg_chunk_plan(
    seq_lens: Sequence[int], *, capacity: int, dummy_sequence: int
) -> torch.Tensor:
    """Build a fixed-size FLA ``(sequence, chunk)`` plan on CPU.

    Unused rows point at a zero-length dummy sequence. FLA kernels already
    mask zero-length variable-length sequences, so the graph launch topology
    stays fixed while the request layout changes at replay.
    """
    rows = [
        (seq_id, chunk_id)
        for seq_id, seq_len in enumerate(seq_lens)
        for chunk_id in range((int(seq_len) + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE)
    ]
    if len(rows) > capacity:
        raise ValueError(
            f"GDN BCG chunk plan needs {len(rows)} rows, capacity is {capacity}"
        )
    rows.extend([(dummy_sequence, 0)] * (capacity - len(rows)))
    return torch.tensor(rows, dtype=torch.int32)


if is_cuda():
    from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
        fused_mamba_state_scatter_with_mask,
        scatter_gdn_prefill_conv_states_with_mask,
        scatter_gdn_prefill_states_with_mask,
    )
    from sglang.srt.layers.attention.mamba.causal_conv1d import (
        causal_conv1d_fn as causal_conv1d_fn_cuda,
    )

    causal_conv1d_fn = causal_conv1d_fn_cuda
elif is_npu():
    from sgl_kernel_npu.fla.fused_gdn_gating import fused_gdn_gating_npu
    from sgl_kernel_npu.mamba.causal_conv1d import (
        causal_conv1d_fn_npu,
        causal_conv1d_update_npu,
    )

    fused_gdn_gating = fused_gdn_gating_npu
    causal_conv1d_fn = causal_conv1d_fn_npu
    causal_conv1d_update = causal_conv1d_update_npu
elif is_cpu():
    from sgl_kernel.mamba import causal_conv1d_fn_cpu, causal_conv1d_update_cpu

    causal_conv1d_fn = causal_conv1d_fn_cpu
    causal_conv1d_update = causal_conv1d_update_cpu
    fused_gdn_gating = torch.ops.sgl_kernel.fused_gdn_gating_cpu


def flashinfer_gdn_prefill_default(model_runner: ModelRunner) -> Optional[str]:
    """FlashInfer for the narrow SM90/SM100 GDN prefill domains we validated, else None."""
    sm_major = torch.cuda.get_device_capability()[0] if is_cuda() else 0
    if (
        get_exec().mamba.linear_attn_prefill_backend is not None
        or get_exec().mamba.linear_attn_backend != "triton"
        or get_exec().deterministic.enable_deterministic_inference
        or get_memory().enable_page_major_kv_layout
        or sm_major not in (9, 10)
    ):
        return None

    # SM100 runs the CUDA>=13 CuTe-DSL chunk kernel on a bf16 state pool;
    # SM90 runs the fused Hopper kernel on an fp32 state pool and tolerates
    # larger chunks. Everything outside these validated domains keeps Triton.
    cuda_version = torch.version.cuda
    if sm_major == 10:
        if cuda_version is None or int(cuda_version.split(".", 1)[0]) < 13:
            return None
        max_chunk = 8192
        expected_state_dtype = torch.bfloat16
    else:
        max_chunk = 32768
        expected_state_dtype = torch.float32

    chunk_size = get_schedule().chunked_prefill_size
    config = hybrid_gdn_config(model_runner.model_config)
    if (
        get_schedule().enable_dynamic_chunking
        or chunk_size is None
        or not 1 <= chunk_size <= max_chunk
        or getattr(config, "linear_key_head_dim", None) != 128
        or getattr(config, "linear_value_head_dim", None) != 128
        or model_runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.dtype
        != expected_state_dtype
    ):
        return None

    from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
        is_flashinfer_gdn_prefill_available,
    )

    if not is_flashinfer_gdn_prefill_available():
        return None

    rank0_log(f"Defaulting SM{sm_major}0 GDN prefill backend to FlashInfer.")
    return "flashinfer"


def _validate_gdn_linear_attn_backends(backends: LinearAttnBackends) -> None:
    if (
        get_exec().deterministic.enable_deterministic_inference
        and backends.prefill.is_flashinfer()
    ):
        raise ValueError(
            "FlashInfer GDN prefill is not supported with "
            "--enable-deterministic-inference. Use "
            "--linear-attn-prefill-backend triton."
        )


class GDNKernelDispatcher:
    """Dispatches GDN kernel calls to the appropriate backend per mode."""

    def __init__(
        self,
        decode_backend: LinearAttnKernelBackend,
        prefill_backend: LinearAttnKernelBackend,
        verify_backend: Optional[LinearAttnKernelBackend] = None,
    ):
        triton_kernel = TritonGDNKernel()
        self.tree_verify_kernel = triton_kernel

        cutedsl_kernel = None
        if decode_backend.is_triton():
            self.decode_kernel = triton_kernel
        elif decode_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # The fused SYCL kernel is dispatched via XpuGDNAttnBackend.forward_fused_gdn,
            # outside this dispatcher; Triton is the dispatcher-level kernel for requests
            # that hook doesn't handle (e.g. verify).
            self.decode_kernel = triton_kernel
        elif decode_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                CuteDSLGDNKernel,
            )

            cutedsl_kernel = CuteDSLGDNKernel()
            self.decode_kernel = cutedsl_kernel
        elif decode_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                FlashInferGDNKernel,
            )

            flashinfer_kernel = FlashInferGDNKernel()
            self.decode_kernel = flashinfer_kernel
        elif decode_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN decode backend: {decode_backend}")

        if prefill_backend.is_triton():
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_intel_xpu():
            if not is_xpu():
                raise ValueError("--linear-attn-backend intel_xpu requires Intel XPU")
            # See the decode branch above: intel_xpu uses Triton as its
            # dispatcher-level fallback kernel.
            self.extend_kernel = triton_kernel
        elif prefill_backend.is_cutedsl():
            if not is_cuda():
                raise ValueError("GDN CuTe DSL backend requires CUDA")
            # Reuse the CuteDSL kernel if already created for decode
            if cutedsl_kernel is None:
                from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import (
                    CuteDSLGDNKernel,
                )

                cutedsl_kernel = CuteDSLGDNKernel()
            # The CuteDSL prefill kernel only exists on SM100+ (Blackwell).
            # On SM90 (Hopper) fall back to Triton so users can pick
            # `cutedsl` uniformly across hardware.
            if cutedsl_kernel.supports_prefill:
                self.extend_kernel = cutedsl_kernel
            else:
                rank0_log(
                    "CuTe DSL GDN prefill is not supported on this GPU "
                    "(requires SM100+). Falling back to Triton for prefill."
                )
                self.extend_kernel = triton_kernel
        elif prefill_backend.is_flashinfer():
            if not is_cuda():
                raise ValueError("FlashInfer GDN backend requires CUDA")
            # Reuse the FlashInfer kernel if already created for decode
            if decode_backend.is_flashinfer():
                self.extend_kernel = flashinfer_kernel
            else:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    FlashInferGDNKernel,
                )

                flashinfer_kernel = FlashInferGDNKernel()
                self.extend_kernel = flashinfer_kernel
        elif prefill_backend.is_helion():
            raise ValueError(
                "The Helion linear-attention backend supports KDA only, not GDN."
            )
        else:
            raise ValueError(f"Unsupported GDN prefill backend: {prefill_backend}")

        # Verify kernel. An explicitly configured verify backend wins; the
        # historical auto rule (FlashInfer when the selected FlashInfer kernel
        # supports MTP verify) only applies when no explicit choice was made.
        # SM90 FlashInfer verify requires a fp32 SSM state, so e.g.
        # --mamba-ssm-dtype bfloat16 setups must be able to force Triton here.
        if verify_backend is not None and verify_backend.is_triton():
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False
        elif (
            decode_backend.is_flashinfer() or prefill_backend.is_flashinfer()
        ) and flashinfer_kernel.supports_target_verify:
            self.verify_kernel = flashinfer_kernel
            self.verify_kernel_is_flashinfer = True
        else:
            self.verify_kernel = triton_kernel
            self.verify_kernel_is_flashinfer = False

        self.supports_packed_decode = getattr(
            self.decode_kernel, "supports_packed_decode", False
        )

        rank0_log(
            f"GDN kernel dispatcher: decode={self.decode_kernel.__class__.__name__}, "
            f"extend={self.extend_kernel.__class__.__name__}, "
            f"verify={self.verify_kernel.__class__.__name__} "
            f"packed_decode={self.supports_packed_decode}"
        )

    @property
    def extend_uses_state_checkpoints(self) -> bool:
        return self.extend_kernel.uses_state_checkpoints

    def packed_decode(
        self,
        mixed_qkv: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        scale: float,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        num_v_heads: int,
        head_v_dim: int,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        """Attempt packed decode. Returns output tensor or None if
        the decode kernel does not support packed decode."""
        if not self.supports_packed_decode:
            return None
        return self.decode_kernel.packed_decode(
            mixed_qkv,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            scale=scale,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            num_v_heads=num_v_heads,
            head_v_dim=head_v_dim,
            **kwargs,
        )

    def decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        return self.decode_kernel.decode(
            q,
            k,
            v,
            a,
            b,
            A_log=A_log,
            dt_bias=dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> tuple:
        return self.extend_kernel.extend(
            q,
            k,
            v,
            g,
            beta,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )

    def target_verify(
        self,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        *,
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # FlashInfer verify supports a linear MTP chain. Tree-shaped drafts
        # carry parent indices and must use Triton even when decode/prefill use
        # FlashInfer.
        verify_kernel = (
            self.tree_verify_kernel
            if kwargs.get("retrieve_parent_token") is not None
            else self.verify_kernel
        )
        return verify_kernel.target_verify(
            A_log=A_log,
            dt_bias=dt_bias,
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
            **kwargs,
        )


class GDNAttnBackend(MambaAttnBackendBase):
    """Attention backend for GDN (Gated Delta Network) linear attention."""

    needs_cpu_seq_lens: bool = False

    def __init__(self, model_runner: ModelRunner):
        _validate_gdn_linear_attn_backends(model_runner.linear_attn_backends)
        super().__init__(model_runner)
        self.conv_states_shape = (
            model_runner.req_to_token_pool.mamba_pool.mamba_cache.conv[0].shape
        )
        if not is_cpu() and not is_npu():
            assert self.conv_states_shape[-1] < FLA_CHUNK_SIZE, (
                f"{self.conv_states_shape[-1]=} should be less than {FLA_CHUNK_SIZE}"
            )

        backends = model_runner.linear_attn_backends
        self.linear_attn_backends = backends
        self.kernel_dispatcher = GDNKernelDispatcher(
            backends.decode, backends.prefill, backends.verify
        )
        # CUDA Triton prefill can keep the GDN body inside BCG once its
        # variable request layout is expressed through stable metadata. Other
        # kernels retain the established eager break until separately audited.
        self.use_captured_forward_metadata_for_breakable_cuda_graph = (
            is_cuda()
            and isinstance(self.kernel_dispatcher.extend_kernel, TritonGDNKernel)
            and not get_memory().enable_page_major_kv_layout
            and not self.enable_unified_memory
            and not model_runner.server_args.enable_prefill_context_parallel
            and model_runner.server_args.attn_cp_size == 1
        )
        self.bcg_radix_tracking_enabled = (
            model_runner.server_args.enable_mamba_extra_buffer()
            and not get_memory().disable_radix_cache
        )
        # Fixed-capacity radix-tracking metadata removes the per-GDN-layer host
        # break, but its padded tracking work grows with the captured bucket.
        # Keep the conservative default configurable so deployments can tune
        # the crossover point for their GPU and request-length distribution.
        self.bcg_tracking_capture_max_tokens = (
            get_exec().mamba.gdn_bcg_tracking_capture_max_tokens
        )
        # Sized past the pool for attn_tp-padded warmup/MLP-sync batches (see helper).
        self.verify_intermediate_state_indices = (
            build_verify_intermediate_state_indices(
                self.req_to_token_pool.size,
                model_runner.server_args,
                model_runner.device,
            )
        )

    @staticmethod
    def _has_active_mamba_tracking(forward_batch: ForwardBatch) -> bool:
        mask = forward_batch.mamba_track_mask
        return mask is not None and bool(mask.any())

    def can_capture_attention_body(
        self, layer: RadixLinearAttention, forward_batch: ForwardBatch
    ) -> bool:
        del layer
        metadata = self.forward_metadata
        return bool(
            self.use_captured_forward_metadata_for_breakable_cuda_graph
            and forward_batch.forward_mode.is_extend()
            and not forward_batch.forward_mode.is_target_verify()
            and metadata is not None
            and metadata.gdn_chunk_indices is not None
        )

    def can_replay_captured_attention_body(self, forward_batch: ForwardBatch) -> bool:
        if not self.use_captured_forward_metadata_for_breakable_cuda_graph:
            return True
        if (
            not forward_batch.forward_mode.is_extend()
            or forward_batch.forward_mode.is_target_verify()
        ):
            return False
        return True

    def _populate_breakable_prefill_metadata(
        self, metadata: ForwardMetadata, forward_batch: ForwardBatch
    ) -> None:
        seq_lens = [int(x) for x in forward_batch.extend_seq_lens_cpu]
        bs = len(seq_lens)
        raw_num_tokens = sum(seq_lens)
        request_capacity = metadata.gdn_bcg_request_capacity
        token_capacity = metadata.gdn_bcg_token_capacity
        max_real_requests = request_capacity - 1  # final slot is always dummy
        if bs > max_real_requests:
            raise ValueError(
                f"GDN BCG batch has {bs} requests, capacity is {max_real_requests}"
            )
        if raw_num_tokens > token_capacity:
            raise ValueError(
                f"GDN BCG batch has {raw_num_tokens} tokens, capacity is {token_capacity}"
            )

        query_start_loc_cpu = torch.full(
            (request_capacity + 1,), raw_num_tokens, dtype=torch.int32
        )
        query_start_loc_cpu[0] = 0
        if bs:
            query_start_loc_cpu[1 : bs + 1] = torch.tensor(
                seq_lens, dtype=torch.int32
            ).cumsum(0)
        metadata.query_start_loc.copy_(query_start_loc_cpu)

        metadata.mamba_cache_indices.fill_(self.pad_slot_id)
        if bs:
            mamba_indices = self.req_to_token_pool.get_mamba_indices(
                forward_batch.req_pool_indices[:bs]
            )
            mamba_indices = self._translate_mamba_indices(mamba_indices).to(
                dtype=metadata.mamba_cache_indices.dtype
            )
            metadata.mamba_cache_indices[:bs].copy_(mamba_indices)

        metadata.gdn_has_initial_states.zero_()
        if bs:
            metadata.gdn_has_initial_states[:bs].copy_(
                forward_batch.extend_prefix_lens[:bs] > 0
            )

        chunk_plan = _build_gdn_bcg_chunk_plan(
            seq_lens,
            capacity=metadata.gdn_chunk_indices.shape[0],
            dummy_sequence=request_capacity - 1,
        )
        metadata.gdn_chunk_indices.copy_(chunk_plan)

        # Keep radix tracking capture-stable too. Without this, every request
        # crossing the normal cache checkpoint makes the entire prefill graph
        # fall back to eager execution.
        metadata.gdn_track_conv_steps.fill_(-1)
        metadata.gdn_track_ssm_h_steps.fill_(-1)
        metadata.gdn_track_ssm_final_steps.fill_(-1)
        metadata.track_conv_indices.zero_()
        metadata.conv_states_mask_indices.zero_()
        metadata.track_ssm_h_src.zero_()
        metadata.track_ssm_h_dst.zero_()
        metadata.track_ssm_final_src.zero_()
        metadata.track_ssm_final_dst.zero_()

        track_mask = forward_batch.mamba_track_mask
        track_dst = forward_batch.mamba_track_indices
        track_seqlens = forward_batch.mamba_track_seqlens
        if (
            bs == 0
            or track_mask is None
            or track_dst is None
            or track_seqlens is None
            or not bool(track_mask.any())
        ):
            return

        track_mask_cpu = track_mask[:bs].detach().to("cpu", dtype=torch.bool)
        track_dst_cpu = (
            self._translate_mamba_indices(track_dst[:bs])
            .detach()
            .to("cpu", dtype=torch.int64)
        )
        valid_track = track_mask_cpu & (track_dst_cpu >= 0)
        if not bool(valid_track.any()):
            return

        prefix_cpu = (
            forward_batch.extend_prefix_lens[:bs].detach().to("cpu", dtype=torch.int64)
        )
        track_lens_cpu = track_seqlens[:bs].detach().to("cpu", dtype=torch.int64)
        seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        starts_cpu = torch.zeros((bs,), dtype=torch.int64)
        if bs > 1:
            starts_cpu[1:] = seq_lens_cpu[:-1].cumsum(0)

        cache_chunk = mamba_cache_chunk_size()
        lens_to_track = track_lens_cpu - prefix_cpu
        aligned_lens = (lens_to_track // cache_chunk) * cache_chunk
        conv_len = self.conv_states_shape[-1]
        conv_starts = starts_cpu + aligned_lens - conv_len
        conv_offsets = torch.arange(conv_len, dtype=torch.int64)
        conv_indices = (conv_starts[:, None] + conv_offsets[None, :]).clamp(
            0, max(raw_num_tokens - 1, 0)
        )
        metadata.track_conv_indices[:bs].copy_(conv_indices)
        metadata.conv_states_mask_indices[:bs].copy_(track_dst_cpu)
        metadata.gdn_track_conv_steps[:bs].copy_(
            torch.where(
                valid_track,
                torch.zeros_like(track_dst_cpu, dtype=torch.int32),
                torch.full_like(track_dst_cpu, -1, dtype=torch.int32),
            )
        )

        # Triton GDN returns h in the same sequence-major order as the real
        # prefix of gdn_chunk_indices. The padded plan rows are never selected.
        num_h_states = (seq_lens_cpu + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE
        h_offsets = torch.zeros_like(num_h_states)
        if bs > 1:
            h_offsets[1:] = num_h_states[:-1].cumsum(0)
        aligned = (lens_to_track % cache_chunk) == 0
        h_valid = valid_track & ~aligned
        final_valid = valid_track & aligned
        h_src = h_offsets + lens_to_track // cache_chunk

        metadata.track_ssm_h_src[:bs].copy_(h_src)
        metadata.track_ssm_h_dst[:bs].copy_(track_dst_cpu)
        metadata.track_ssm_final_src[:bs].copy_(
            metadata.mamba_cache_indices[:bs].detach().to("cpu", dtype=torch.int64)
        )
        metadata.track_ssm_final_dst[:bs].copy_(track_dst_cpu)
        metadata.gdn_track_ssm_h_steps[:bs].copy_(
            torch.where(
                h_valid,
                torch.zeros_like(track_dst_cpu, dtype=torch.int32),
                torch.full_like(track_dst_cpu, -1, dtype=torch.int32),
            )
        )
        metadata.gdn_track_ssm_final_steps[:bs].copy_(
            torch.where(
                final_valid,
                torch.zeros_like(track_dst_cpu, dtype=torch.int32),
                torch.full_like(track_dst_cpu, -1, dtype=torch.int32),
            )
        )

    def init_forward_metadata_for_breakable_cuda_graph_capture(
        self, forward_batch: ForwardBatch
    ) -> ForwardMetadata:
        token_capacity = int(forward_batch.input_ids.shape[0])
        if (
            self.bcg_radix_tracking_enabled
            and token_capacity > self.bcg_tracking_capture_max_tokens
        ):
            # Returning ordinary metadata makes can_capture_attention_body
            # retain the existing per-layer eager breaks for this bucket.
            self.init_forward_metadata(forward_batch)
            return self.forward_metadata
        max_real_requests = min(self.req_to_token_pool.size, token_capacity)
        # An extra zero-length request gives padded chunk-plan rows a target
        # that can never alias a real state slot.
        request_capacity = max_real_requests + 1
        chunk_capacity = _gdn_bcg_chunk_plan_capacity(token_capacity, max_real_requests)
        metadata = ForwardMetadata(
            query_start_loc=torch.empty(
                (request_capacity + 1,), dtype=torch.int32, device=self.device
            ),
            mamba_cache_indices=torch.full(
                (request_capacity,),
                self.pad_slot_id,
                dtype=torch.int32,
                device=self.device,
            ),
            gdn_has_initial_states=torch.zeros(
                (request_capacity,), dtype=torch.bool, device=self.device
            ),
            gdn_chunk_indices=torch.empty(
                (chunk_capacity, 2), dtype=torch.int32, device=self.device
            ),
            track_conv_indices=torch.zeros(
                (request_capacity, self.conv_states_shape[-1]),
                dtype=torch.int64,
                device=self.device,
            ),
            conv_states_mask_indices=torch.zeros(
                (request_capacity,), dtype=torch.int64, device=self.device
            ),
            track_ssm_h_src=torch.zeros(
                (request_capacity,), dtype=torch.int64, device=self.device
            ),
            track_ssm_h_dst=torch.zeros(
                (request_capacity,), dtype=torch.int64, device=self.device
            ),
            track_ssm_final_src=torch.zeros(
                (request_capacity,), dtype=torch.int64, device=self.device
            ),
            track_ssm_final_dst=torch.zeros(
                (request_capacity,), dtype=torch.int64, device=self.device
            ),
            gdn_track_conv_steps=torch.full(
                (request_capacity,), -1, dtype=torch.int32, device=self.device
            ),
            gdn_track_ssm_h_steps=torch.full(
                (request_capacity,), -1, dtype=torch.int32, device=self.device
            ),
            gdn_track_ssm_final_steps=torch.full(
                (request_capacity,), -1, dtype=torch.int32, device=self.device
            ),
            # The tracking branch is always present in the captured graph;
            # ordinary rows are masked no-ops.
            has_mamba_track_mask=True,
            gdn_bcg_token_capacity=token_capacity,
            gdn_bcg_request_capacity=request_capacity,
        )
        self._populate_breakable_prefill_metadata(metadata, forward_batch)
        self.forward_metadata = metadata
        return metadata

    def prepare_forward_metadata_for_breakable_cuda_graph_replay(
        self,
        capture_metadata: ForwardMetadata,
        forward_batch: ForwardBatch,
        *,
        static_forward_batch: Optional[ForwardBatch] = None,
    ) -> None:
        del static_forward_batch
        if capture_metadata.gdn_chunk_indices is None:
            self.init_forward_metadata(forward_batch)
            return
        self._populate_breakable_prefill_metadata(capture_metadata, forward_batch)
        self.forward_metadata = capture_metadata

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        super().init_forward_metadata(forward_batch)
        if self.forward_metadata.has_mamba_track_mask:
            self.forward_metadata.mamba_track_mask_indices = (
                forward_batch.mamba_track_mask.nonzero(as_tuple=True)[0]
            )
            self.forward_metadata.conv_states_mask_indices = (
                forward_batch.mamba_track_indices[
                    self.forward_metadata.mamba_track_mask_indices
                ]
            )
            if self.kernel_dispatcher.extend_uses_state_checkpoints:
                from sglang.srt.layers.attention.linear.kernels.gdn_flashinfer import (
                    maybe_build_flashinfer_checkpoint_plan,
                )

                maybe_build_flashinfer_checkpoint_plan(
                    forward_batch, self.forward_metadata, self.device
                )

    def forward_decode(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        if _is_hip and isinstance(mixed_qkv, torch.Tensor) and mixed_qkv.shape[0] == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        layer_cache = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = layer_cache.conv[0]
        ssm_states = layer_cache.temporal
        query_start_loc = self.forward_metadata.query_start_loc
        cache_indices = self.forward_metadata.mamba_cache_indices
        # GDN ReplaySSM (slice 1a): per-layer ring slices + the once-per-forward
        # per-row write cursor. All None unless --enable-linear-replayssm, so the
        # legacy dispatch below is byte-identical when the flag is off.
        replayssm_write_pos = self.forward_metadata.replayssm_write_pos
        # GDN ReplaySSM (slice 2b): per-row force-flush at radix track
        # boundaries (None unless --enable-linear-replayssm). When present the
        # kernel folds the ring into temporal[slot] on the snapshot steps.
        replayssm_force_flush = self.forward_metadata.replayssm_force_flush
        replayssm_d = layer_cache.replayssm_d
        replayssm_k = layer_cache.replayssm_k
        replayssm_g = layer_cache.replayssm_g

        assert isinstance(mixed_qkv, torch.Tensor)
        mixed_qkv = causal_conv1d_update(
            mixed_qkv,
            conv_states,
            layer.conv_weights,
            layer.bias,
            layer.activation,
            conv_state_indices=cache_indices,
        )

        # Skip split + reshape + separate gating kernel by consuming
        # the packed mixed_qkv directly in a single fused Triton kernel.
        if self.kernel_dispatcher.supports_packed_decode:
            core_attn_out = self.kernel_dispatcher.packed_decode(
                mixed_qkv=mixed_qkv,
                a=a,
                b=b,
                A_log=layer.A_log,
                dt_bias=layer.dt_bias,
                scale=layer.head_k_dim**-0.5,
                ssm_states=ssm_states,
                cache_indices=cache_indices,
                num_v_heads=layer.num_v_heads,
                head_v_dim=layer.head_v_dim,
                replayssm_d=replayssm_d,
                replayssm_k=replayssm_k,
                replayssm_g=replayssm_g,
                replayssm_write_pos=replayssm_write_pos,
                replayssm_force_flush=replayssm_force_flush,
            )
            self._track_mamba_state_decode(
                forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
            )
            return core_attn_out

        query, key, value = torch.split(
            mixed_qkv,
            [layer.q_dim, layer.k_dim, layer.v_dim],
            dim=-1,
        )
        # Reshape from [bs, h*d] to [1, bs, h, d]
        bs = forward_batch.batch_size
        query = query.view(1, bs, layer.num_q_heads, layer.head_q_dim)
        key = key.view(1, bs, layer.num_k_heads, layer.head_k_dim)
        value = value.view(1, bs, layer.num_v_heads, layer.head_v_dim)

        core_attn_out = self.kernel_dispatcher.decode(
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            ssm_states=ssm_states,
            cache_indices=cache_indices,
            query_start_loc=query_start_loc,
        )

        self._track_mamba_state_decode(
            forward_batch, conv_states, ssm_states, cache_indices, layer.layer_id
        )

        return core_attn_out

    def forward_extend(
        self,
        layer: RadixLinearAttention,
        forward_batch: ForwardBatch,
        mixed_qkv: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs,
    ):
        assert isinstance(mixed_qkv, torch.Tensor)
        seq_len = mixed_qkv.shape[0]

        if _is_hip and seq_len == 0:
            return mixed_qkv.new_zeros((1, 0, layer.num_v_heads, layer.head_v_dim))

        is_target_verify = forward_batch.forward_mode.is_target_verify()
        forward_metadata = self.forward_metadata

        query_start_loc = forward_metadata.query_start_loc
        cache_indices = forward_metadata.mamba_cache_indices
        retrieve_next_token = forward_metadata.retrieve_next_token
        retrieve_next_sibling = forward_metadata.retrieve_next_sibling
        retrieve_parent_token = forward_metadata.retrieve_parent_token

        mamba_cache_params = self.req_to_token_pool.mamba2_layer_cache(layer.layer_id)
        conv_states = mamba_cache_params.conv[0]
        ssm_states = mamba_cache_params.temporal
        if is_target_verify:
            assert isinstance(mamba_cache_params, MambaPool.SpeculativeState)
            intermediate_state_cache = mamba_cache_params.intermediate_ssm
            intermediate_conv_window_cache = (
                mamba_cache_params.intermediate_conv_window[0]
            )
            intermediate_state_indices = self.verify_intermediate_state_indices
        else:
            has_initial_states = (
                forward_metadata.gdn_has_initial_states
                if forward_metadata.gdn_has_initial_states is not None
                else forward_batch.extend_prefix_lens > 0
            )

        # Page-major envelope: the prefill kernels (CUDA causal_conv1d_fwd,
        # chunk_gated_delta_rule) write state back in place assuming a contiguous
        # slot layout, so they silently drop the write to the strided envelope
        # pool. CUDA causal conv also requires its working state dtype to match
        # the activation. Use request-local copies for either constraint and
        # scatter the result back.
        # CPU kernels (causal_conv1d_fwd_cpu, chunk_gated_delta_rule_cpu) use
        # proper indexed writes and handle non-contiguous pools directly via
        # cache_indices, so the gather/scatter round-trip is unnecessary on CPU.
        # TODO(ch-wan): drop these .contiguous() copies by making the prefill conv
        # and chunk_gated_delta_rule kernels honor the pool's real slot stride +
        # int64 indexing, like packed_decode / causal_conv1d_update already do.
        needs_conv_gather = (
            (not is_target_verify)
            and (not is_cpu())
            and (
                not conv_states.is_contiguous()
                or (is_cuda() and conv_states.dtype != mixed_qkv.dtype)
            )
        )
        needs_ssm_gather = (
            (not is_target_verify) and (not is_cpu()) and not ssm_states.is_contiguous()
        )
        if needs_conv_gather or needs_ssm_gather:
            local_cache_indices = torch.arange(
                cache_indices.shape[0],
                device=cache_indices.device,
                dtype=cache_indices.dtype,
            )
            local_cache_indices.masked_fill_(cache_indices < 0, self.pad_slot_id)

        if needs_conv_gather:
            conv_states_contig = conv_states[cache_indices.clamp_min(0)].contiguous()
            if conv_states_contig.dtype != mixed_qkv.dtype:
                conv_states_contig = conv_states_contig.to(mixed_qkv.dtype)
            conv_cache_indices = local_cache_indices
        else:
            conv_states_contig = conv_states
            conv_cache_indices = cache_indices

        if needs_ssm_gather:
            ssm_states_contig = ssm_states[cache_indices.clamp_min(0)].contiguous()
            ssm_cache_indices = local_cache_indices
        else:
            ssm_states_contig = ssm_states
            ssm_cache_indices = cache_indices

        if is_target_verify:
            batch_size = seq_len // forward_batch.spec_info.draft_token_num
            draft_token_num = forward_batch.spec_info.draft_token_num
            mixed_qkv_reshaped = mixed_qkv.view(
                batch_size, draft_token_num, -1
            ).transpose(1, 2)
            mixed_qkv_processed = causal_conv1d_update(
                mixed_qkv_reshaped,
                conv_states,
                layer.conv_weights,
                layer.bias,
                layer.activation,
                conv_state_indices=cache_indices[:batch_size],
                intermediate_conv_window=intermediate_conv_window_cache,
                intermediate_state_indices=intermediate_state_indices[:batch_size],
                retrieve_next_token=retrieve_next_token,
                retrieve_next_sibling=retrieve_next_sibling,
                retrieve_parent_token=retrieve_parent_token,
            )
            mixed_qkv = mixed_qkv_processed.transpose(1, 2).view(seq_len, -1)
        else:
            mixed_qkv = mixed_qkv.transpose(0, 1)
            if forward_metadata.has_mamba_track_mask:
                if forward_metadata.gdn_track_conv_steps is not None:
                    scatter_gdn_prefill_conv_states_with_mask(
                        dst=conv_states,
                        src=mixed_qkv,
                        src_token_indices=forward_metadata.track_conv_indices,
                        dst_indices=forward_metadata.conv_states_mask_indices,
                        steps=forward_metadata.gdn_track_conv_steps,
                    )
                else:
                    mixed_qkv_to_track = mixed_qkv[
                        :, forward_metadata.track_conv_indices
                    ].transpose(0, 1)
                    conv_states[forward_metadata.conv_states_mask_indices] = (
                        mixed_qkv_to_track.to(conv_states.dtype)
                    )

            conv_kwargs = dict(
                activation=layer.activation,
                conv_states=conv_states_contig,
                has_initial_state=has_initial_states,
                cache_indices=conv_cache_indices,
                query_start_loc=query_start_loc,
            )
            # The list-valued seq_lens fast path builds a host launch grid and
            # cannot be replayed. Omitting it selects the compiled CUDA kernel;
            # its temporary contiguous copy is captured with a stable address.
            if forward_metadata.gdn_chunk_indices is None:
                conv_kwargs["seq_lens_cpu"] = forward_batch.extend_seq_lens_cpu
            mixed_qkv = causal_conv1d_fn(
                mixed_qkv,
                layer.conv_weights,
                layer.bias,
                **conv_kwargs,
            ).transpose(0, 1)[:seq_len]

        actual_seq_len = mixed_qkv.shape[0]
        qkv_dim = layer.q_dim + layer.k_dim + layer.v_dim
        if (is_cuda() or is_hip() or is_xpu()) and qkv_dim <= MAX_FUSED_QKV_SPLIT_DIM:
            query, key, value = fused_qkv_split_gdn_prefill(
                mixed_qkv,
                layer.num_q_heads,
                layer.num_k_heads,
                layer.num_v_heads,
                layer.head_q_dim,
                layer.head_k_dim,
                layer.head_v_dim,
            )
        else:
            query, key, value = torch.split(
                mixed_qkv,
                [layer.q_dim, layer.k_dim, layer.v_dim],
                dim=-1,
            )
            query = query.view(1, actual_seq_len, layer.num_q_heads, layer.head_q_dim)
            key = key.view(1, actual_seq_len, layer.num_k_heads, layer.head_k_dim)
            value = value.view(1, actual_seq_len, layer.num_v_heads, layer.head_v_dim)

        if is_target_verify:
            # ReplaySSM verify protocols: fold-every-commit (ring-write during
            # verify, fold on commit), circular ring, or the snapshotting
            # fallback when neither ring is allocated.
            mamba_pool = self.req_to_token_pool.mamba_pool
            use_replayssm_fold = (
                mamba_cache_params.replayssm_rawv is not None
                and getattr(mamba_pool, "replayssm_spec_fold", False)
                and not getattr(mamba_pool, "replayssm_is_kda", False)
            )
            use_replayssm_spec = (
                mamba_cache_params.replayssm_d is not None
                and getattr(mamba_pool, "replayssm_cache_base", None) is not None
                and not getattr(mamba_pool, "replayssm_is_kda", False)
            )
            if use_replayssm_fold:
                core_attn_out = self._replayssm_fold_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    layer_cache=mamba_cache_params,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    retrieve_parent_token=retrieve_parent_token,
                )
            elif use_replayssm_spec:
                core_attn_out = self._replayssm_target_verify(
                    layer=layer,
                    query=query,
                    key=key,
                    value=value,
                    a=a,
                    b=b,
                    mamba_pool=mamba_pool,
                    layer_cache=mamba_cache_params,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    draft_token_num=forward_batch.spec_info.draft_token_num,
                )
            else:
                # The recurrent fallback needs the per-draft snapshots, which
                # the pool gates OFF under --enable-linear-replayssm-spec (the
                # same flag that makes `use_replayssm_spec` true above), so
                # this branch is unreachable with a None buffer by
                # construction -- keep it loud rather than silently frozen.
                assert intermediate_state_cache is not None, (
                    "recurrent target_verify fallback requires intermediate_ssm, "
                    "which is not allocated under --enable-linear-replayssm-spec"
                )
                core_attn_out = self.kernel_dispatcher.target_verify(
                    A_log=layer.A_log,
                    dt_bias=layer.dt_bias,
                    q=query,
                    k=key,
                    v=value,
                    a=a,
                    b=b,
                    ssm_states=ssm_states,
                    cache_indices=cache_indices,
                    query_start_loc=query_start_loc,
                    intermediate_states_buffer=intermediate_state_cache,
                    intermediate_state_indices=intermediate_state_indices,
                    cache_steps=forward_batch.spec_info.draft_token_num,
                    retrieve_parent_token=retrieve_parent_token,
                )
        else:
            g, beta = fused_gdn_gating(layer.A_log, a, b, layer.dt_bias)
            core_attn_out, last_recurrent_state, h = self.kernel_dispatcher.extend(
                q=query,
                k=key,
                v=value,
                g=g,
                beta=beta,
                ssm_states=ssm_states_contig,
                cache_indices=ssm_cache_indices,
                query_start_loc=query_start_loc,
                state_checkpoint_cu_starts=(
                    forward_metadata.state_checkpoint_cu_starts
                ),
                num_state_checkpoints=forward_metadata.num_state_checkpoints,
                state_checkpoint_every_n_tokens=(
                    forward_metadata.state_checkpoint_every_n_tokens
                ),
                chunk_indices=forward_metadata.gdn_chunk_indices,
            )

            if is_npu() and last_recurrent_state is not None:
                last_recurrent_state = last_recurrent_state.to(
                    ssm_states.dtype, copy=False
                )
                ssm_states[cache_indices] = last_recurrent_state

            if needs_conv_gather:
                if conv_states.is_contiguous() and is_cuda():
                    scatter_steps = torch.where(
                        cache_indices >= 0,
                        torch.zeros_like(cache_indices),
                        torch.full_like(cache_indices, self.pad_slot_id),
                    )
                    fused_mamba_state_scatter_with_mask(
                        dst=conv_states.unsqueeze(0),
                        src=conv_states_contig.unsqueeze(0).unsqueeze(2),
                        dst_indices_raw=cache_indices,
                        step_indices_raw=scatter_steps,
                    )
                else:
                    valid = cache_indices >= 0
                    conv_states[cache_indices[valid]] = conv_states_contig[valid].to(
                        conv_states.dtype
                    )
            if needs_ssm_gather:
                valid = cache_indices >= 0
                ssm_states[cache_indices[valid]] = ssm_states_contig[valid]

            if forward_metadata.has_mamba_track_mask:
                if forward_metadata.gdn_track_ssm_h_steps is not None:
                    assert h is not None
                    scatter_gdn_prefill_states_with_mask(
                        dst=ssm_states,
                        src=h.squeeze(0),
                        src_indices=forward_metadata.track_ssm_h_src,
                        dst_indices=forward_metadata.track_ssm_h_dst,
                        steps=forward_metadata.gdn_track_ssm_h_steps,
                    )
                    scatter_gdn_prefill_states_with_mask(
                        dst=ssm_states,
                        src=ssm_states,
                        src_indices=forward_metadata.track_ssm_final_src,
                        dst_indices=forward_metadata.track_ssm_final_dst,
                        steps=forward_metadata.gdn_track_ssm_final_steps,
                    )
                else:
                    self._track_mamba_state_extend(
                        forward_batch, h, ssm_states, forward_metadata
                    )

        return core_attn_out

    def _replayssm_fold_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        layer_cache: "MambaPool.SpeculativeState",
        ssm_states: torch.Tensor,
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        retrieve_parent_token: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Ring-writing verify; the commit fold replays the accepted prefix
        into ``temporal``. Uses the vendored CuTe DSL MTP kernel when the
        dispatcher selected the FlashInfer bf16-state verify, else the Triton
        recurrent kernel (both store the same raw window)."""
        from sglang.kernels.ops.attention.fla.fused_sigmoid_gating_recurrent import (
            fused_sigmoid_gating_delta_rule_update,
        )

        assert retrieve_parent_token is None, (
            "ReplaySSM fold-every-commit supports a linear draft chain only "
            "(topk <= 1); EAGLE tree verify must use the recurrent verify."
        )
        seq_len = query.shape[1]
        batch_size = query_start_loc.shape[0] - 1
        draft_token_num = seq_len // batch_size
        if (
            self.kernel_dispatcher.verify_kernel_is_flashinfer
            and ssm_states.dtype == torch.bfloat16
            and draft_token_num >= 3
        ):
            from sglang.kernels.ops.attention.cutedsl_gdn_mtp_ring import (
                gated_delta_rule_mtp,
            )

            num_v_heads = value.shape[2]
            head_v_dim = value.shape[3]
            out = gated_delta_rule_mtp(
                A_log=layer.A_log.detach(),
                a=a.view(batch_size, draft_token_num, num_v_heads),
                dt_bias=layer.dt_bias.detach(),
                q=query.view(batch_size, draft_token_num, *query.shape[2:]),
                k=key.view(batch_size, draft_token_num, *key.shape[2:]),
                v=value.view(batch_size, draft_token_num, num_v_heads, head_v_dim),
                b=b.view(batch_size, draft_token_num, num_v_heads),
                initial_state_source=ssm_states,
                initial_state_indices=cache_indices,
                use_qk_l2norm_in_kernel=True,
                disable_state_update=True,
                cache_ring=True,
                replayssm_rawv=layer_cache.replayssm_rawv,
                replayssm_rawk=layer_cache.replayssm_rawk,
                replayssm_g=layer_cache.replayssm_g,
                replayssm_beta=layer_cache.replayssm_beta,
            )
            return out.view(1, seq_len, num_v_heads, head_v_dim)
        return fused_sigmoid_gating_delta_rule_update(
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            q=query,
            k=key,
            v=value,
            a=a,
            b=b,
            initial_state_source=ssm_states,
            initial_state_indices=cache_indices,
            cu_seqlens=query_start_loc,
            use_qk_l2norm_in_kernel=True,
            softplus_beta=1.0,
            softplus_threshold=20.0,
            is_kda=False,
            disable_state_update=True,
            cache_ring=True,
            replayssm_rawv=layer_cache.replayssm_rawv,
            replayssm_rawk=layer_cache.replayssm_rawk,
            replayssm_g=layer_cache.replayssm_g,
            replayssm_beta=layer_cache.replayssm_beta,
        )

    def _replayssm_target_verify(
        self,
        *,
        layer: RadixLinearAttention,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        a: torch.Tensor,
        b: torch.Tensor,
        mamba_pool: MambaPool,
        layer_cache: "MambaPool.SpeculativeState",
        cache_indices: torch.Tensor,
        query_start_loc: torch.Tensor,
        draft_token_num: int,
    ) -> torch.Tensor:
        """ReplaySSM GDN spec-verify (Part B of #28511).

        Reconstructs the verify output for the whole draft window from the frozen
        checkpoint (``temporal``) + the per-slot circular ``(d, k, g)`` ring, and
        appends this window's drafts to the rings (chunked ``d`` for output
        reconstruction; raw ``v`` / pre-norm ``k`` / fp32 ``beta`` for the
        closed-loop exact fold that replays the recurrent update into the fp32
        checkpoint at flush). The rings are PER-LAYER
        (sliced via ``mamba2_layer_cache``), while the cursors (write_pos,
        cache_base, is_flush) are PER-SLOT pool attributes shared by all GDN layers
        of the step; the cursors persist across steps and are advanced once per step
        by the worker (commit_gdn_replayssm_spec) -- here we only read them and
        write this step's ring entries. GDN has K == V, so ``temporal``
        ([slots, HV, K, V]) is consumed directly as the kernel's [slots, HV, V, K]
        checkpoint.
        """
        from sglang.kernels.ops.attention.fla.gdn_replayssm_spec_decode import (
            gdn_replayssm_spec_decode,
        )

        H, K = layer.num_k_heads, layer.head_k_dim
        HV, V = layer.num_v_heads, layer.head_v_dim
        # q/k/v may be [1, seq, *] (fallback split) or [seq, *] (fused split);
        # derive the packed token count from numel so both layouts flatten.
        seq_len = query.numel() // (H * K)
        q = query.reshape(seq_len, H, K)
        k = key.reshape(seq_len, H, K)
        v = value.reshape(seq_len, HV, V)
        a = a.reshape(seq_len, HV)
        b = b.reshape(seq_len, HV)
        d_cache = layer_cache.replayssm_d  # [slots, HV, L, V]
        max_cache_len = d_cache.shape[-2]  # ring length L
        out = q.new_empty(seq_len, HV, V)
        gdn_replayssm_spec_decode(
            q=q,
            k=k,
            v=v,
            a=a,
            b=b,
            A_log=layer.A_log,
            dt_bias=layer.dt_bias,
            checkpoint_state=layer_cache.temporal,
            d_cache=d_cache,
            k_cache=layer_cache.replayssm_k,
            g_cache=layer_cache.replayssm_g,
            # Closed-loop exact-fold rings: raw v / raw pre-norm k / fp32 beta.
            # The flush replays these through the recurrent update (bit-identical
            # to the recurrent baseline) instead of folding `d` open-loop.
            rawv_cache=layer_cache.replayssm_rawv,
            rawk_cache=layer_cache.replayssm_rawk,
            beta_cache=layer_cache.replayssm_beta,
            out=out,
            query_start_loc=query_start_loc,
            ssm_state_indices=cache_indices,
            # Per-slot cursors live on the pool (shared across all GDN layers),
            # NOT in forward_metadata: the verify kernel reads/writes them
            # block-keyed via ssm_state_indices and must NOT advance write_pos
            # (the worker does that after acceptance), so the decode-path
            # forward_metadata.replayssm_write_pos snapshot is not used here.
            write_pos=mamba_pool.replayssm_write_pos,
            cache_base=mamba_pool.replayssm_cache_base,
            is_flush=mamba_pool.replayssm_is_flush,
            max_cache_len=max_cache_len,
            max_spec_len=draft_token_num,
            scale=K**-0.5,
            use_qk_l2norm_in_kernel=True,
            # SGLang marks invalid/padding requests with a negative mamba slot
            # index (valid slots start at 0), so the kernel's "null block"
            # sentinel is -1, not the vLLM default of 0.
            null_block_id=-1,
        )
        # Match the recurrent target_verify output shape (== value.shape).
        return out.reshape(value.shape)
