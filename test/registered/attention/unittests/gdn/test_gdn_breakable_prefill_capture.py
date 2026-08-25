import unittest

import torch
from sglang.kernels.ops.attention.fla.chunk import chunk_gated_delta_rule
from sglang.srt.layers.attention.linear.gdn_backend import (
    _build_gdn_bcg_chunk_offsets,
    _build_gdn_bcg_chunk_plan,
    _gdn_bcg_chunk_plan_capacity,
)
from sglang.srt.layers.attention.mamba.causal_conv1d import causal_conv1d_fn
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.attention_unittest.attention_methods.gdn_attention import (
    GDNAttentionCase,
    build_gdn_attention_fixture,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@unittest.skipUnless(
    torch.cuda.is_available() and not is_hip(),
    "CUDA graph capture requires NVIDIA CUDA",
)
class TestGDNBreakablePrefillCapture(CustomTestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.device = torch.device("cuda")
        self.seq_lens = [65, 31]
        self.num_tokens = sum(self.seq_lens)
        self.request_capacity = 4
        self.query_start_loc = torch.tensor(
            [0, 65, 96, 96, 96], dtype=torch.int32, device=self.device
        )
        self.cache_indices = torch.tensor(
            [1, 4, -1, -1], dtype=torch.int32, device=self.device
        )

    @staticmethod
    def _warmup(fn):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fn()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

    def test_fixed_chunk_plan_captures_and_replays(self):
        """A padded fixed-capacity FLA plan must survive graph replay.

        Regression for #35851: rebuilding the plan from ``cu_seqlens`` calls
        ``tolist()`` on every GDN layer and therefore forces an eager break.
        """
        dtype = torch.bfloat16
        num_heads, key_dim, value_dim = 2, 64, 64
        q = torch.randn(
            1,
            self.num_tokens,
            num_heads,
            key_dim,
            dtype=dtype,
            device=self.device,
        )
        k = torch.randn_like(q)
        v = torch.randn(
            1,
            self.num_tokens,
            num_heads,
            value_dim,
            dtype=dtype,
            device=self.device,
        )
        g = torch.nn.functional.logsigmoid(
            torch.randn(
                1,
                self.num_tokens,
                num_heads,
                dtype=dtype,
                device=self.device,
            )
        )
        beta = torch.sigmoid(torch.randn_like(g))
        state_init = torch.randn(
            8,
            num_heads,
            value_dim,
            key_dim,
            dtype=torch.float32,
            device=self.device,
        )
        chunk_capacity = _gdn_bcg_chunk_plan_capacity(
            self.num_tokens, self.request_capacity - 1
        )
        chunk_plan = _build_gdn_bcg_chunk_plan(
            [self.num_tokens],
            capacity=chunk_capacity,
            dummy_sequence=self.request_capacity - 1,
        ).to(self.device)
        chunk_offsets = _build_gdn_bcg_chunk_offsets(
            [self.num_tokens], request_capacity=self.request_capacity
        ).to(self.device)
        self.query_start_loc.copy_(
            torch.tensor([0, 96, 96, 96, 96], device=self.device)
        )

        def run(state):
            return chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=g,
                beta=beta,
                initial_state=state,
                initial_state_indices=self.cache_indices,
                cu_seqlens=self.query_start_loc,
                use_qk_l2norm_in_kernel=True,
                chunk_indices=chunk_plan,
                chunk_offsets=chunk_offsets,
            )[0]

        graph_state = state_init.clone()
        self._warmup(lambda: run(graph_state))
        graph_state.copy_(state_init)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run(graph_state)

        # Replay a different two-request layout through the same captured
        # tensor addresses. Both index and offset metadata must be refreshed.
        self.query_start_loc.copy_(
            torch.tensor([0, 65, 96, 96, 96], device=self.device)
        )
        chunk_plan.copy_(
            _build_gdn_bcg_chunk_plan(
                self.seq_lens,
                capacity=chunk_capacity,
                dummy_sequence=self.request_capacity - 1,
            ).to(self.device)
        )
        chunk_offsets.copy_(
            _build_gdn_bcg_chunk_offsets(
                self.seq_lens, request_capacity=self.request_capacity
            ).to(self.device)
        )
        eager_state = state_init.clone()
        eager_output = run(eager_state)
        graph_state.copy_(state_init)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        torch.testing.assert_close(graph_state, eager_state, rtol=0, atol=0)

    def test_compiled_causal_conv_captures_padded_request_axis(self):
        """The padded request axis must be accepted by captured causal conv.

        Regression for #35851: the Triton prefill path consumes a Python launch
        grid, while the compiled CUDA path can replay stable tensor metadata.
        """
        dtype = torch.float16
        channels, width = 128, 4
        # GDN produces this non-contiguous (channels, tokens) transpose. The
        # compiled CUDA wrapper makes its stable contiguous copy in the graph.
        x = torch.randn(
            self.num_tokens, channels, dtype=dtype, device=self.device
        ).transpose(0, 1)
        weight = torch.randn(channels, width, dtype=dtype, device=self.device)
        bias = torch.randn(channels, dtype=dtype, device=self.device)
        state_init = torch.randn(
            8, channels, width - 1, dtype=dtype, device=self.device
        )
        has_initial_state = torch.tensor(
            [False, True, False, False], dtype=torch.bool, device=self.device
        )

        def run(state):
            return causal_conv1d_fn(
                x,
                weight,
                bias,
                query_start_loc=self.query_start_loc,
                cache_indices=self.cache_indices,
                has_initial_state=has_initial_state,
                conv_states=state,
                activation="silu",
            )

        eager_state = state_init.clone()
        eager_output = run(eager_state)

        graph_state = state_init.clone()
        self._warmup(lambda: run(graph_state))
        graph_state.copy_(state_init)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run(graph_state)

        graph_state.copy_(state_init)
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        torch.testing.assert_close(graph_state, eager_state, rtol=0, atol=0)

    def test_fp16_gdn_captures_with_bf16_conv_cache(self):
        """Regression for #36048: AWQ FP16 must not require an FP16 cache env."""
        case = GDNAttentionCase(
            name="gdn_bcg_fp16_activation_bf16_conv_cache",
            backend="triton",
            forward_mode=ForwardMode.EXTEND,
            num_k_heads=2,
            num_v_heads=2,
            page_size=16,
            prefix_lens=(0, 0),
            extend_lens=(65, 31),
        )
        fixture = build_gdn_attention_fixture(
            self,
            case,
            dtype=torch.float16,
            max_context_len=128,
            runner_batch_size=4,
        )
        linear_backend = fixture.backend.linear_attn_backend
        conv_cache = fixture.runner.req_to_token_pool.mamba_pool.mamba_cache.conv
        conv_cache[0] = conv_cache[0].to(torch.bfloat16)
        self.assertTrue(
            linear_backend.use_captured_forward_metadata_for_breakable_cuda_graph
        )
        linear_backend.init_forward_metadata_for_breakable_cuda_graph_capture(
            fixture.forward_batch
        )

        def run():
            with (
                torch.no_grad(),
                forward_context(ForwardContext(attn_backend=fixture.backend)),
            ):
                return fixture.actual_module(
                    fixture.forward_batch,
                    fixture.mixed_qkv,
                    fixture.a,
                    fixture.b,
                )

        initial_conv = conv_cache[0].clone()
        initial_ssm = (
            fixture.runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.clone()
        )
        eager_output = run()
        eager_conv = conv_cache[0].clone()
        eager_ssm = (
            fixture.runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.clone()
        )

        conv_cache[0].copy_(initial_conv)
        fixture.runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.copy_(
            initial_ssm
        )
        self._warmup(run)
        conv_cache[0].copy_(initial_conv)
        fixture.runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.copy_(
            initial_ssm
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output = run()

        conv_cache[0].copy_(initial_conv)
        fixture.runner.req_to_token_pool.mamba_pool.mamba_cache.temporal.copy_(
            initial_ssm
        )
        graph.replay()
        torch.cuda.synchronize()

        torch.testing.assert_close(graph_output, eager_output, rtol=0, atol=0)
        torch.testing.assert_close(conv_cache[0], eager_conv, rtol=0, atol=0)
        torch.testing.assert_close(
            fixture.runner.req_to_token_pool.mamba_pool.mamba_cache.temporal,
            eager_ssm,
            rtol=0,
            atol=0,
        )

        fixture.forward_batch.mamba_track_mask = torch.ones(
            2, dtype=torch.bool, device=self.device
        )
        fixture.forward_batch.mamba_track_indices = torch.tensor(
            [3, 4], dtype=torch.int64, device=self.device
        )
        fixture.forward_batch.mamba_track_seqlens = torch.tensor(
            [64, 31], dtype=torch.int64, device=self.device
        )
        fixture.backend.init_forward_metadata(fixture.forward_batch)
        tracked_output = run()
        torch.cuda.synchronize()
        self.assertTrue(torch.isfinite(tracked_output).all())
        self.assertEqual(conv_cache[0].dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main()
