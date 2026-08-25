import unittest
from types import SimpleNamespace

import torch
from sglang.srt.layers.attention.linear.gdn_backend import (
    GDNAttnBackend,
    _build_gdn_bcg_chunk_plan,
    _gdn_bcg_chunk_plan_capacity,
)
from sglang.srt.layers.attention.mamba.mamba2_metadata import ForwardMetadata
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _ReqPool:
    size = 8

    def get_mamba_indices(self, req_pool_indices):
        return req_pool_indices.to(torch.int32) + 10

    def translate_mamba_indices(self, indices):
        return indices


class TestGDNBreakableCapture(CustomTestCase):
    def test_chunk_plan_has_fixed_capacity_and_safe_dummy_rows(self):
        """The derived capacity must cover every partition of a token bucket."""
        capacity = _gdn_bcg_chunk_plan_capacity(160, 8)
        plan = _build_gdn_bcg_chunk_plan(
            [65, 63, 16], capacity=capacity, dummy_sequence=8
        )

        self.assertEqual(plan.shape, (capacity, 2))
        self.assertEqual(plan[:4].tolist(), [[0, 0], [0, 1], [1, 0], [2, 0]])
        self.assertTrue(torch.all(plan[4:] == torch.tensor([8, 0], dtype=torch.int32)))

    def test_chunk_plan_rejects_capacity_overflow(self):
        """An undersized capture plan must fail before a kernel reads past it."""
        with self.assertRaisesRegex(ValueError, "needs 3 rows"):
            _build_gdn_bcg_chunk_plan([65, 1], capacity=2, dummy_sequence=2)

    def test_metadata_refresh_preserves_captured_tensor_addresses(self):
        """Replay preparation must update values without replacing graph inputs."""
        backend = object.__new__(GDNAttnBackend)
        backend.req_to_token_pool = _ReqPool()
        backend.pad_slot_id = -1
        metadata = ForwardMetadata(
            query_start_loc=torch.empty(6, dtype=torch.int32),
            mamba_cache_indices=torch.empty(5, dtype=torch.int32),
            gdn_has_initial_states=torch.empty(5, dtype=torch.bool),
            gdn_chunk_indices=torch.empty(5, 2, dtype=torch.int32),
            track_conv_indices=torch.empty(5, 3, dtype=torch.int64),
            conv_states_mask_indices=torch.empty(5, dtype=torch.int64),
            track_ssm_h_src=torch.empty(5, dtype=torch.int64),
            track_ssm_h_dst=torch.empty(5, dtype=torch.int64),
            track_ssm_final_src=torch.empty(5, dtype=torch.int64),
            track_ssm_final_dst=torch.empty(5, dtype=torch.int64),
            gdn_track_conv_steps=torch.empty(5, dtype=torch.int32),
            gdn_track_ssm_h_steps=torch.empty(5, dtype=torch.int32),
            gdn_track_ssm_final_steps=torch.empty(5, dtype=torch.int32),
            gdn_bcg_token_capacity=96,
            gdn_bcg_request_capacity=5,
        )
        pointers = (
            metadata.query_start_loc.data_ptr(),
            metadata.mamba_cache_indices.data_ptr(),
            metadata.gdn_has_initial_states.data_ptr(),
            metadata.gdn_chunk_indices.data_ptr(),
        )
        batch = SimpleNamespace(
            extend_seq_lens_cpu=[32, 16],
            req_pool_indices=torch.tensor([3, 5]),
            extend_prefix_lens=torch.tensor([0, 7]),
            mamba_track_mask=None,
            mamba_track_indices=None,
            mamba_track_seqlens=None,
        )

        backend._populate_breakable_prefill_metadata(metadata, batch)

        self.assertEqual(metadata.query_start_loc.tolist(), [0, 32, 48, 48, 48, 48])
        self.assertEqual(metadata.mamba_cache_indices.tolist(), [13, 15, -1, -1, -1])
        self.assertEqual(
            metadata.gdn_has_initial_states.tolist(),
            [False, True, False, False, False],
        )
        self.assertEqual(
            metadata.gdn_chunk_indices.tolist(),
            [[0, 0], [1, 0], [4, 0], [4, 0], [4, 0]],
        )
        self.assertEqual(
            pointers,
            (
                metadata.query_start_loc.data_ptr(),
                metadata.mamba_cache_indices.data_ptr(),
                metadata.gdn_has_initial_states.data_ptr(),
                metadata.gdn_chunk_indices.data_ptr(),
            ),
        )

    def test_capture_admission_accepts_stable_tracking_metadata(self):
        """Fixed-capacity tracking inputs do not change capture topology."""
        backend = object.__new__(GDNAttnBackend)
        backend.use_captured_forward_metadata_for_breakable_cuda_graph = True
        backend.forward_metadata = ForwardMetadata(
            query_start_loc=torch.empty(2, dtype=torch.int32),
            mamba_cache_indices=torch.empty(1, dtype=torch.int32),
            gdn_chunk_indices=torch.empty(1, 2, dtype=torch.int32),
        )
        batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)

        self.assertTrue(backend.can_capture_attention_body(None, batch))
        backend.forward_metadata.has_mamba_track_mask = True
        self.assertTrue(backend.can_capture_attention_body(None, batch))

    def test_replay_admission_accepts_active_mamba_tracking(self):
        """A live tracked batch refreshes stable masks instead of going eager."""
        backend = object.__new__(GDNAttnBackend)
        backend.use_captured_forward_metadata_for_breakable_cuda_graph = True
        batch = SimpleNamespace(
            forward_mode=ForwardMode.EXTEND,
            mamba_track_mask=torch.tensor([False, True]),
        )

        self.assertTrue(backend.can_replay_captured_attention_body(batch))

    def test_tracking_capture_threshold_controls_long_bucket_admission(self):
        """Long buckets fall back unless the operator raises the tuned limit."""
        backend = object.__new__(GDNAttnBackend)
        backend.bcg_radix_tracking_enabled = True
        backend.bcg_tracking_capture_max_tokens = 512
        backend.req_to_token_pool = _ReqPool()
        backend.device = torch.device("cpu")
        backend.pad_slot_id = -1
        backend.conv_states_shape = (1, 1, 3)
        backend.init_forward_metadata = lambda batch: setattr(
            backend,
            "forward_metadata",
            ForwardMetadata(
                query_start_loc=torch.empty(0, dtype=torch.int32),
                mamba_cache_indices=torch.empty(0, dtype=torch.int32),
            ),
        )
        batch = SimpleNamespace(input_ids=torch.empty(1024, dtype=torch.int32))

        metadata = backend.init_forward_metadata_for_breakable_cuda_graph_capture(batch)
        self.assertIsNone(metadata.gdn_chunk_indices)

        backend.bcg_tracking_capture_max_tokens = 1024
        backend._populate_breakable_prefill_metadata = lambda metadata, batch: None
        metadata = backend.init_forward_metadata_for_breakable_cuda_graph_capture(batch)
        self.assertIsNotNone(metadata.gdn_chunk_indices)


if __name__ == "__main__":
    unittest.main()
