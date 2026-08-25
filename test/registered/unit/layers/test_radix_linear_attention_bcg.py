import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
from sglang.srt.layers import radix_linear_attention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestRadixLinearAttentionBCG(CustomTestCase):
    def test_backend_capability_controls_linear_attention_graph_break(self):
        """Regression for #35851: capturable GDN must use the in-graph op."""
        layer = RadixLinearAttention(
            layer_id=7,
            num_q_heads=1,
            num_k_heads=1,
            num_v_heads=1,
            head_q_dim=2,
            head_k_dim=2,
            head_v_dim=2,
        )
        forward_batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
        mixed_qkv = torch.zeros(4, 6)
        a = torch.zeros(4, 1)
        b = torch.zeros(4, 1)

        for can_capture in (True, False):
            with self.subTest(can_capture=can_capture):
                backend = SimpleNamespace(
                    can_capture_attention_body=Mock(return_value=can_capture)
                )
                with (
                    patch.object(
                        radix_linear_attention,
                        "get_tc_piecewise_forward_context",
                        return_value=object(),
                    ),
                    patch.object(
                        radix_linear_attention,
                        "get_attn_backend",
                        return_value=backend,
                    ),
                    patch.object(
                        radix_linear_attention,
                        "is_in_breakable_cuda_graph",
                        return_value=True,
                    ),
                    patch.object(
                        radix_linear_attention,
                        "unified_linear_attention_with_output",
                    ) as captured_op,
                    patch.object(
                        radix_linear_attention,
                        "bcg_unified_linear_attention_with_output",
                    ) as eager_op,
                ):
                    layer(forward_batch, mixed_qkv, a, b)

                backend.can_capture_attention_body.assert_called_once_with(
                    layer, forward_batch
                )
                if can_capture:
                    captured_op.assert_called_once()
                    eager_op.assert_not_called()
                else:
                    captured_op.assert_not_called()
                    eager_op.assert_called_once()


if __name__ == "__main__":
    unittest.main()
