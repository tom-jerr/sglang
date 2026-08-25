import unittest

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.runner_backend.breakable_cuda_graph_backend import (
    BreakableCudaGraphBackend,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _backend_without_init():
    return object.__new__(BreakableCudaGraphBackend)


class TestBreakableCudaGraphLogitsOutput(CustomTestCase):
    def test_field_specific_capacities_and_slices(self):
        """Regression for #36048: logits and EAGLE hidden rows may differ."""
        backend = _backend_without_init()
        largest = LogitsProcessorOutput(
            next_token_logits=torch.zeros(8, 16),
            hidden_states=torch.zeros(24, 4),
        )
        buffer = backend._alloc_full_buffer(largest, size=8)
        current = LogitsProcessorOutput(
            next_token_logits=torch.arange(48, dtype=torch.float32).view(3, 16),
            hidden_states=torch.arange(28, dtype=torch.float32).view(7, 4),
        )

        backend._copy_output_to_buffer(current, buffer, num_tokens=3)
        stored = backend._slice_output_like(buffer, current, fallback_rows=3)

        self.assertEqual(stored.next_token_logits.shape, (3, 16))
        self.assertEqual(stored.hidden_states.shape, (7, 4))
        torch.testing.assert_close(stored.next_token_logits, current.next_token_logits)
        torch.testing.assert_close(stored.hidden_states, current.hidden_states)

    def test_dynamic_python_fields_are_rejected(self):
        """Captured output must not silently retain request-specific Python data."""
        backend = _backend_without_init()
        output = LogitsProcessorOutput(
            next_token_logits=torch.zeros(2, 4),
            customized_info={"request": [1, 2]},
        )

        with self.assertRaisesRegex(TypeError, "customized_info"):
            backend._alloc_full_buffer(output, size=2)


if __name__ == "__main__":
    unittest.main()
