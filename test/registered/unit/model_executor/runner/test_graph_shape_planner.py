import unittest

from sglang.srt.model_executor.cuda_graph_config import Backend
from sglang.srt.model_executor.runner.shape_key import (
    GraphShape,
    GraphShapePlanner,
    ShapeKey,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestGraphShapePlanner(CustomTestCase):
    def test_sparse_two_dimensional_selection(self):
        planner = GraphShapePlanner(
            [
                GraphShape(1024, 4),
                GraphShape(1024, 16),
                GraphShape(2048, 8),
                GraphShape(2048, 32),
            ],
            max_token_padding_factor=2,
        )

        self.assertEqual(
            planner.select(num_tokens=900, num_requests=7),
            GraphShape(1024, 16),
        )
        self.assertEqual(
            planner.select(num_tokens=1200, num_requests=7),
            GraphShape(2048, 8),
        )
        self.assertIsNone(planner.select(num_tokens=900, num_requests=33))

    def test_exact_token_tier_selection_for_family_specific_planner(self):
        planner = GraphShapePlanner(
            [GraphShape(192, 32), GraphShape(192, 192), GraphShape(384, 64)]
        )

        # This is the compact-ragged/DSpark case: one token tier can safely
        # retain distinct request-slot geometries in the common key contract.
        self.assertEqual(
            planner.select(
                num_tokens=180,
                num_requests=40,
                token_capacity=192,
            ),
            GraphShape(192, 192),
        )
        self.assertIsNone(
            planner.select(
                num_tokens=180,
                num_requests=193,
                token_capacity=192,
            )
        )

    def test_shape_key_keeps_legacy_and_two_dimensional_keys_distinct(self):
        self.assertNotEqual(
            ShapeKey(size=192),
            ShapeKey(size=192, request_capacity=32),
        )
        self.assertNotEqual(
            ShapeKey(size=192, request_capacity=32),
            ShapeKey(size=192, request_capacity=192),
        )
        legacy_positional = ShapeKey(192, 1, "lora", "dense")
        self.assertEqual(legacy_positional.stream_idx, 1)
        self.assertEqual(legacy_positional.variant_label, "lora")
        self.assertEqual(legacy_positional.dsa_variant, "dense")
        self.assertIsNone(legacy_positional.request_capacity)

    def test_server_config_projects_sparse_shapes_to_token_buckets(self):
        args = ServerArgs(
            model_path="dummy",
            cuda_graph_config={
                "prefill": {
                    "backend": Backend.BREAKABLE,
                    "shape_buckets": [
                        [2048, 8],
                        [1024, 16],
                        [1024, 4],
                        [1024, 4],
                    ],
                }
            },
        )

        args._parse_cuda_graph_config()

        self.assertEqual(args.cuda_graph_config.prefill.bs, [1024, 2048])
        self.assertEqual(args.cuda_graph_config.prefill.max_bs, 2048)
        self.assertEqual(
            args.cuda_graph_config.prefill.shape_buckets,
            [[1024, 4], [1024, 16], [2048, 8]],
        )

    def test_server_config_rejects_ambiguous_one_and_two_dimensional_tables(self):
        args = ServerArgs(
            model_path="dummy",
            cuda_graph_config={
                "prefill": {
                    "backend": Backend.BREAKABLE,
                    "bs": [1024],
                    "shape_buckets": [[1024, 4]],
                }
            },
        )

        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            args._parse_cuda_graph_config()


if __name__ == "__main__":
    unittest.main()
