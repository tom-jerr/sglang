import unittest

from sglang.srt.managers.data_parallel_controller import (
    BufferedGenerateRequest,
    WorkerLoadSnapshot,
    compute_staggered_assignments,
)


class TestStaggeredStageAwareAssignments(unittest.TestCase):
    def test_long_prefills_are_spread_across_workers(self):
        buffered_reqs = [
            BufferedGenerateRequest(object(), 0.0, 400, 500),
            BufferedGenerateRequest(object(), 1.0, 350, 450),
            BufferedGenerateRequest(object(), 2.0, 40, 50),
        ]
        worker_loads = [WorkerLoadSnapshot(dp_rank=0), WorkerLoadSnapshot(dp_rank=1)]

        assignments = compute_staggered_assignments(
            buffered_reqs=buffered_reqs,
            worker_loads=worker_loads,
            active_status=[True, True],
            kv_outlier_k=1.5,
        )

        first_two_ranks = [rank for _, rank in assignments[:2]]
        self.assertEqual(set(first_two_ranks), {0, 1})

    def test_iqr_guard_skips_decode_outlier(self):
        buffered_reqs = [BufferedGenerateRequest(object(), 0.0, 128, 64)]
        worker_loads = [
            WorkerLoadSnapshot(dp_rank=0, num_tokens=100),
            WorkerLoadSnapshot(dp_rank=1, num_tokens=100),
            WorkerLoadSnapshot(dp_rank=2, num_tokens=100),
            WorkerLoadSnapshot(dp_rank=3, num_tokens=1000),
        ]

        assignments = compute_staggered_assignments(
            buffered_reqs=buffered_reqs,
            worker_loads=worker_loads,
            active_status=[True, True, True, True],
            kv_outlier_k=1.5,
        )

        self.assertEqual(len(assignments), 1)
        self.assertNotEqual(assignments[0][1], 3)

    def test_fallback_when_all_workers_exceed_iqr_threshold(self):
        buffered_reqs = [BufferedGenerateRequest(object(), 0.0, 128, 256)]
        worker_loads = [
            WorkerLoadSnapshot(dp_rank=0, num_tokens=1000),
            WorkerLoadSnapshot(dp_rank=1, num_tokens=1000),
        ]

        assignments = compute_staggered_assignments(
            buffered_reqs=buffered_reqs,
            worker_loads=worker_loads,
            active_status=[True, True],
            kv_outlier_k=1.5,
        )

        self.assertEqual(len(assignments), 1)
        self.assertIn(assignments[0][1], {0, 1})


if __name__ == "__main__":
    unittest.main()
