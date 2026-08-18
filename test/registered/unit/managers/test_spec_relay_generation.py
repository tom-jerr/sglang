from types import SimpleNamespace

import pytest
import torch
import sglang.srt.managers.overlap_utils as overlap_utils
from sglang.srt.managers.overlap_utils import FutureMap
from sglang.srt.speculative.eagle_info import EagleDraftInput


def _draft(rows, generations, forwards, modes):
    return EagleDraftInput(
        future_indices=torch.tensor(rows, dtype=torch.int64),
        future_indices_cpu=torch.tensor(rows, dtype=torch.int64),
        future_generations=torch.tensor(generations, dtype=torch.int64),
        future_producer_forwards=torch.tensor(forwards, dtype=torch.int64),
        future_producer_modes=list(modes),
    )


def test_ordinary_eagle_does_not_allocate_ticket_state_or_payload_event(monkeypatch):
    monkeypatch.setattr(overlap_utils, "_is_cuda", False)
    monkeypatch.setattr(overlap_utils, "_DEBUG_ASSERT", False)
    monkeypatch.delenv("SGLANG_SPEC_RELAY_PARITY_DIR", raising=False)
    algo = SimpleNamespace(is_eagle=lambda: True)
    pool = SimpleNamespace(
        req_to_token=torch.empty((8, 16), dtype=torch.int64),
        req_generation=torch.zeros(8, dtype=torch.int64),
    )

    future_map = FutureMap(
        torch.device("cpu"), algo, pool, relay_tickets_enabled=False
    )

    assert not future_map.relay_tickets_enabled
    assert future_map.committed_generations is None
    assert future_map.published_generations is None
    assert not hasattr(future_map, "payload_ready")


def test_generation_ticket_survives_filter_and_mixed_to_pure_transition():
    draft = _draft(
        rows=[7, 9, 11],
        generations=[2, 4, 8],
        forwards=[30, 31, 31],
        modes=["mixed", "mixed", "mixed"],
    )

    # Request 7 finishes; request 9 is filtered too.  The surviving pure batch
    # must retain row 11's exact producer ticket rather than a positional view
    # of one of the removed requests.
    draft.filter_batch(torch.tensor([2]), [2])

    torch.testing.assert_close(draft.future_indices, torch.tensor([11]))
    torch.testing.assert_close(draft.future_indices_cpu, torch.tensor([11]))
    torch.testing.assert_close(draft.future_generations, torch.tensor([8]))
    torch.testing.assert_close(draft.future_producer_forwards, torch.tensor([31]))
    assert draft.future_producer_modes == ["mixed"]


def test_generation_ticket_mixed_to_mixed_merge_preserves_per_row_producers():
    left = _draft([3, 4], [1, 1], [40, 40], ["mixed", "mixed"])
    right = _draft([8], [6], [43], ["mixed"])

    left.merge_batch(right)

    torch.testing.assert_close(left.future_indices, torch.tensor([3, 4, 8]))
    torch.testing.assert_close(left.future_indices_cpu, torch.tensor([3, 4, 8]))
    torch.testing.assert_close(left.future_generations, torch.tensor([1, 1, 6]))
    torch.testing.assert_close(
        left.future_producer_forwards, torch.tensor([40, 40, 43])
    )
    assert left.future_producer_modes == ["mixed", "mixed", "mixed"]


def _future_map_for_validation(current_generations):
    future_map = object.__new__(FutureMap)
    future_map.req_to_token_pool = SimpleNamespace(
        req_generation=torch.tensor(current_generations, dtype=torch.int64)
    )
    size = len(current_generations)
    future_map.committed_generations = torch.full((size,), -1, dtype=torch.int64)
    future_map.committed_producer_forwards = torch.full(
        (size,), -1, dtype=torch.int64
    )
    future_map.published_generations = torch.full((size,), -1, dtype=torch.int64)
    future_map.published_producer_forwards = torch.full(
        (size,), -1, dtype=torch.int64
    )
    future_map.relay_parity_dir = None
    future_map._relay_consume_index = 0
    return future_map


def _batch(rows, request_ids=("r0", "r1")):
    return SimpleNamespace(
        req_pool_indices_cpu=torch.tensor(rows, dtype=torch.int64),
        forward_iter=12,
        forward_mode="consumer",
        reqs=[SimpleNamespace(rid=rid) for rid in request_ids],
    )


def _commit_ticket(future_map, rows, generations, forwards):
    rows = torch.tensor(rows, dtype=torch.int64)
    generations = torch.tensor(generations, dtype=torch.int64)
    forwards = torch.tensor(forwards, dtype=torch.int64)
    future_map.committed_generations[rows] = generations
    future_map.committed_producer_forwards[rows] = forwards
    future_map.published_generations[rows] = generations
    future_map.published_producer_forwards[rows] = forwards


def test_generation_ticket_accepts_atomic_seq_and_payload_commit():
    future_map = _future_map_for_validation([0, 0, 3, 5])
    draft = _draft([2, 3], [3, 5], [10, 10], ["mixed", "mixed"])
    _commit_ticket(future_map, [2, 3], [3, 5], [10, 10])

    future_map._validate_relay_ticket(_batch([2, 3]), draft)


def test_mixed_resolve_updates_shallow_copied_verify_child_before_parent():
    future_map = _future_map_for_validation([0, 0, 3, 5])
    future_map.relay_tickets_enabled = False
    future_map.publish_ready = None
    future_map.needs_cpu_seq_lens = True
    future_map.fwd_prepare_d2h_stream = None
    future_map.new_seq_lens_buf = torch.tensor([0, 0, 104, 205])

    verify_draft = SimpleNamespace(future_indices=torch.tensor([2, 3]))
    stale_verify_lens = torch.tensor([100, 200])
    verify = SimpleNamespace(
        spec_mixed_prefill_batch=None,
        spec_mixed_verify_batch=None,
        spec_info=verify_draft,
        seq_lens=stale_verify_lens,
        seq_lens_cpu=stale_verify_lens.clone(),
        req_pool_indices_cpu=torch.tensor([2, 3]),
    )
    prefill = SimpleNamespace(
        seq_lens=torch.tensor([512]), seq_lens_cpu=torch.tensor([512])
    )
    parent = SimpleNamespace(
        spec_mixed_prefill_batch=prefill,
        spec_mixed_verify_batch=verify,
        # This is the post-merge tensor. Resolving the parent by rebinding it
        # must not leave verify's shallow-copied old tensor behind.
        seq_lens=torch.tensor([512, 100, 200]),
        seq_lens_cpu=torch.tensor([512, 100, 200]),
    )

    future_map.resolve_seq_lens_cpu(parent)

    torch.testing.assert_close(verify.seq_lens, torch.tensor([104, 205]))
    torch.testing.assert_close(verify.seq_lens_cpu, torch.tensor([104, 205]))
    torch.testing.assert_close(parent.seq_lens, torch.tensor([512, 104, 205]))
    torch.testing.assert_close(parent.seq_lens_cpu, torch.tensor([512, 104, 205]))
    assert parent.seq_lens_sum == 821


@pytest.mark.parametrize(
    "mutation, failed_check",
    [
        ("slot_reuse", "current_generation"),
        ("payload_overwrite", "producer_forward"),
        ("seq_payload_split", "seq_producer_forward"),
        ("filter_reorder", "row_order"),
    ],
)
def test_generation_ticket_rejects_first_cross_round_divergence(
    mutation, failed_check
):
    future_map = _future_map_for_validation([0, 0, 3, 5])
    draft = _draft([2, 3], [3, 5], [10, 10], ["mixed", "mixed"])
    _commit_ticket(future_map, [2, 3], [3, 5], [10, 10])
    batch = _batch([2, 3])

    if mutation == "slot_reuse":
        future_map.req_to_token_pool.req_generation[2] += 1
    elif mutation == "payload_overwrite":
        future_map.committed_producer_forwards[2] = 11
    elif mutation == "seq_payload_split":
        future_map.published_producer_forwards[2] = 9
    elif mutation == "filter_reorder":
        batch.req_pool_indices_cpu = torch.tensor([3, 2])

    with pytest.raises(RuntimeError, match=failed_check):
        future_map._validate_relay_ticket(batch, draft)
