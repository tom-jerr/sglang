from types import SimpleNamespace

import pytest
import torch
from sglang.srt.layers.attention.flashattention_backend import (
    CompositePrefillVerifyFlashAttentionMetadata,
    FlashAttentionBackend,
)
from sglang.srt.layers.attention.triton_backend import (
    CompositeForwardMetadataScratch,
    CompositePrefillVerifyMetadata,
    TritonAttnBackend,
)
from sglang.srt.layers.logits_processor import (
    LogitsMetadata,
    LogitsProcessor,
    LogitsProcessorOutput,
    split_composition_logits_output,
)
from sglang.srt.model_executor.forward_batch_info import (
    ForwardBatch,
    ForwardComposition,
    ForwardCompositionTensorScratch,
    ForwardMode,
    pack_prefill_and_verify_forward,
)
from sglang.srt.speculative.parity import (
    AttentionTrace,
    KVRows,
    OperatorTrace,
    attention_parity,
    install_operator_trace_hooks,
    logits_parity,
    operator_parity,
    record_operators,
    remove_operator_trace_hooks,
)


def _batch(mode: ForwardMode, num_tokens: int, batch_size: int) -> ForwardBatch:
    return ForwardBatch(
        forward_mode=mode,
        batch_size=batch_size,
        input_ids=torch.arange(num_tokens, dtype=torch.int64),
        req_pool_indices=torch.arange(batch_size, dtype=torch.int64),
        seq_lens=torch.full((batch_size,), num_tokens, dtype=torch.int64),
        seq_lens_cpu=torch.full((batch_size,), num_tokens, dtype=torch.int64),
        out_cache_loc=torch.arange(num_tokens, dtype=torch.int64),
        seq_lens_sum=num_tokens,
    )


def _composition() -> ForwardComposition:
    return ForwardComposition(
        kind="prefill_spec_verify",
        prefill_batch=_batch(ForwardMode.EXTEND, num_tokens=3, batch_size=1),
        verify_batch=_batch(ForwardMode.TARGET_VERIFY, num_tokens=4, batch_size=2),
        prefill_num_tokens=3,
        verify_num_tokens=4,
    )


def test_forward_composition_accepts_valid_segments():
    _composition().validate(parent_num_tokens=7)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: setattr(value, "kind", "unknown"), "Unsupported"),
        (
            lambda value: setattr(value, "verify_num_tokens", 5),
            "do not match",
        ),
        (
            lambda value: setattr(
                value.prefill_batch, "forward_mode", ForwardMode.DECODE
            ),
            "EXTEND",
        ),
        (
            lambda value: setattr(
                value.verify_batch, "forward_mode", ForwardMode.EXTEND
            ),
            "TARGET_VERIFY",
        ),
        (
            lambda value: setattr(value, "prefill_num_tokens", 0),
            "non-empty",
        ),
    ],
)
def test_forward_composition_rejects_invalid_segments(mutate, message):
    composition = _composition()
    mutate(composition)

    with pytest.raises(ValueError, match=message):
        composition.validate(parent_num_tokens=7)


def test_forward_composition_rejects_segment_tensor_length_mismatch():
    composition = _composition()
    composition.verify_batch.input_ids = torch.arange(3)

    with pytest.raises(ValueError, match="Verify segment input_ids"):
        composition.validate(parent_num_tokens=7)


def test_composition_logits_gather_keeps_prefill_tails_and_all_verify_rows():
    prefill = _batch(ForwardMode.EXTEND, num_tokens=3, batch_size=2)
    prefill.extend_seq_lens = torch.tensor([2, 1], dtype=torch.int32)
    composition = ForwardComposition(
        kind="prefill_spec_verify",
        prefill_batch=prefill,
        verify_batch=_batch(ForwardMode.TARGET_VERIFY, num_tokens=4, batch_size=2),
        prefill_num_tokens=3,
        verify_num_tokens=4,
    )

    gather_indices = composition.build_logits_gather_indices()
    assert gather_indices.tolist() == [1, 2, 3, 4, 5, 6]

    hidden_states = torch.arange(14, dtype=torch.float32).view(7, 2)
    metadata = LogitsMetadata(
        forward_mode=ForwardMode.MIXED,
        composition_gather_indices=gather_indices,
    )
    pruned, before_norm, aux, sample_indices, input_indices, token_to_seq = (
        LogitsProcessor._get_pruned_states(
            None,
            hidden_states,
            hidden_states_before_norm=None,
            aux_hidden_states=None,
            logits_metadata=metadata,
        )
    )

    assert torch.equal(pruned, hidden_states[gather_indices])
    assert before_norm is None
    assert aux is None
    assert sample_indices is None
    assert input_indices is None
    assert token_to_seq == []


def test_pack_and_split_composition_use_stable_segment_views():
    prefill = _batch(ForwardMode.EXTEND, num_tokens=3, batch_size=2)
    prefill.extend_seq_lens = torch.tensor([2, 1], dtype=torch.int32)
    prefill.extend_seq_lens_cpu = [2, 1]
    prefill.extend_prefix_lens = torch.tensor([8, 19], dtype=torch.int64)
    prefill.extend_prefix_lens_cpu = [8, 19]
    prefill.extend_start_loc = torch.tensor([0, 2], dtype=torch.int32)
    prefill.orig_seq_lens = torch.tensor([10, 20], dtype=torch.int64)
    prefill.positions = torch.tensor([10, 11, 20], dtype=torch.int64)
    prefill.num_token_non_padded = torch.tensor(3, dtype=torch.int32)
    prefill.num_token_non_padded_cpu = 3
    prefill.global_num_tokens_cpu = [3]
    prefill.global_num_tokens_gpu = torch.tensor([3], dtype=torch.int32)
    prefill.global_num_tokens_for_logprob_gpu = torch.tensor([3], dtype=torch.int32)
    prefill.global_num_tokens_for_logprob_cpu = [3]
    prefill.global_dp_buffer_len = 3
    verify = _batch(ForwardMode.TARGET_VERIFY, num_tokens=4, batch_size=2)
    verify.seq_lens = torch.tensor([32, 42], dtype=torch.int64)
    verify.seq_lens_cpu = torch.tensor([32, 42], dtype=torch.int64)
    verify.orig_seq_lens = torch.tensor([30, 40], dtype=torch.int64)
    verify.positions = torch.tensor([30, 31, 40, 41], dtype=torch.int64)
    prefill_ids = prefill.input_ids
    verify_ids = verify.input_ids

    packed = pack_prefill_and_verify_forward(prefill, verify)

    assert packed.forward_mode == ForwardMode.MIXED
    assert packed.composition.prefill_batch is prefill
    assert packed.composition.verify_batch is verify
    assert prefill.input_ids is prefill_ids
    assert verify.input_ids is verify_ids
    assert packed.input_ids.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert packed.seq_lens_cpu.shape == (4,)
    assert packed.positions.tolist() == [10, 11, 20, 30, 31, 40, 41]
    assert packed.out_cache_loc.tolist() == [0, 1, 2, 0, 1, 2, 3]
    assert packed.extend_seq_lens.tolist() == [2, 1, 2, 2]
    assert packed.extend_seq_lens_cpu == [2, 1, 2, 2]
    assert packed.extend_prefix_lens.tolist() == [8, 19, 30, 40]
    assert packed.extend_prefix_lens_cpu == [8, 19, 30, 40]
    assert packed.extend_start_loc.tolist() == [0, 2, 3, 5]
    assert packed.orig_seq_lens.tolist() == [10, 20, 30, 40]
    assert packed.extend_seq_lens.data_ptr() != prefill.extend_seq_lens.data_ptr()
    assert packed.num_token_non_padded_cpu == 7
    assert packed.num_token_non_padded.item() == 7
    assert (
        packed.num_token_non_padded.data_ptr()
        != prefill.num_token_non_padded.data_ptr()
    )
    assert packed.global_num_tokens_cpu == [7]
    assert packed.global_num_tokens_gpu.tolist() == [7]
    assert packed.global_num_tokens_for_logprob_cpu == [7]
    assert packed.global_dp_buffer_len == 7
    assert packed.can_run_dp_breakable_cuda_graph

    logits = torch.arange(36, dtype=torch.float32).view(6, 6)
    hidden = torch.arange(14, dtype=torch.float32).view(7, 2)
    packed_output = LogitsProcessorOutput(
        next_token_logits=logits,
        hidden_states=hidden,
    )
    prefill_output, verify_output = split_composition_logits_output(
        packed_output,
        prefill_requests=2,
        prefill_tokens=3,
        verify_tokens=4,
    )

    assert prefill_output.next_token_logits.data_ptr() == logits[:2].data_ptr()
    assert verify_output.next_token_logits.data_ptr() == logits[2:].data_ptr()
    assert prefill_output.hidden_states.data_ptr() == hidden[:3].data_ptr()
    assert verify_output.hidden_states.data_ptr() == hidden[3:].data_ptr()


def test_composition_metadata_uses_disjoint_views_of_shared_arenas():
    scratch = CompositeForwardMetadataScratch(
        int32_indptr=torch.zeros((2, 5), dtype=torch.int32),
        int64_indptr=torch.zeros((4, 5), dtype=torch.int64),
    )
    prefill, verify = scratch.segments(3, 4)

    prefill.kv_indptr.fill_(11)
    prefill.qo_indptr.fill_(12)
    prefill.mask_indptr.fill_(13)

    assert verify.kv_indptr.tolist() == [0] * 5
    assert verify.qo_indptr.tolist() == [0] * 5
    assert verify.mask_indptr.tolist() == [0] * 5
    assert (
        prefill.kv_indptr.untyped_storage().data_ptr()
        == verify.kv_indptr.untyped_storage().data_ptr()
    )
    assert (
        prefill.qo_indptr.untyped_storage().data_ptr()
        == verify.qo_indptr.untyped_storage().data_ptr()
    )
    assert (
        prefill.kv_indices.untyped_storage().data_ptr()
        == verify.kv_indices.untyped_storage().data_ptr()
    )
    assert prefill.kv_indices.numel() == 3
    assert verify.kv_indices.numel() == 4


def test_composition_kv_indices_arena_grows_and_then_reuses_storage():
    scratch = CompositeForwardMetadataScratch(
        int32_indptr=torch.zeros((2, 5), dtype=torch.int32),
        int64_indptr=torch.zeros((4, 5), dtype=torch.int64),
    )
    scratch.segments(3, 4)
    first_storage = scratch.kv_indices.untyped_storage().data_ptr()
    first_capacity = scratch.kv_indices.numel()

    scratch.segments(2, 2)
    assert scratch.kv_indices.untyped_storage().data_ptr() == first_storage
    assert scratch.kv_indices.numel() == first_capacity

    scratch.segments(first_capacity, 1)
    assert scratch.kv_indices.numel() >= first_capacity + 1
    assert scratch.kv_indices.numel() & (scratch.kv_indices.numel() - 1) == 0


def test_pack_reuses_persistent_buffers_and_precomputes_logits_gather():
    scratch = ForwardCompositionTensorScratch()

    def build():
        prefill = _batch(ForwardMode.EXTEND, num_tokens=3, batch_size=2)
        prefill.extend_seq_lens = torch.tensor([2, 1], dtype=torch.int32)
        prefill.extend_prefix_lens = torch.tensor([8, 19], dtype=torch.int64)
        prefill.positions = torch.tensor([10, 11, 20], dtype=torch.int64)
        verify = _batch(ForwardMode.TARGET_VERIFY, num_tokens=4, batch_size=2)
        verify.seq_lens = torch.tensor([32, 42], dtype=torch.int64)
        verify.positions = torch.tensor([30, 31, 40, 41], dtype=torch.int64)
        return pack_prefill_and_verify_forward(prefill, verify, scratch=scratch)

    first = build()
    input_storage = first.input_ids.untyped_storage().data_ptr()
    gather_storage = first.composition.logits_gather_indices.untyped_storage().data_ptr()
    assert first.composition.logits_gather_indices.tolist() == [1, 2, 3, 4, 5, 6]

    second = build()
    assert second.input_ids.untyped_storage().data_ptr() == input_storage
    assert (
        second.composition.logits_gather_indices.untyped_storage().data_ptr()
        == gather_storage
    )
    assert (
        second.composition.build_logits_gather_indices().data_ptr()
        == second.composition.logits_gather_indices.data_ptr()
    )


def _bare_triton_backend() -> TritonAttnBackend:
    backend = TritonAttnBackend.__new__(TritonAttnBackend)
    backend.dcp_size = 1
    backend.enable_deterministic = False
    backend.sliding_window_size = None
    return backend


def _bare_fa3_backend() -> FlashAttentionBackend:
    backend = FlashAttentionBackend.__new__(FlashAttentionBackend)
    backend.fa_impl_ver = 3
    backend.attn_cp_size = 1
    backend.use_mla = False
    backend.has_local_attention = False
    backend.has_swa = False
    backend.is_encoder_decoder = False
    backend.fa_skip_kv_cache = False
    backend.topk = 1
    backend.speculative_num_draft_tokens = 2
    return backend


def test_triton_composition_capability_accepts_p0_shape():
    backend = _bare_triton_backend()

    assert backend.supports_forward_composition(
        "prefill_spec_verify", topk=1, fixed_q_len=2
    )


def test_fa3_composition_capability_accepts_p0_shape():
    backend = _bare_fa3_backend()

    assert backend.supports_forward_composition(
        "prefill_spec_verify", topk=1, fixed_q_len=2
    )


@pytest.mark.parametrize(
    ("attribute", "value"),
    [
        ("fa_impl_ver", 4),
        ("attn_cp_size", 2),
        ("use_mla", True),
        ("has_local_attention", True),
        ("has_swa", True),
        ("is_encoder_decoder", True),
        ("fa_skip_kv_cache", True),
    ],
)
def test_fa3_composition_capability_rejects_unsupported_runtime_modes(
    attribute, value
):
    backend = _bare_fa3_backend()
    setattr(backend, attribute, value)

    assert not backend.supports_forward_composition(
        "prefill_spec_verify", topk=1, fixed_q_len=2
    )


@pytest.mark.parametrize(
    ("kind", "topk", "fixed_q_len"),
    [
        ("unknown", 1, 2),
        ("prefill_spec_verify", 0, 2),
        ("prefill_spec_verify", 2, 2),
        ("prefill_spec_verify", 1, 0),
    ],
)
def test_triton_composition_capability_rejects_unsupported_shape(
    kind, topk, fixed_q_len
):
    backend = _bare_triton_backend()

    assert not backend.supports_forward_composition(
        kind, topk=topk, fixed_q_len=fixed_q_len
    )


def test_triton_composition_capability_accepts_deterministic_mode():
    backend = _bare_triton_backend()
    backend.enable_deterministic = True

    assert backend.supports_forward_composition(
        "prefill_spec_verify", topk=1, fixed_q_len=2
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_batch_invariant_qkv_is_exact_across_packed_m_dimension():
    from sglang.srt.batch_invariant_ops import (
        disable_batch_invariant_mode,
        enable_batch_invariant_mode,
    )

    generator = torch.Generator(device="cuda").manual_seed(19)
    hidden = torch.randn(
        (257, 320), device="cuda", dtype=torch.bfloat16, generator=generator
    )
    qkv_weight = torch.randn(
        (320, 384), device="cuda", dtype=torch.bfloat16, generator=generator
    )

    enable_batch_invariant_mode(enable_bmm=False)
    try:
        separated = torch.mm(hidden[:4], qkv_weight)
        packed_view = torch.mm(hidden, qkv_weight)[:4]
        assert torch.equal(separated, packed_view)
    finally:
        disable_batch_invariant_mode()


def test_triton_composition_capability_rejects_unsupported_runtime_modes():
    backend = _bare_triton_backend()
    backend.dcp_size = 2
    assert not backend.supports_forward_composition(
        "prefill_spec_verify", topk=1, fixed_q_len=2
    )

    backend.dcp_size = 1
    backend.sliding_window_size = 4096
    assert not backend.supports_forward_composition(
        "prefill_spec_verify", topk=1, fixed_q_len=2
    )


def test_logits_parity_locates_first_fork_and_top2_margin():
    reference = torch.tensor([[5.0, 4.0, 0.0], [1.00, 0.99, 0.0]])
    actual = torch.tensor([[5.0, 4.0, 0.0], [0.98, 1.01, 0.0]])

    report = logits_parity(reference, actual)

    assert report["first_divergent_row"] == 1
    assert report["num_argmax_divergences"] == 1
    assert report["first_divergence"]["reference_top2_ids"] == [0, 1]
    assert report["first_divergence"]["actual_top2_ids"] == [1, 0]
    assert report["first_divergence"]["reference_margin"] == pytest.approx(0.01)


def test_attention_parity_joins_separated_segments_before_comparison():
    reference = AttentionTrace()
    reference.record(0, torch.tensor([[1.0], [2.0]]))
    reference.record(0, torch.tensor([[3.0]]))
    actual = AttentionTrace()
    actual.record(0, torch.tensor([[1.0], [2.0], [3.25]]))

    report = attention_parity(reference, actual)

    assert report["0"]["shape"] == [3, 1]
    assert report["0"]["max_abs"] == pytest.approx(0.25)


def test_operator_parity_hooks_compare_only_selected_token_rows():
    class ToyAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qkv_proj = torch.nn.Linear(2, 3, bias=False)

        def forward(self, value):
            return self.qkv_proj(value)

    class ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = ToyAttention()

        def forward(self, value):
            return self.self_attn(value)

    model = ToyModel()
    handles = install_operator_trace_hooks(model)
    try:
        reference = OperatorTrace()
        with record_operators(reference, token_start=0):
            model(torch.tensor([[3.0, 4.0]]))

        actual = OperatorTrace()
        packed = torch.tensor([[100.0, 200.0], [3.0, 4.0]])
        with record_operators(actual, token_start=1):
            model(packed)
    finally:
        remove_operator_trace_hooks(handles)

    report = operator_parity(reference, actual)
    assert report["first_mismatch"] is None
    assert report["operators"][0]["shape"][0] == 1

    actual.outputs[actual.order[-1]][0, 0].add_(1.0)
    report = operator_parity(reference, actual)
    assert report["first_mismatch"]["max_abs"] == pytest.approx(1.0)


class _FakeKVPool:
    def __init__(self):
        self.k = [torch.arange(12, dtype=torch.float32).view(4, 3)]
        self.v = [torch.arange(12, 24, dtype=torch.float32).view(4, 3)]

    def get_kv_buffer(self, layer_id):
        return self.k[layer_id], self.v[layer_id]


def test_kv_rows_capture_compare_and_restore_selected_locations():
    pool = _FakeKVPool()
    locations = torch.tensor([3, 1, 3])
    initial = KVRows.capture(pool, [0], locations)
    pool.k[0][1].add_(0.5)
    changed = KVRows.capture(pool, [0], locations)

    report = initial.compare(changed)
    assert report["0"]["key"]["max_abs"] == pytest.approx(0.5)
    assert report["0"]["value"]["max_abs"] == 0.0

    initial.restore(pool)
    restored = KVRows.capture(pool, [0], locations)
    assert initial.compare(restored)["0"]["key"]["max_abs"] == 0.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA GPU")
def test_gpu_logits_and_kv_parity_metrics_stay_on_device():
    reference = torch.randn((8, 128), device="cuda", dtype=torch.bfloat16)
    actual = reference.clone()
    actual[3, 7] += 0.125
    logits_report = logits_parity(reference, actual)
    assert logits_report["tensor_error"]["max_abs"] > 0

    class GPUKVPool:
        def __init__(self):
            self.k = [torch.randn((16, 2, 8), device="cuda", dtype=torch.bfloat16)]
            self.v = [torch.randn((16, 2, 8), device="cuda", dtype=torch.bfloat16)]

        def get_kv_buffer(self, layer_id):
            return self.k[layer_id], self.v[layer_id]

    pool = GPUKVPool()
    locations = torch.tensor([2, 5, 9], device="cuda")
    snapshot = KVRows.capture(pool, [0], locations)
    snapshot.restore(pool)
    second = KVRows.capture(pool, [0], locations)
    assert snapshot.compare(second)["0"]["key"]["max_abs"] == 0.0


def test_triton_composition_matches_two_separate_segment_baselines():
    backend = _bare_triton_backend()
    prefill_metadata = object()
    verify_metadata = object()
    saved_metadata = object()
    backend.forward_metadata = saved_metadata
    backend.forward_composition_metadata = CompositePrefillVerifyMetadata(
        prefill=prefill_metadata,
        verify=verify_metadata,
        prefill_num_tokens=3,
        verify_num_tokens=4,
    )
    composition = _composition()
    parent = _batch(ForwardMode.MIXED, num_tokens=7, batch_size=3)
    parent.composition = composition
    q = torch.arange(14, dtype=torch.float32).view(7, 2)
    k = q + 100
    v = q + 200
    calls = []

    def fake_forward(q_part, k_part, v_part, layer, child_batch, **kwargs):
        calls.append(
            (
                kwargs["forward_metadata"],
                q_part,
                k_part,
                v_part,
                child_batch,
                kwargs,
            )
        )
        offset = 10 if child_batch is composition.prefill_batch else 20
        child_batch._attn_output.copy_(q_part + offset)
        return child_batch._attn_output

    backend.forward_extend = fake_forward
    output = TritonAttnBackend._forward_extend_composition(
        backend,
        q,
        k,
        v,
        layer=SimpleNamespace(qk_head_dim=2, v_head_dim=2, tp_q_head_num=1),
        forward_batch=parent,
        save_kv_cache=True,
    )

    assert len(calls) == 2
    assert calls[0][0] is prefill_metadata
    assert calls[1][0] is verify_metadata
    assert torch.equal(calls[0][1], q[:3])
    assert torch.equal(calls[1][1], q[3:])
    assert torch.equal(calls[0][2], k[:3])
    assert torch.equal(calls[1][3], v[3:])
    assert calls[0][4] is composition.prefill_batch
    assert calls[1][4] is composition.verify_batch
    assert calls[0][5]["save_kv_cache"] is True
    separate_baseline = torch.empty_like(q)
    separate_baseline[:3].copy_(q[:3] + 10)
    separate_baseline[3:].copy_(q[3:] + 20)
    assert torch.equal(output, separate_baseline)
    assert backend.forward_metadata is saved_metadata
    assert composition.prefill_batch._attn_output is None
    assert composition.verify_batch._attn_output is None


def test_triton_composition_forward_restores_metadata_after_segment_error():
    backend = _bare_triton_backend()
    saved_metadata = object()
    verify_metadata = object()
    backend.forward_metadata = saved_metadata
    backend.forward_composition_metadata = CompositePrefillVerifyMetadata(
        prefill=object(),
        verify=verify_metadata,
        prefill_num_tokens=3,
        verify_num_tokens=4,
    )
    composition = _composition()
    parent = _batch(ForwardMode.MIXED, num_tokens=7, batch_size=3)
    parent.composition = composition

    def fake_forward(q_part, k_part, v_part, layer, child_batch, **kwargs):
        if kwargs["forward_metadata"] is verify_metadata:
            raise RuntimeError("verify failed")
        child_batch._attn_output.copy_(q_part)
        return child_batch._attn_output

    backend.forward_extend = fake_forward
    q = torch.zeros((7, 2))

    with pytest.raises(RuntimeError, match="verify failed"):
        TritonAttnBackend._forward_extend_composition(
            backend,
            q,
            q,
            q,
            layer=SimpleNamespace(qk_head_dim=2, v_head_dim=2, tp_q_head_num=1),
            forward_batch=parent,
            save_kv_cache=True,
        )

    assert backend.forward_metadata is saved_metadata
    assert composition.prefill_batch._attn_output is None
    assert composition.verify_batch._attn_output is None


def test_fa3_composition_metadata_builds_two_independent_plans():
    backend = _bare_fa3_backend()
    backend.forward_metadata = None
    backend.forward_composition_metadata = None
    composition = _composition()
    parent = _batch(ForwardMode.MIXED, num_tokens=7, batch_size=3)
    parent.composition = composition
    plans = {
        id(composition.prefill_batch): object(),
        id(composition.verify_batch): object(),
    }
    calls = []

    def fake_build(child_batch):
        calls.append(child_batch)
        backend.forward_metadata = plans[id(child_batch)]
        return plans[id(child_batch)]

    backend._build_forward_metadata = fake_build
    backend.init_forward_metadata(parent)

    assert calls == [composition.prefill_batch, composition.verify_batch]
    assert backend.forward_composition_metadata.prefill is plans[
        id(composition.prefill_batch)
    ]
    assert backend.forward_composition_metadata.verify is plans[
        id(composition.verify_batch)
    ]
    assert backend.forward_metadata is plans[id(composition.prefill_batch)]


def test_fa3_composition_uses_views_and_native_forward_extend_twice():
    backend = _bare_fa3_backend()
    prefill_metadata = object()
    verify_metadata = object()
    saved_metadata = object()
    backend.forward_metadata = saved_metadata
    backend.forward_composition_metadata = (
        CompositePrefillVerifyFlashAttentionMetadata(
            prefill=prefill_metadata,
            verify=verify_metadata,
            prefill_num_tokens=3,
            verify_num_tokens=4,
        )
    )
    composition = _composition()
    parent = _batch(ForwardMode.MIXED, num_tokens=7, batch_size=3)
    parent.composition = composition
    q = torch.arange(14, dtype=torch.float32).view(7, 2)
    k = q + 100
    v = q + 200
    calls = []

    def fake_forward(q_part, k_part, v_part, layer, child_batch, **kwargs):
        calls.append((q_part, k_part, v_part, child_batch, kwargs))
        offset = 10 if child_batch is composition.prefill_batch else 20
        child_batch._attn_output.copy_(q_part + offset)
        return child_batch._attn_output

    backend.forward_extend = fake_forward
    output = FlashAttentionBackend._forward_extend_composition(
        backend,
        q,
        k,
        v,
        layer=SimpleNamespace(v_head_dim=2, tp_q_head_num=1),
        forward_batch=parent,
        save_kv_cache=True,
    )

    assert len(calls) == 2
    assert calls[0][4]["forward_metadata"] is prefill_metadata
    assert calls[1][4]["forward_metadata"] is verify_metadata
    assert calls[0][0].untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    assert calls[1][0].untyped_storage().data_ptr() == q.untyped_storage().data_ptr()
    assert calls[0][1].untyped_storage().data_ptr() == k.untyped_storage().data_ptr()
    assert calls[1][2].untyped_storage().data_ptr() == v.untyped_storage().data_ptr()
    assert torch.equal(output[:3], q[:3] + 10)
    assert torch.equal(output[3:], q[3:] + 20)
    assert backend.forward_metadata is saved_metadata
    assert composition.prefill_batch._attn_output is None
    assert composition.verify_batch._attn_output is None


def test_fa3_composition_restores_child_output_views_after_segment_error():
    backend = _bare_fa3_backend()
    verify_metadata = object()
    backend.forward_composition_metadata = (
        CompositePrefillVerifyFlashAttentionMetadata(
            prefill=object(),
            verify=verify_metadata,
            prefill_num_tokens=3,
            verify_num_tokens=4,
        )
    )
    composition = _composition()
    parent = _batch(ForwardMode.MIXED, num_tokens=7, batch_size=3)
    parent.composition = composition

    def fake_forward(q_part, k_part, v_part, layer, child_batch, **kwargs):
        if kwargs["forward_metadata"] is verify_metadata:
            raise RuntimeError("verify failed")
        child_batch._attn_output.copy_(q_part)
        return child_batch._attn_output

    backend.forward_extend = fake_forward
    q = torch.zeros((7, 2))

    with pytest.raises(RuntimeError, match="verify failed"):
        FlashAttentionBackend._forward_extend_composition(
            backend,
            q,
            q,
            q,
            layer=SimpleNamespace(v_head_dim=2, tp_q_head_num=1),
            forward_batch=parent,
            save_kv_cache=True,
        )

    assert composition.prefill_batch._attn_output is None
    assert composition.verify_batch._attn_output is None
