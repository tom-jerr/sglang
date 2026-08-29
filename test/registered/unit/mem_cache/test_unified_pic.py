import threading
import unittest

import torch
from sglang.srt.mem_cache.pic.cdc import ContentDefinedChunker, content_digest
from sglang.srt.mem_cache.pic.component import PicSpanComponent
from sglang.srt.mem_cache.pic.config import PicConfig
from sglang.srt.mem_cache.pic.mla import delta_rotate_kr
from sglang.srt.mem_cache.pic.registry import PicSpanRegistry
from sglang.srt.mem_cache.pic.types import (
    DependencyMode,
    PicNamespace,
    PicSpanIdentity,
    PicSpanPayload,
    SeamMetadata,
    ShareScope,
)


def _namespace(tenant: str = "tenant-a", *, session: str | None = None) -> PicNamespace:
    return PicNamespace(
        tenant_id=tenant,
        session_id=session,
        share_scope=ShareScope.SESSION if session else ShareScope.TENANT,
        model_fingerprint="deepseek-ai/DeepSeek-V2-Lite@main",
        tokenizer_fingerprint="deepseek-v2-tokenizer",
        cache_format="mla-bf16-ckv-kr-v1",
    )


class TestContentDefinedChunker(unittest.TestCase):
    def test_chunks_are_content_defined_not_absolute_position_defined(self):
        config = PicConfig(
            min_chunk_tokens=32,
            target_chunk_tokens=64,
            max_chunk_tokens=128,
        )
        chunker = ContentDefinedChunker(config)
        tokens = tuple((i * 7919) % 32000 for i in range(1024))

        at_64 = chunker.chunks(tokens, absolute_start=64)
        at_509 = chunker.chunks(tokens, absolute_start=509)

        self.assertEqual(
            [(c.start - 64, c.end - 64, c.digest) for c in at_64],
            [(c.start - 509, c.end - 509, c.digest) for c in at_509],
        )
        self.assertTrue(all(32 <= c.token_count <= 128 for c in at_64[:-1]))

    def test_empty_input(self):
        self.assertEqual(ContentDefinedChunker(PicConfig()).chunks(()), ())

    def test_64_token_marker_stabilizes_shifted_shared_region(self):
        chunker = ContentDefinedChunker(
            PicConfig(
                min_chunk_tokens=32,
                target_chunk_tokens=64,
                max_chunk_tokens=128,
            )
        )
        marker = tuple(50000 + i for i in range(64))
        shared = tuple((i * 104729) % 64000 for i in range(2048))
        first_prefix = tuple(range(71))
        second_prefix = tuple(range(9000, 9173))
        first = first_prefix + marker + shared
        second = second_prefix + marker + shared
        first_shared_start = len(first_prefix) + len(marker)
        second_shared_start = len(second_prefix) + len(marker)

        first_digests = {
            chunk.digest
            for chunk in chunker.chunks(first)
            if chunk.start >= first_shared_start
        }
        second_digests = {
            chunk.digest
            for chunk in chunker.chunks(second)
            if chunk.start >= second_shared_start
        }
        self.assertGreater(len(first_digests & second_digests), 8)


class TestPicSpanRegistry(unittest.TestCase):
    def setUp(self):
        self.tokens = tuple(range(64))
        self.identity = PicSpanIdentity(
            namespace_digest=_namespace().fingerprint(),
            content_digest=content_digest(self.tokens),
            token_count=len(self.tokens),
        )

    def _register(self, registry, payload_handle, on_retire=None):
        return registry.register(
            identity=self.identity,
            token_ids=self.tokens,
            dependency_mode=DependencyMode.MLA_POSITION_FREE,
            seam=SeamMetadata(marker_stabilized=True),
            payload=PicSpanPayload(
                c_kv_handle=payload_handle,
                k_r_base_handle=f"kr-{payload_handle}",
                source_position=96,
            ),
            on_retire=on_retire,
        )

    def test_concurrent_publish_has_one_canonical_payload(self):
        registry = PicSpanRegistry(max_entries=100)
        barrier = threading.Barrier(24)
        results = []
        results_lock = threading.Lock()

        def publish(worker_id):
            barrier.wait()
            result = self._register(registry, f"candidate-{worker_id}")
            with results_lock:
                results.append(result)

        threads = [threading.Thread(target=publish, args=(i,)) for i in range(24)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())

        self.assertEqual(sum(result.inserted for result in results), 1)
        self.assertEqual({result.record.handle for result in results}, {1})
        self.assertEqual(
            {result.record.payload.c_kv_handle for result in results},
            {results[0].record.payload.c_kv_handle},
        )

    def test_hash_collision_is_resolved_by_exact_tokens(self):
        registry = PicSpanRegistry(max_entries=100)
        first = self._register(registry, "first")
        collided_tokens = tuple(range(1, 65))
        second = registry.register(
            identity=self.identity,
            token_ids=collided_tokens,
            dependency_mode=DependencyMode.MLA_POSITION_FREE,
            seam=SeamMetadata(),
            payload=PicSpanPayload(
                c_kv_handle="second",
                k_r_base_handle="kr-second",
                source_position=128,
            ),
        )

        self.assertTrue(first.inserted)
        self.assertTrue(second.inserted)
        self.assertNotEqual(first.record.handle, second.record.handle)
        self.assertEqual(
            registry.probe(self.identity, collided_tokens).payload.c_kv_handle,
            "second",
        )

    def test_retire_waits_for_last_reader(self):
        retired = []
        registry = PicSpanRegistry(max_entries=100)
        record = self._register(
            registry, "canonical", on_retire=lambda payload: retired.append(payload)
        ).record
        lease = registry.acquire(self.identity, self.tokens, target_position=224)
        self.assertIsNotNone(lease)
        self.assertEqual(lease.delta, 128)

        self.assertTrue(registry.retire(record.handle))
        self.assertIsNone(
            registry.acquire(self.identity, self.tokens, target_position=0)
        )
        self.assertEqual(retired, [])
        lease.release()
        self.assertEqual([payload.c_kv_handle for payload in retired], ["canonical"])
        self.assertEqual(registry.stats()["active_readers"], 0)


class TestPicSpanComponent(unittest.TestCase):
    def test_same_physical_ckv_is_observed_at_different_offsets(self):
        component = PicSpanComponent(
            PicConfig(
                min_chunk_tokens=32,
                target_chunk_tokens=64,
                max_chunk_tokens=128,
            )
        )
        tokens = tuple((i * 3571) % 30000 for i in range(768))
        published = component.observe_publish(
            tokens, namespace=_namespace(), absolute_start=64
        )
        plan = component.observe_match(
            tokens,
            namespace=_namespace(),
            prefix_tokens=0,
            absolute_start=320,
        )

        self.assertGreater(len(published), 1)
        self.assertEqual(plan.hit_tokens, len(tokens))
        self.assertEqual({hit.delta for hit in plan.hits}, {256})
        self.assertEqual(
            len({result.record.handle for result in published}), len(published)
        )

    def test_tenant_and_session_namespace_isolation(self):
        component = PicSpanComponent(PicConfig())
        tokens = tuple(range(256))
        component.observe_publish(
            tokens, namespace=_namespace("tenant-a"), absolute_start=64
        )

        tenant_miss = component.observe_match(
            tokens,
            namespace=_namespace("tenant-b"),
            prefix_tokens=0,
            absolute_start=128,
        )
        session_miss = component.observe_match(
            tokens,
            namespace=_namespace("tenant-a", session="private-session"),
            prefix_tokens=0,
            absolute_start=128,
        )
        self.assertEqual(tenant_miss.hit_tokens, 0)
        self.assertEqual(session_miss.hit_tokens, 0)

    def test_sequence_start_chunk_is_carved_out(self):
        component = PicSpanComponent(
            PicConfig(
                min_chunk_tokens=32,
                target_chunk_tokens=32,
                max_chunk_tokens=64,
            )
        )
        tokens = tuple((i * 17) % 1024 for i in range(256))
        published = component.observe_publish(tokens, namespace=_namespace())
        plan = component.observe_match(tokens, namespace=_namespace(), prefix_tokens=0)

        self.assertTrue(published)
        self.assertGreater(plan.carveout_tokens, 0)
        self.assertTrue(
            any(miss.reason == "first_chunk_carveout" for miss in plan.misses)
        )


class TestMLADeltaRotation(unittest.TestCase):
    @staticmethod
    def _assert_composition(is_neox_style: bool):
        torch.manual_seed(7)
        base = torch.randn(3, 5, 64, dtype=torch.float32)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        source_position = 117
        target_position = 2309

        at_source = delta_rotate_kr(
            base, source_position, inv_freq, is_neox_style=is_neox_style
        )
        corrected = delta_rotate_kr(
            at_source,
            target_position - source_position,
            inv_freq,
            is_neox_style=is_neox_style,
        )
        direct = delta_rotate_kr(
            base, target_position, inv_freq, is_neox_style=is_neox_style
        )
        torch.testing.assert_close(corrected, direct, rtol=2e-4, atol=2e-4)

    def test_neox_rotation_composes(self):
        self._assert_composition(is_neox_style=True)

    def test_interleaved_rotation_composes(self):
        self._assert_composition(is_neox_style=False)

    def test_request_local_deltas_broadcast(self):
        base = torch.randn(2, 4, 64)
        inv_freq = torch.rand(32)
        deltas = torch.tensor([11, -7])
        rotated = delta_rotate_kr(base, deltas, inv_freq)
        expected = torch.stack(
            [delta_rotate_kr(base[i], int(deltas[i]), inv_freq) for i in range(2)]
        )
        torch.testing.assert_close(rotated, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_concurrent_request_local_bf16_views_on_cuda(self):
        torch.manual_seed(11)
        base = torch.randn(32, 128, 64, device="cuda", dtype=torch.bfloat16)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, 64, 2, device="cuda").float() / 64))
        source_positions = torch.arange(32, device="cuda") * 17
        target_positions = source_positions.flip(0) + 4096

        at_source = delta_rotate_kr(base, source_positions, inv_freq)
        corrected = delta_rotate_kr(
            at_source, target_positions - source_positions, inv_freq
        )
        direct = delta_rotate_kr(base, target_positions, inv_freq)

        relative_l2 = (
            corrected.float() - direct.float()
        ).norm() / direct.float().norm()
        self.assertLess(relative_l2.item(), 4.7e-3)

    def test_rejects_wrong_rotary_width(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            delta_rotate_kr(torch.randn(2, 32), 1, torch.randn(32))


class TestPicConfig(unittest.TestCase):
    def test_rejects_live_mode_until_attention_path_exists(self):
        with self.assertRaisesRegex(ValueError, "observer_only"):
            PicConfig(observer_only=False)

    def test_rejects_non_power_of_two_target(self):
        with self.assertRaisesRegex(ValueError, "power of two"):
            PicConfig(target_chunk_tokens=96)


if __name__ == "__main__":
    unittest.main()
