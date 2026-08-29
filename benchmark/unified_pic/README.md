# Unified PIC for SGLang

This branch starts from upstream `main` at `97781eb7f33ea3b64f8a35bf04dd63833383f292`.
Phase 1 is an MLA-only observer/reference implementation. It intentionally does
not report live TTFT savings: no prompt token is skipped until the split MLA
pool and fused attention correction land.

## Runtime shape

```text
UnifiedRadixCache
├── PREFIX                 existing UnifiedTreeCore
├── PIC-SPAN               content-addressed, position-independent index
│   ├── namespace          tenant + model + tokenizer + cache format
│   ├── strong content hash + exact-token collision check
│   ├── source position and dependency mode
│   ├── seam / first-chunk recompute metadata
│   └── concurrent lease + two-phase retirement
├── FULL / SWA             existing components
├── GDN transition         reserved dependency mode; phase 3
└── HiCache L1/L2/L3       existing residency plane; phase 3 adapter
```

PIC-SPAN is deliberately not a `ComponentType` tree value. FULL/SWA/MAMBA data
is aligned to a prefix-tree node; a content span is found independently of its
absolute location and can be referenced by several tree paths. It therefore
composes beside `UnifiedTreeCore` and uses a lease protocol to join physical
pool lifetime safely.

## Phase 1 invariants

- MLA only; hybrid SWA/SSM, HiCache, PD, and speculative decoding fail fast.
- A canonical span is keyed by 128-bit namespace and content digests, token
  count, and an exact-token collision check.
- Tenant is always in the namespace. Model revision, tokenizer identity, and KV
  format prevent cross-runtime aliasing. Session scope is already representable.
- The first content-defined chunk beginning below position 32 is recomputed.
- `c_KV` is represented by one opaque canonical handle. Concurrent requests
  acquire leases to that handle and carry independent target positions.
- `k_r(target) = R(target - source) k_r(source)`, with `inv_freq` taken from the
  model rotary object. BF16 `k_r` remains request-local in phase 1.
- Retirement first removes a span from lookup and frees the physical payload
  only after the final request/transfer lease releases.
- `--enable-unified-pic` is observer-only: ordinary prefix matching, allocation,
  prefill, attention, and CUDA Graph behavior are unchanged.

Launch an observer server on a GPU that can hold DeepSeek-V2-Lite:

```bash
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --enable-unified-pic \
  --disable-overlap-schedule \
  --log-level debug
```

PIC decisions appear as `PIC observer` debug records. Do not enable HiCache,
PD flags, or speculative decoding in this phase.

For an offline tokenized JSONL trace:

```bash
PYTHONPATH=python python benchmark/unified_pic/phase1_observer.py TRACE.jsonl \
  --model-fingerprint deepseek-ai/DeepSeek-V2-Lite@main \
  --tokenizer-fingerprint deepseek-v2
```

## Phase 2: low-bit canonical c_KV

The physical adapter will split the current `(c_KV, k_r)` row into:

```text
canonical_c_kv_pool[canonical_span, layer, token]  INT8 or FP8
request_k_r_pool[request, logical_position, layer] BF16
logical_map[request, logical_position]             canonical slot + k_r slot
```

Attention must gather canonical `c_KV`, dequantize it, and delta-rotate `k_r`
from a static logical-position buffer in one load path. The benchmark matrix is
BF16 baseline versus INT8 and FP8 canonical spans at the same recovered-token
set. Report HBM/PD bytes saved, fused-kernel time, TTFT, throughput, output
argmax agreement, per-token KL, and relative-L2 correction error. The existing
`feature/fp8-per-token-per-head` work is input to this adapter, but is not
cherry-picked into phase 1 because it currently quantizes per-head MHA-shaped
K/V rather than position-free MLA `c_KV` alone.

## Phase 3: distributed residency and graph safety

- HiCache L3 keys canonical spans by namespace/content/format and transfers one
  low-bit payload regardless of the number of logical request views.
- PD sends only missing canonical spans plus compact logical-map and correction
  metadata; P and D workers share the same generation/lease protocol.
- CUDA Graph captures fixed-capacity `canonical_slot`, `k_r_slot`, `delta`, and
  validity buffers. Request updates mutate buffer contents, never graph shape or
  pointers.
- Speculative decoding journals logical-map mutations and releases leases on
  rollback; canonical payloads are immutable.
- GDN transition spans use the reserved dependency mode and explicit boundary
  state/recompute metadata instead of claiming MLA-style position invariance.

The live gate is strict: no physical reuse until allocator eviction, radix
locking, request teardown, transfer cancellation, and CUDA Graph replay all
participate in the same lease/generation protocol.
