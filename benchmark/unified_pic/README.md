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

## Initial two-RTX-4090 validation

Validated on 2026-08-30 with two RTX 4090 GPUs connected over PCIe/PHB (no
NVLink), SGLang upstream revision `97781eb7`, and
`TechxGenus/DeepSeek-V2-Lite-AWQ` revision
`f96b1c6ffa7c53b642b6e6bbf1b75405d1558f81`. The model is native
`DeepseekV2ForCausalLM`; its live KV cache remained FP16. HiCache, PD,
speculative decoding, overlap scheduling, and CUDA Graph were disabled.

This particular AWQ checkpoint cannot be split with TP=2: the rank-local
`down_proj` input size is not aligned to the AWQ group shape. The validation
therefore used `TP=1, PP=2`, which still executes every request across both
GPUs without changing quantized weight semantics. Each stage loaded 4.11/4.64
GiB of weights and allocated an FP16 MLA KV pool; observed total device memory
was 16.96/18.29 GiB.

Use the same server arguments for both runs and add `--enable-unified-pic` only
to the observer run:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=python python -m sglang.launch_server \
  --model-path /workspace/models/DeepSeek-V2-Lite-AWQ \
  --tp-size 1 --pp-size 2 \
  --host 127.0.0.1 --port 30000 \
  --context-length 4096 --mem-fraction-static 0.75 \
  --cuda-graph-backend-decode disabled \
  --cuda-graph-backend-prefill disabled \
  --disable-overlap-schedule --max-running-requests 16 \
  --random-seed 20260830 --attention-backend triton
```

The workload has a 64-token CDC stabilization marker, 1,536 shared body
tokens, and eight distinct prefixes of 69 through 328 tokens. The first token
differs in every request, so ordinary radix matching reports zero cached
tokens. Run both the eight-request concurrency test and the sequential output
oracle:

```bash
python benchmark/unified_pic/validate_dual_gpu.py \
  --label RUN_LABEL --output RUN.json \
  --concurrency 8 --shared-tokens 1536 --max-new-tokens 8 \
  --request-mode concurrent

python benchmark/unified_pic/validate_dual_gpu.py \
  --label RUN_LABEL --output RUN.json \
  --concurrency 8 --shared-tokens 1536 --max-new-tokens 8 \
  --request-mode sequential

python benchmark/unified_pic/compare_runs.py \
  BASELINE.json PIC.json --output COMPARISON.json
```

Initial result:

| Check | Result |
| --- | --- |
| PP=2 startup and FP16 MLA KV allocation | pass on both RTX 4090s |
| Eight independent concurrent HTTP requests | 8/8 successful |
| Ordinary radix prefix reuse | 0 tokens, as intended |
| Runtime PIC candidate coverage | 10,202 / 14,388 prompt tokens (70.91%) |
| Runtime candidate spans | 127; 15-16 per request |
| Request position deltas | -27 through +232 tokens |
| Sequential baseline/PIC output oracle | 8/8 texts and 64/64 token IDs exact |
| Sequential observer latency | 0.4885 s vs 0.4876 s baseline mean |
| Physical `c_KV` remap / skipped prefill | not implemented in observer phase |

The focused regression suite completed with 23 passed tests plus 8 subtests:
all 16 PIC tests and seven UnifiedRadixCache allocation/eviction/component
tests. A broader HiCache test reaches an unrelated host prerequisite: the JIT
native hash extension cannot compile because this image lacks the OpenSSL
development header `openssl/sha.h`. HiCache remains disabled for this phase.

The unconstrained concurrent and fixed-batch A/B runs each showed one greedy
token divergence across independent server restarts even though observer mode
does not mutate the live KV map. SGLang's deterministic-inference mode cannot
serve as an oracle on this hardware/configuration: its default FA3 path rejects
MLA on SM89, while the Triton batch-invariant MLA BMM requests 106,496 bytes of
shared memory against the RTX 4090 limit of 101,376 bytes. The sequential
single-request oracle removes batch-shape variation and is exact. Consequently,
the latency differences from the observer runs are not claimed as PIC speedups;
only candidate coverage, concurrent safety, and the sequential output oracle
are phase-1 conclusions.

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
