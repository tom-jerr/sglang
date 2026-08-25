# Hybrid linear-attention prefill BCG design

## Status and scope

This design enables breakable CUDA graph (BCG) capture for the Gated Delta Net
(GDN) linear-attention body used by hybrid Qwen models. It was developed with
Qwen3.5-0.8B and accepted with the official Qwen3.8-27B-FP8 checkpoint on four
NVIDIA Ada (SM89) GPUs. Qwen3.8 was exercised with TP2 and TP4, native MTP, and
both full and breakable decode CUDA graphs.

The implementation change targets prefill. Decode graph and native MTP paths
are not modified here, but they are part of the acceptance matrix because they
share model state, Mamba cache allocation, graph buckets, and memory budget.
DFLASH/DSpark/EAGLE3 auxiliary hidden-state capture remains a separate
workstream.

## Problem

Qwen3.5-0.8B has 24 language-model layers: 18 GDN linear-attention layers and
6 full-attention layers. Before this change, dynamic GDN metadata forced a BCG
break at every GDN layer. A 385-token profile therefore replayed 25 graph
segments and launched the GDN bodies eagerly between them.

The dynamic inputs include:

- GDN chunk indices and initial-state flags;
- translated Mamba cache slots;
- convolution-window indices;
- radix-cache checkpoint destinations;
- intermediate and final SSM-state source indices.

Their values change per batch, but their storage addresses and shapes do not
need to change within a captured token bucket.

## Goals

- Capture consecutive GDN layer bodies without a per-layer graph break.
- Preserve full-attention breaks where metadata remains dynamic.
- Keep radix-cache checkpoint tracking active during capture and replay.
- Preserve output token correctness and deterministic replay.
- Fall back to the established eager-break path when capture is not profitable
  or its invariants do not hold.

## Non-goals

- Capture target-verify or non-extend forward modes.
- Remove all full-attention graph breaks.
- Change GDN mathematical kernels or their numerical precision.
- Claim one fixed token threshold is optimal on every GPU architecture.

## Execution design

For each configured prefill token bucket, capture allocates fixed-capacity
metadata once. Replay updates those buffers once during forward preparation,
then every GDN layer reads the same stable addresses.

```text
request batch
    |
    v
populate stable bucket metadata once
    |
    v
graph segment -> full-attention break -> graph segment -> ... -> graph segment
                 ^
                 only genuinely dynamic attention bodies remain eager
```

For Qwen3.5-0.8B this changes the expected segmentation from one boundary per
model layer (25 launches) to six full-attention boundaries plus the final
segment (7 launches).

### Capture admission

`GDNAttnBackend.can_capture_attention_body` admits a GDN body only when:

- captured-forward metadata is enabled;
- the forward mode is extend/prefill, not target verify;
- fixed GDN chunk metadata is present.

Replay applies the same forward-mode restriction. Active radix tracking is not
a rejection condition because its inputs are represented by stable buffers and
mask values.

### Stable chunk metadata

The capture metadata owns fixed-size tensors for query offsets, sequence
lengths, cache indices, initial-state flags, and GDN chunk indices. A chunk plan
contains all real `(request, chunk)` rows followed by padded rows. Padded rows
refer to an extra zero-length dummy request, so they cannot alias a real state
slot.

The request capacity is bounded by the smaller of the request pool and token
bucket, plus the dummy request. The chunk-plan capacity is:

```text
min(token_capacity,
    ceil(token_capacity / FLA_CHUNK_SIZE) + max_real_requests - 1)
```

### Stable radix-tracking metadata

Radix tracking checkpoints convolution and SSM state at request-specific
positions. The capture metadata holds fixed-capacity tensors for:

- convolution source token indices and destination cache slots;
- intermediate SSM source and destination slots;
- final SSM source and destination slots;
- one step mask for each operation (`0` for live rows and `-1` for no-op rows).

Replay clears the masks, translates live request indices, and fills valid rows.
The captured graph topology never changes when a request starts or stops
tracking.

### Mask-first Triton scatter

Two capture-safe Triton helpers replace eager advanced indexing:

- `scatter_gdn_prefill_conv_states_with_mask` copies convolution windows;
- `scatter_gdn_prefill_states_with_mask` copies intermediate or final SSM rows.

Each program reads the step mask first. A padded row returns before reading a
source or destination index. This ordering is required because padded fixed-
capacity rows can deliberately contain out-of-range sentinel indices.

The helpers accept envelope-strided Mamba pool destinations as long as the
trailing state entry is contiguous. They use the real request stride rather
than requiring the entire tensor to be contiguous.

### Configurable radix-tracking threshold

With radix tracking enabled, GDN-body admission is controlled by
`--gdn-bcg-tracking-capture-max-tokens`. The default remains the conservative
512-token limit established by the small-model Ada profile. A value of zero
disables tracking capture, and buckets above the configured limit retain the
existing per-layer eager GDN breaks.

The threshold is a measured performance guard, not a correctness limit or a
universal architecture constant. The initial Qwen3.5 experiment found that an
unconditional 1024-token capture could regress on RTX 4090. The later
Qwen3.8-27B-FP8 TP4 matrix found that 1024/2048/4096-token captures were
profitable when the server used a 4096-token chunk size and the model's 48 GDN
layers supplied enough removable host work. The runtime option preserves both
results: safe defaults for unknown workloads and an explicit profiled override
for the validated Qwen3.8 deployment.

The limit applies to the scheduled prefill token bucket, not the original
prompt length. With `chunked_prefill_size=4096`, an 8K or 16K request replays
the 4096-token graph two or four times. Capturing 8K/16K buckets is therefore
unnecessary for this policy and wastes graph memory.

When radix cache is disabled, the extra tracking work is absent and the backend
can capture larger buckets. This remains subject to model- and hardware-level
benchmarking.

## Correctness invariants

1. Every tensor referenced by a captured GDN body has a stable address for the
   life of that bucket graph.
2. Replay changes tensor contents, never tensor shapes or Python control flow.
3. Padded rows are masked before any potentially invalid memory read.
4. The dummy request cannot alias a live Mamba cache entry.
5. The GDN layer body does not retain a live `ForwardBatch` dependency.
6. Eager metadata and tracking functions remain the fallback for uncaptured
   buckets and unsupported modes.

## Changed components

| Component | Responsibility |
| --- | --- |
| `gdn_backend.py` | Capture admission, stable bucket metadata, radix-tracking preparation, and the configurable tracking threshold |
| `server_args.py` | Exposes and validates `--gdn-bcg-tracking-capture-max-tokens` |
| `mamba2_metadata.py` | Fixed-capacity tracking mask fields |
| `mamba_state_scatter_triton.py` | Mask-first convolution and SSM tracking scatter kernels |
| GDN BCG unit tests | Metadata stability and active-tracking admission |
| Mamba scatter unit tests | CUDA correctness for valid, padded, and sentinel rows |

## Validation strategy

- Compare baseline and BCG output text and token IDs on uncached active-tracking
  prefill.
- Compare selected-token log probabilities and report absolute differences.
- Verify deterministic replay against a repeated BCG request.
- Profile one 385-token prefill and count graph launches, direct kernel
  launches, CPU operations, and GPU kernel time.
- Benchmark 32, 256, and 1024 input tokens at concurrency 1, 8, and 32 with
  decode CUDA graphs disabled to isolate prefill behavior.
- Keep targeted CPU and CUDA unit tests in CI.

## Qwen3.8 acceptance design

The acceptance model is `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. FP8 is required for the TP2
24-GiB configuration; the unquantized checkpoint does not leave sufficient
space for target weights, the native MTP layer, FP32 Mamba state, KV cache, and
captured graphs.

The target has 64 language-model layers: 48 linear-attention layers and 16
full-attention layers, with one native MTP layer. Mamba SSM state remains FP32
as declared by the checkpoint. The acceptance matrix intentionally crosses the
two tensor-parallel sizes with both decode graph backends:

| Dimension | TP2 | TP4 |
| --- | --- | --- |
| No MTP, full decode graph | Capture, replay, correctness, serving benchmark | Capture, replay, correctness, serving benchmark |
| Native MTP, breakable decode graph | Target verify + draft decode/extend capture, replay, correctness, serving benchmark | Same at graph batch sizes 1/2/4 |
| Native MTP, full decode graph | Target verify + draft decode/extend capture, replay, spot correctness, serving benchmark | Same at graph batch sizes 1/2/4 |

Both TP sizes use breakable prefill buckets 64 and 256. TP2 uses one running
request and five Mamba cache entries; TP4 uses four running requests and 20
Mamba cache entries. This keeps native FP32 Mamba state allocation explicit
instead of masking a cache-sizing failure with a lower-precision override.

For MTP, `EAGLE` uses three speculative steps, top-k 1, and four draft tokens.
`enable_linear_replayssm_spec` is deliberately left disabled because the test
is for the checkpoint's native hybrid-state path on SM89. Correctness uses
temperature-zero prompts and output token IDs. Cross-TP reductions may choose a
different token when top logits are nearly tied, so acceptance requires exact
matches for stable prompts, deterministic replay within one configuration, and
semantically correct output for observed near-tie divergences.

### Long-prefill threshold experiment

The 1K-to-16K matrix is a separate performance-capacity experiment from the
FP32 acceptance matrix above. It uses TP4, 4096-token chunks, prefill buckets
1024/2048/4096, decode graphs disabled, MTP disabled, and compares threshold 0
against threshold 4096. Each request generates one token so E2E latency is
effectively TTFT.

Concurrency 32 with 16K prompts requires 524,288 KV tokens. On four 24-GiB
4090s, the tested capacity point uses 160 Mamba cache entries and BF16 Mamba SSM
state, producing 543,634 KV-token capacity. This dtype override is only for the
capacity/performance experiment. FP32 remains the accuracy-qualified default.
Accordingly, performance acceptance and numerical acceptance are reported
separately.

TP2 cannot represent the full requested 16K/C32 matrix on this host: per-rank
weights, 160 hybrid-state entries, and at least 524,288 KV tokens exceed 24 GiB.
TP4 is therefore the only topology used for the full grid; earlier TP2/TP4
small-bucket acceptance remains documented separately.

Three controls reduce false conclusions from short serving runs:

1. every cell flushes radix cache and performs one warmup;
2. the same exact random lengths and seed are used on both sides;
3. the three near-zero or contradictory cells are repeated with two to four
   times as many prompts.

## Follow-up work

- Replace the operator-configured threshold with optional architecture- and
  workload-aware capture policy data once enough benchmark data exists.
- Fuse or batch radix-tracking scatters across GDN layers so 1024-token capture
  also becomes profitable for smaller hybrid models.
- Hoist remaining batch preparation and dtype conversions out of replay.
- Add a repeatable Qwen3.8 capture/replay smoke test that can use a compact or
  synthetic checkpoint in CI; the 28.75-GiB acceptance checkpoint is too large
  for ordinary unit-test jobs.
- Add DFLASH/DSpark/EAGLE stable auxiliary-output sinks independently.
