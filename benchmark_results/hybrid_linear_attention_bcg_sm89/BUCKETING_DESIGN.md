# Generic packed-shape bucketing for Prefill BCG and DSpark

## Outcome

The graph identity is split into two independent parts:

1. A graph-family/topology key: phase, backend, LoRA/DSA variant, prefix
   topology, stream, and other control-flow choices.
2. A packed-shape key: `(token_capacity, request_capacity)`.

Only values that change tensor extents, launch grids, or captured control flow
belong in the key. Sequence lengths, request starts, state-slot ids, prefix
flags, tracking indices, and chunk plans remain in stable-address metadata
buffers that are refreshed before replay.

This implementation adds the second dimension without replacing the existing
`ShapeKey.size` field. `size` remains the phase's primary dimension and
`ShapeKey.request_capacity` is optional, so existing one-dimensional decode
and prefill graph users remain source compatible.

## Prefill BCG policy

Breakable prefill accepts an optional sparse table in `--cuda-graph-config`:

```json
{
  "prefill": {
    "backend": "breakable",
    "shape_buckets": [
      [1024, 1], [1024, 4], [1024, 8], [1024, 16], [1024, 32],
      [2048, 1], [2048, 4], [2048, 8], [2048, 16], [2048, 32],
      [4096, 1], [4096, 4], [4096, 8], [4096, 16], [4096, 32],
      [8192, 1], [8192, 4], [8192, 8], [8192, 16], [8192, 32],
      [16384, 1], [16384, 4], [16384, 8], [16384, 16], [16384, 32]
    ]
  }
}
```

The table is sparse by design; there is no implicit Cartesian product.
Selection requires both capacities to cover the runtime batch and chooses the
smallest token tier, then the smallest request tier. The existing maximum 2x
token-padding admission rule still applies. If `shape_buckets` is absent, each
legacy token bucket gets the former implicit request capacity, preserving old
behavior.

All TP ranks receive the same static configuration and select from the same
deterministic rule. Existing DP graph-admission voting remains authoritative
for collective safety.

## Why GDN benefits

The previous capture allocated each token bucket for
`min(request_pool_size, token_capacity)` real requests. That made a nominally
one-dimensional graph work for variable batches, but it over-provisioned every
request-axis buffer.

For a selected `(T, B)` bucket, GDN now allocates:

- `B + 1` request rows, where the final row is a dedicated zero-length dummy;
- stable query offsets, Mamba slot ids, initial-state flags, and tracking rows
  for that capacity;
- stable FLA chunk indices and cumulative chunk offsets, both refreshed in
  place before replay;
- a fixed FLA chunk plan bounded by
  `min(T, ceil(T / chunk_size) + B - 1)`.

There is intentionally no third chunk dimension in the graph key. The chunk
capacity is a deterministic consequence of `(T, B)` and the kernel's fixed
chunk size. Runtime chunk rows are written into the fixed plan and unused rows
target the dummy sequence.

Both chunk inputs are required. FLA's eager helper memoizes offsets derived
from the `cu_seqlens` tensor identity. A graph keeps that tensor address stable,
so using the memoized helper would retain the capture-time single-request
offsets when a multi-request layout is replayed. Passing an explicit stable
`chunk_offsets` buffer removes that hidden host/cache dependency.

For example, at `T=16384`, chunk size 64, and a request pool of 2048, the old
implicit plan can contain 2303 rows. A `B=32` graph needs at most 287 rows,
while request/tracking metadata falls from 2049 internal rows to 33. The exact
GPU-time benefit still depends on the workload, but this removes avoidable
metadata population, H2D traffic, capture memory, and padded GDN work.

## DSpark compatibility

Compact ragged target verify has the same underlying shape problem: a token
tier does not uniquely identify its request-slot geometry. A 192-token tier
may represent 32 requests of width 6 or many more shorter rows. Reusing one
token-only graph can therefore bind incompatible dense/ragged layouts.

The decode runner now records its existing compact-ragged geometry as
`ShapeKey(size=token_tier, request_capacity=captured_slots)`. This is a
no-expansion migration: it makes the two-dimensional contract explicit but
does not capture extra DSpark graphs yet.

The follow-up DSpark rollout should use the same `GraphShapePlanner` while
keeping a separate graph family:

1. Generate a bounded sparse slot-tier set per token tier. Candidate slot
   capacities must satisfy `ceil(T / verify_width) <= B <= min(T, max_bs)`.
2. Start with workload-derived or configured slot tiers; do not generate a
   full Cartesian product. A geometric set plus observed hot geometries is a
   reasonable default.
3. Key `_captured_ragged_layouts`, graph outputs, and backend metadata by the
   full `ShapeKey`, not by token tier alone.
4. At admission, select a graph covering both total verify tokens and real
   requests. If no pair exists, fall back to eager.
5. A graph may cover a smaller live request count only after every DSpark/DSV4
   consumer is audited for zero-length padded rows. Until then, select an exact
   slot tier or reject replay.

Prefill GDN and DSpark should not share metadata buffers or graph families.
They share only the finite-envelope type, validation, and deterministic bucket
selection algorithm.

## Correctness invariants

- `0 < real_tokens <= token_capacity`.
- `0 < real_requests <= request_capacity <= token_capacity`.
- A configured prefill shape must be able to represent its token capacity
  within the model context length.
- Request capacity cannot exceed the request-pool capacity.
- Captured tensor addresses never change; replay mutates buffer contents only.
- Tensor-derived launch-layout inputs are explicit; replay never relies on a
  host memoization cache keyed only by tensor identity.
- Dummy GDN rows use a dedicated state slot and masked tracking steps.
- Missing two-dimensional coverage causes eager fallback, never selection of
  a graph with incompatible geometry.
- Graph selection must be deterministic and rank-consistent before entering
  TP/DP collectives.

## Validation completed

- CPU/config/runner suites cover sparse selection, padding rejection,
  graph-key uniqueness, legacy projection, request-pool bounds, and GDN
  metadata sizing.
- A CUDA regression captures a one-request GDN layout and replays a
  two-request layout through the same addresses; output and final recurrent
  state match eager bit-for-bit.
- Qwen3.5-0.8B on RTX 4090 replayed the same 993-token, three-request workload
  with legacy `(1024, 16)` and sparse `(1024, 4)` graphs. Generated token ids
  and per-token log probabilities matched exactly across the two servers.
- Qwen3.8-27B-FP8 TP4 on four RTX 4090s captured `(1024, 4)`, `(2048, 4)`, and
  `(4096, 4)` in 10.82 seconds using 0.59 GB per rank. A four-request,
  3384-token prefill selected BCG and all four stable prompts produced the
  same token ids. Against the prior three-tier legacy capture (13.1 seconds,
  0.56 GB), capture time fell 17.4%, while measured graph memory rose slightly;
  request bucketing should therefore be treated as a replay/work reduction,
  not yet as a demonstrated graph-memory reduction.

## Remaining validation plan

1. Capture/replay Qwen3.8 on Ada SM89 with request tiers 1/4/8/16/32 and token
   tiers 1K/2K/4K/8K/16K.
2. Compare output tokens/logits against eager-break BCG for exact and padded
   shapes.
3. Profile graph replay, host preparation, H2D metadata copies, GDN launch
   grids, and GPU idle gaps.
4. Measure TTFT and input-token throughput by `(input length, concurrency)`,
   reporting eager fallback and selected `(T, B)` for every point.
5. Before enabling multiple DSpark slot tiers, run allocator/canary tests over
   every zero-row consumer and verify TP2/TP4 graph-key agreement.
