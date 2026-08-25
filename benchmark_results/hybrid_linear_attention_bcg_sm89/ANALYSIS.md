# When hybrid linear-attention BCG is effective

## Decision summary

Enable captured GDN bodies when all of the following are true:

- the model contains many linear-attention layers, preferably in consecutive
  groups between relatively few full-attention layers;
- prefill is host/launch-bound rather than dominated by long GPU kernels;
- batch metadata can be represented by fixed-shape, stable-address buffers;
- padding to the selected CUDA graph bucket is modest;
- radix-tracking scatter and other fixed-capacity work cost less than the host
  breaks they replace;
- correctness is validated with cache tracking active, not only with radix
  cache disabled.

For the measured SM89 Qwen3.5 configuration, the profitable default is to
capture GDN bodies through the 512-token bucket and keep the established eager
GDN breaks above it when radix tracking is enabled.

## Performance model

The expected prefill change can be approximated as:

```text
delta TTFT =
    - removed_breaks * host_cost_per_break
    + graph_segment_launch_cost
    + fixed_bucket_padding_cost
    + stable_metadata_update_cost
    + radix_tracking_scatter_cost
```

Capture is useful when the removed host work is larger than the added device
and fixed-capacity work. Counting kernels alone is insufficient: the first
FlashInfer GDN fusion experiment reduced kernel count without materially
improving throughput because the dominant cost was the repeated graph break and
host preparation around each layer.

## What the profiler proves

### Break topology

Qwen3.5-0.8B alternates three GDN layers and one full-attention layer six times.
The baseline 385-token trace contains 25 `cudaGraphLaunch` calls. This is the
model-level symptom of a break around each layer body. The final trace contains
7 launches: one segment on each side of the six full-attention breaks.

```text
baseline: [GDN] break [GDN] break [GDN] break [full attention] ... = 25 segments
final:    [GDN GDN GDN] break [full attention] ...               = 7 segments
```

The 72% graph-launch reduction is exactly the intended topology change. It is
not an incidental kernel optimization.

### Host-side effect

The final trace reduces direct kernel launches from 255 to 90 and CPU operations
from 2,678 to 1,071. `aten::empty` falls from 201 to 53, `aten::index_put_` from
42 to 6, and `aten::arange` from 12 to 6. All 18 eager
`ChunkGatedDeltaRuleFunction` CPU records disappear into captured segments.

The language-model prefill host annotation falls from 42.712 ms to 5.355 ms.
Profiler timings include instrumentation overhead, so the serving benchmark is
the source of truth for user-visible latency, but this span and the call counts
show where the TTFT improvement originates.

### Device-side tradeoff

GPU kernel count only falls from 693 to 654, while total measured GPU kernel
duration increases from 7.307 ms to 14.368 ms. The captured 385-token request
replays a 512-token graph, and every GDN layer includes fixed-capacity tracking
scatters. The graph removes host gaps but does not eliminate GDN arithmetic.

This explains both sides of the result:

- short and medium prefill benefits because host breaks dominate;
- long prefill can regress because padding and tracking dominate.

The device-side increase also explains why a policy based only on “fewer graph
breaks” is unsafe.

## Observed operating regions

| Region | SM89 result | Recommended policy |
| --- | --- | --- |
| 32-token input | TTFT -24.9% to -60.5%; C32 is directional | Capture |
| 256-token input | TTFT -5.0% to -16.8% | Capture |
| 385-token request in 512 bucket | 25 to 7 graph launches; exact output tokens | Capture, but monitor padding ratio |
| 1024-token bucket with radix tracking captured | Repeated C1 TTFT approximately +5.5% | Do not capture with current tracking kernels |
| 1024-token bucket with guarded fallback | TTFT -3.2% to +0.4% | Preserve eager GDN breaks |
| 1024-token input with radix cache disabled | C1 approximately neutral in the exploratory run | Capture can be allowed, then benchmark |

The large 32/C32 improvement should not be used alone for capacity planning:
its baseline point was anomalously high and 48 prompts are fewer than the
recommended steady-state sample for concurrency 32.

## Conditions that increase the benefit

### Many removable linear-attention breaks

The benefit scales with the number of GDN breaks removed. A hybrid model with
long consecutive GDN runs and infrequent full attention is a strong candidate.
A model with a full-attention break after every GDN layer has fewer kernels per
captured segment and less host work to amortize.

### Small kernels and low batch occupancy

Ada runs the small GDN and metadata kernels quickly enough that Python, tensor
construction, dispatcher work, and launch latency are visible. Lower
concurrency and shorter prompts often expose this host-bound region most
directly. At high concurrency, scheduler batching and queueing increasingly
dominate TTFT.

### Exact or close bucket fit

CUDA graph replay executes the captured bucket shape. An input close to 256 in
the 256 bucket wastes little work. A 385-token input in a 512 bucket executes
approximately 33% more token capacity than the logical request before
considering request padding. Bucket selection should therefore be included in
any production rollout analysis.

### Stable batch preparation

The optimization requires all per-request values to be copied into stable
buffers before replay. If an implementation allocates `empty`, constructs
offsets, converts dtypes, or performs advanced indexing inside every eager
layer break, BCG has high potential. If the same work is already fused or
performed once per batch, the incremental benefit is smaller.

## Conditions that reduce or reverse the benefit

### Long token buckets with active radix tracking

Tracking adds convolution and SSM scatters in every GDN layer. Their cost grows
with fixed request capacity and state size. On the tested 4090, the 1024-token
bucket crosses the break-even point. The 512 guard is therefore enabled only
when the Mamba extra buffer/radix tracking path is active.

### Mostly full-attention models

This change does not capture dynamic full-attention bodies. A model with few or
no GDN layers cannot remove enough breaks to justify the added metadata and
capture complexity.

### Decode-bound workloads

The implementation change is a prefill optimization. Long generated outputs,
slow sampling/logits processing, or decode graph breaks dominate TPOT and
end-to-end latency. The isolated Qwen3.5 profiler disables decode CUDA graphs,
so near-zero TPOT movement is expected there. Qwen3.8 acceptance separately
enables full and breakable decode graphs, with and without native MTP, to prove
that the prefill change composes with those paths.

### Large or irregular request batches

Fixed-capacity masks process padded rows. Highly variable batches can select a
large graph bucket for little live work, increasing wasted device execution.
Chunked prefill can also split one request across buckets, so production traces
should measure the actual bucket distribution rather than only prompt length.

### Architectures other than SM89

The break-even threshold is hardware-specific. Faster GPUs can make host launch
overhead more important, but different graph-launch costs, state bandwidth,
kernel implementations, and CPU performance can move the threshold in either
direction. Do not copy the 512 value to Hopper, Blackwell, ROCm, or multi-GPU
configurations without measurement.

## Qwen3.8 TP and MTP analysis

### What was validated

The official Qwen3.8-27B-FP8 checkpoint removes the three practical blockers
that motivated the acceptance run on this branch:

- TP2 and TP4 allocate the native FP32 hybrid Mamba cache without a dtype
  override;
- target prefill BCG captures and replays on Ada for the configured 64- and
  256-token buckets;
- native MTP captures target verify, draft decode, and draft extend under both
  full and breakable decode backends.

No launch hung, graph capture failure, allocator assertion, or request
allocation failure occurred. The upstream structured-output static-buffer path
also passes its two targeted unit tests, so `LogitsProcessorOutput` no longer
blocks decode BCG in this checkout.

### Why TP2 MTP wins and TP4 MTP loses

For an average accepted speculative length `A`, MTP is beneficial only when:

```text
target_decode_cost * A
    > draft_cost(A) + target_verify_cost(A) + extra_TP_communication(A)
```

TP2/C1 accepts 2.79 tokens per speculative iteration. It nearly doubles output
throughput and reduces TPOT by 56.9%, showing that avoided target decode steps
comfortably pay for draft and verification. Its TTFT almost doubles because a
single short request pays MTP's first draft/verify preparation before there is
enough decode work to amortize it. MTP should therefore be selected for
generation-heavy TP2 traffic, not latency-critical requests with very short
outputs.

TP4/C4 accepts slightly more, 2.90 tokens, yet output throughput falls 19.3%
and TPOT rises 24.0%. Acceptance length alone is insufficient: the one-layer
MTP runner and multi-token target verification add four-way collectives on a
consumer 4090 PCIe topology. At concurrency four, normal target decode already
uses the GPUs efficiently, so speculative work and synchronization displace
useful decode rather than filling host launch gaps.

This also explains why increasing TP is not automatically a latency
optimization. TP4/C4 delivers much greater aggregate output throughput than
TP2/C1, but the configurations have different concurrency and cannot be read as
a per-request scaling comparison.

### Full versus breakable decode graphs

All full and breakable variants are correct and replay successfully. Their
single-run performance crosses over:

| MTP workload | Breakable vs full TPOT | Interpretation |
| --- | ---: | --- |
| TP2 / C1 | -9.3% | Single-run advantage; cause not isolated |
| TP4 / C4 | +10.0% | Segment/control overhead is amplified by TP communication |

The difference is directional because each cell is one short serving run. It
does establish that capture success is not a sufficient enablement rule. Use
breakable decode when a profiler shows meaningful eager host gaps inside the
MTP target/draft path; use full graphs when the whole path is capturable and
communication or graph replay dominates.

### Why breakable graphs can help decode

Breakable CUDA graphs are phase-agnostic infrastructure, even though prefill is
their most common use. Ordinary single-token decode has fixed shapes and stable
metadata, so a full CUDA graph is normally simpler and faster. Breakable decode
becomes relevant when the complete step cannot be captured but large regions
still can be, for example:

- MTP target verification processes multiple candidate tokens and alternates
  target verify, draft decode, and draft extend paths;
- logits processing returns structured outputs that must be flattened into
  stable buffers and reconstructed after replay;
- hybrid attention or cache metadata introduces genuinely dynamic boundaries.

BCG retains eager execution only at those boundaries and replays the stable
regions between them. This is a clear advantage over an all-eager fallback when
full capture is impossible. It is not normally an advantage over a valid full
graph, which has fewer launches. The TP2/C1 9.3% lead over full is only one
short run and can reflect variance or fixed-capacity/static-buffer work in the
full path; a decode trace and alternating repeats are required before assigning
a cause. TP4/C4 is 10.0% worse, consistent with segment and communication
overhead. Breakable decode should therefore be selected for capture coverage,
not presumed faster than full decode.

It is also separate from this branch's optimization. The code change admits
GDN bodies during extend/prefill; it does not optimize ordinary decode. The
earlier MTP full-versus-breakable measurements ran with the same optimized
prefill code on both sides, so their TPOT difference must not be attributed to
the GDN prefill patch.

### Controlled prefill A/B interpretation

The Qwen3.8 A/B disables decode graphs and MTP and generates one token. Compared
with the exact upstream base, the optimized branch improves every admitted
64/256/512 bucket on TP2 and TP4. The measured ranges are:

- TP2: TTFT -4.31% to -19.51%, input throughput +4.49% to +24.22%;
- TP4: TTFT -3.35% to -5.41%, input throughput +3.35% to +5.62%.

Qwen3.8 has 48 GDN and 16 full-attention layers. Before admission, each GDN body
creates its own eager boundary. For an exact captured bucket, the intended
topology changes from approximately 65 graph segments to the 16 unavoidable
full-attention boundaries plus the final segment, approximately 17. The
Qwen3.5 profiler directly measured the analogous 25-to-7 change; the Qwen3.8
serving A/B confirms that the same mechanism improves user-visible TTFT.

TP2 at 256 tokens is the best point because host gaps are a large fraction of
the step and bucket padding is zero. The smaller 512-token gain shows the
device-side cost of fixed-capacity radix tracking beginning to offset removed
host work. TP4 gains less because each layer also pays four-way communication,
which the prefill patch does not remove.

At 1024 tokens, the SM89 active-radix guard rejects GDN-body admission. TP2 is
within 1% of baseline and TP4 within 2.1% in the single run, which is the
expected near-baseline region. Do not use those small deltas to claim a captured
GDN improvement; repeat measurements and a graph-launch trace would be needed
to separate run variance from incidental eager-path changes.

### Prefill graph bucket coverage under concurrent traffic

The target 256-token requests hit prefill BCG when scheduled singly. Under
TP4/C4, the scheduler sometimes combines two prompts into 512 new tokens. Since
the acceptance launch intentionally configures only 64 and 256 buckets, those
combined batches log `cuda graph: False`, while 254/256-token chunks log
`cuda graph: True`. This is a bucket-coverage issue rather than failed capture.

Production configuration should derive prefill graph buckets from scheduler
batch-token histograms. Adding a 512 bucket can improve coverage, but it must be
benchmarked with active radix tracking because the earlier profiler shows
fixed-capacity tracking and padding eventually reverse the benefit. A graph
bucket list that matches individual prompt lengths but not scheduled batch
shapes leaves performance on the table.

### Ada FP8 and communication caveats

SGLang reports missing RTX 4090-specific W8A8 block-FP8 configuration files for
the Qwen3.8 matrix shapes and falls back to defaults. This affects absolute
throughput and may move the full/BCG crossover, but not the functional capture
result. Custom all-reduce is disabled because its P2P path is not a suitable
default for this four-4090 setup; NCCL remains part of the measured TP cost.

### Qwen3.8 enablement policy

Use the following policy on a similar SM89 host:

1. Enable prefill BCG for measured small/medium buckets with active radix
   tracking; include scheduled batch shapes, not only prompt lengths. On the
   validated long-prefill TP4 profile, set the threshold to 4096 and use graph
   buckets 1024/2048/4096 with a 4096-token chunk size.
2. Enable native MTP for TP2 generation-heavy traffic when average acceptance
   remains near 2.8 or higher and the output is long enough to amortize TTFT.
3. Leave MTP off by default for TP4/C4 on PCIe-connected 4090s until a longer
   steady-state benchmark or communication optimization shows positive TPOT.
4. Prefer full decode when it captures correctly. Select breakable when full
   capture is unavailable, or only after alternating repeats and a decode trace
   demonstrate a stable workload-specific advantage.
5. Keep FP32 Mamba state unless a separate accuracy study authorizes a dtype
   override. The earlier TP2/TP4 acceptance cache sizes fit without one; the
   separate 16K/C32 capacity experiment does not.

### Why the 4096 threshold helps 1K through 16K

The threshold controls a scheduled graph bucket, not an end-to-end prompt. The
4096-token chunk policy maps the requested lengths as follows:

| Prompt length | Captured work |
| ---: | --- |
| 1K | one 1024-token replay |
| 2K | one 2048-token replay |
| 4K | one 4096-token replay |
| 8K | two 4096-token replays |
| 16K | four 4096-token replays |

Qwen3.8 has 48 GDN layers and 16 full-attention layers. For every replayed
chunk, admitting the GDN bodies removes approximately 48 eager boundaries and
leaves the 16 genuine attention boundaries plus the final segment. The earlier
Qwen3.5 profiler measured the analogous topology directly: graph launches
dropped 25 to 7, direct launches 255 to 90, CPU operations 2,678 to 1,071,
`aten::empty` 201 to 53, and `aten::index_put_` 42 to 6. The Qwen3.8 serving
matrix does not include a second profiler trace, so its 65-to-17 topology is an
inference from model structure and the same capture path, supported by
`cuda graph: True` replay logs rather than claimed as a measured launch count.

The user-visible pattern matches that mechanism:

- 2K is the cleanest row: all five cells improve TTFT by 10.3% to 14.6% and
  input throughput by 7.5% to 17.1%; host breaks are large enough to matter and
  fixed-capacity work is still moderate.
- 8K and 16K remain profitable because each additional 4K chunk repeats the
  host-break saving. Their row-average TTFT gains are 9.3% and 11.0%, and input
  throughput gains are 9.4% and 12.8%.
- 1K/C32 is the weakest repeated point at TTFT -2.8% and throughput +2.5%.
  Saturated scheduling and four-way TP communication reduce the relative host
  fraction, so this point defines the measured crossover margin.
- The three primary-grid anomalies disappeared when prompt counts increased.
  This demonstrates why a single wave, especially four prompts at C4, cannot
  be used to tune the threshold.

Capturing 16K directly is the wrong response. A separate full-16K capture probe
consumed about 2.12 GB of graph memory, while the 4K-chunk policy needs only
0.56 GB and already covers 8K/16K requests. The recommended rule is therefore
`capture threshold = maximum scheduled chunk bucket`, not `maximum prompt
length`.

### Capacity, topology, and numerical boundary

The full 16K/C32 grid needs 524,288 input-token slots. TP4 with 160 BF16 Mamba
state entries provides 543,634 KV-token capacity and completed all 32 requests
without retraction or allocator failure. FCFS progressive chunking does not put
all 32 requests into the first scheduler batch: the queue drains while the
running set grows, even though the client and benchmark peak concurrency are
32. Results should be read as serving-level C32, not a single 524K-token GPU
kernel.

TP2 cannot fit the same capacity target on 24-GiB cards. Halving TP doubles the
per-rank weight and KV burden; weights, 160 hybrid states, and 524K KV tokens
already exceed the device before graph workspace. Comparing a reduced TP2 grid
against the full TP4 grid would answer a different question, so it was not
reported as equivalent evidence.

BF16 Mamba state is also a numerical boundary. It enabled C32 capacity but the
one-token random-prompt comparison matched 9/16 at 1K and 8/8 at 4K. The
optimized replay itself was deterministic, and a fixed prompt kept the same
first token with a 0.0173 selected-logprob difference, but later greedy tokens
diverged. This is not evidence of state corruption, yet it is insufficient for
exact-parity acceptance. Production policy remains FP32 unless task-level evals
explicitly approve BF16 state.

### Threshold rollout rule

For an SM89/Qwen3.8 deployment matching this experiment:

1. set `chunked_prefill_size=max_prefill_tokens=4096`;
2. capture prefill buckets 1024, 2048, and 4096;
3. set `gdn_bcg_tracking_capture_max_tokens=4096`;
4. require alternating or expanded repeats to improve both TTFT and input
   throughput by at least 2% at the shortest/highest-concurrency boundary;
5. fall back to the default 512 threshold if the model, GPU, radix policy, TP
   size, or scheduler chunk distribution changes.

This policy provides measured positive results at every requested cell after
expanded repeats. It is not a mathematical guarantee for other hardware or
traffic distributions; the runtime knob exists precisely because the
fixed-capacity tracking crossover is workload dependent.

## Production enablement checklist

1. Confirm the model layer topology and count removable GDN breaks.
2. Run an active-radix-tracking request and verify the server logs
   `cuda graph: True`.
3. Compare output token IDs and text against the established baseline.
4. Profile one uncached prefill near each bucket boundary.
5. Require graph-launch count to approach `full_attention_layers + 1`.
6. Record direct kernel launches, CPU operation count, GPU kernel time, and the
   prefill host annotation together.
7. Benchmark TTFT at representative input lengths and concurrency, including
   enough prompts for steady state.
8. Keep a fallback for buckets where GPU padding/tracking cost exceeds saved
   host work.
9. Evaluate TPOT separately with the intended decode backend.

## Profiler interpretation guide

| Signal | Interpretation |
| --- | --- |
| `cudaGraphLaunch` remains near layer count | GDN bodies are still breaking per layer |
| `cudaGraphLaunch` becomes zero on active tracking | The entire prefill likely fell back to eager execution |
| Graph launches fall but TTFT does not | Padding, tracking, or another scheduler/logits path dominates |
| Direct launches and `aten::empty/index_put_` fall | Host preparation has moved into capture or one-time batch preparation |
| GPU time rises while host annotation falls | Capture is trading device work for removed host gaps; benchmark the break-even point |
| TPOT changes materially with decode graphs disabled | Treat it as batching/scheduler variance or an unintended cross-phase effect |

When using `/start_profile`, flush the radix cache before starting the profiler,
then start profiling and submit the target request without a multi-second gap.
The scheduler records idle-loop annotations; waiting after profiler start can
inflate a one-step trace from less than 1 MB to hundreds of MB and make gzip
export appear stalled.

## Next optimization priorities

1. Fuse radix-tracking convolution and SSM updates with adjacent GDN output
   handling or batch them across layers.
2. Learn a per-architecture capture threshold from bucket-level benchmark data
   instead of hard-coding a universal value.
3. Add RTX 4090 W8A8 block-FP8 tuning files for the Qwen3.8 TP2/TP4 matrix
   shapes, then repeat the full-versus-breakable crossover measurement.
4. Profile TP4 MTP collectives and target verification to determine whether
   communication fusion, smaller speculative steps, or a lower draft-token
   count restores positive TPOT.
5. Add confidence intervals and at least five prompts per concurrency slot for
   saturation tests.
6. Add a compact Qwen3.8 native-MTP capture/replay regression test suitable for
   continuous integration.
