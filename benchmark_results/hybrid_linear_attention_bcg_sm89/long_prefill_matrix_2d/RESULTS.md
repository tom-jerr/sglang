# Qwen3.8 TP4 long-prefill BCG results

## Result

On four RTX 4090 GPUs, capturing consecutive GDN bodies instead of breaking at
every attention layer improved input throughput in **25/25** controlled cells.
The geometric-mean gain was **+7.52%** (median **+5.83%**, range
**+1.65% to +28.32%**). Mean TTFT improved in **24/25** cells, with a
geometric-mean reduction of **7.72%** (median **6.07%**, range
**-22.10% to +3.47%**).

The controlled comparison uses the same two-dimensional graph buckets on both
sides. The only functional switch is
`gdn_bcg_tracking_capture_max_tokens=0` versus `4096`, so the main table
isolates GDN body capture rather than mixing it with graph-key policy changes.

### Aggregate by input length

| Input length | Mean TTFT change | Input throughput change |
|---:|---:|---:|
| 1K | -8.34% | +9.60% |
| 2K | -13.53% | +13.02% |
| 4K | -6.60% | +6.20% |
| 8K | -5.13% | +4.44% |
| 16K | -4.73% | +4.58% |

The fixed host-side saving is most visible at 1K-2K. It is amortized by GPU
compute as the request grows, but remains positive through 16K.

### Aggregate by client concurrency

| Concurrency | Mean TTFT change | Input throughput change |
|---:|---:|---:|
| 1 | -9.42% | +10.38% |
| 4 | -6.59% | +8.40% |
| 8 | -7.10% | +4.86% |
| 16 | -8.77% | +7.29% |
| 32 | -6.70% | +6.76% |

The only TTFT regression was 1K/C4 (+3.47%), while its aggregate input
throughput still improved by 16.71%. With four samples and one output token,
mean TTFT at this cell is sensitive to request arrival and scheduler batching.

## Full controlled matrix

`A` is per-layer break with two-dimensional buckets. `C` is consecutive GDN
body capture with the identical buckets. TTFT is in milliseconds; throughput
is input tokens per second.

| Input | Conc. | A TTFT | C TTFT | TTFT delta | A tok/s | C tok/s | Tok/s delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 1 | 1001.24 | 899.21 | -10.19% | 1019.46 | 1134.04 | +11.24% |
| 1024 | 4 | 4500.71 | 4656.82 | +3.47% | 747.68 | 872.64 | +16.71% |
| 1024 | 8 | 5022.63 | 4617.82 | -8.06% | 1143.13 | 1184.61 | +3.63% |
| 1024 | 16 | 9004.94 | 7691.27 | -14.59% | 1187.42 | 1315.14 | +10.76% |
| 1024 | 32 | 14723.34 | 13053.97 | -11.34% | 1235.19 | 1311.17 | +6.15% |
| 2048 | 1 | 2161.32 | 1683.70 | -22.10% | 946.56 | 1214.67 | +28.32% |
| 2048 | 4 | 4808.50 | 4155.36 | -13.58% | 1212.81 | 1344.60 | +10.87% |
| 2048 | 8 | 8125.09 | 7335.20 | -9.72% | 1249.20 | 1354.29 | +8.41% |
| 2048 | 16 | 15032.76 | 13406.81 | -10.82% | 1256.56 | 1362.38 | +8.42% |
| 2048 | 32 | 26925.33 | 24011.28 | -10.82% | 1236.20 | 1363.44 | +10.29% |
| 4096 | 1 | 3393.41 | 3275.41 | -3.48% | 1206.04 | 1249.56 | +3.61% |
| 4096 | 4 | 8798.20 | 7594.84 | -13.68% | 1223.75 | 1342.36 | +9.69% |
| 4096 | 8 | 14699.11 | 13614.14 | -7.38% | 1299.18 | 1351.97 | +4.06% |
| 4096 | 16 | 27497.14 | 25625.48 | -6.81% | 1283.54 | 1358.43 | +5.83% |
| 4096 | 32 | 47387.23 | 46821.42 | -1.19% | 1261.83 | 1362.03 | +7.94% |
| 8192 | 1 | 6456.31 | 6140.51 | -4.89% | 1268.31 | 1333.57 | +5.14% |
| 8192 | 4 | 16017.35 | 15171.01 | -5.28% | 1299.35 | 1347.93 | +3.74% |
| 8192 | 8 | 28918.68 | 27253.72 | -5.76% | 1298.75 | 1351.64 | +4.07% |
| 8192 | 16 | 54508.96 | 51675.25 | -5.20% | 1276.39 | 1347.53 | +5.57% |
| 8192 | 32 | 104951.56 | 100226.64 | -4.50% | 1300.28 | 1348.51 | +3.71% |
| 16384 | 1 | 12989.18 | 12328.84 | -5.08% | 1261.08 | 1328.63 | +5.36% |
| 16384 | 4 | 31251.67 | 30393.49 | -2.75% | 1324.28 | 1346.18 | +1.65% |
| 16384 | 8 | 57162.06 | 54602.80 | -4.48% | 1296.24 | 1350.62 | +4.20% |
| 16384 | 16 | 109844.04 | 103175.78 | -6.07% | 1274.72 | 1350.61 | +5.95% |
| 16384 | 32 | 211290.95 | 200193.90 | -5.25% | 1276.28 | 1350.37 | +5.80% |

## Profiler attribution

The profiler comparison uses exactly one 1024-token request and one graph
replay on TP rank 0. Both sides use the same two-dimensional bucket. GPU events
are selected by CUDA correlation IDs issued from the replay window; durations
are unioned to avoid double counting overlapping events.

| TP0 replay metric | Per-layer break | GDN body capture | Change |
|---|---:|---:|---:|
| Host replay wall time | 138.333 ms | 12.912 ms | -90.7% |
| Breakable replay segments | 64 | 16 | -75.0% |
| `cudaGraphLaunch` | 65 | 17 | -73.8% |
| Eager `cudaLaunchKernel` | 443 | 48 | -89.2% |
| `cudaLaunchKernelExC` | 16 | 16 | unchanged |
| `aten::empty` | 404 | 16 | -96.0% |
| `aten::index_put_` | 96 | 0 | -100% |
| `aten::_to_copy` | 54 | 0 | -100% |
| CUDA runtime API time | 28.372 ms | 5.633 ms | -80.1% |
| Correlated GPU busy union | 952.621 ms | 869.745 ms | -8.7% |
| Correlated GPU span | 960.964 ms | 870.483 ms | -9.4% |
| Gaps inside GPU span | 8.343 ms | 0.737 ms | -91.2% |

Qwen3.8-27B has 64 language layers: 48 linear-attention layers and 16
full-attention layers. The segment count falling from 64 to 16 is therefore the
direct signature of the intended behavior: GDN bodies stay captured, while
the truly dynamic full-attention layers remain break boundaries.

The important result is not merely fewer Triton kernels. GDN kernels still
execute, but their launch topology and metadata preparation move inside graph
replay. The largest changes are host replay time, eager launches, allocation
operators, dtype copies, index updates, and the gaps between GPU work.

## What was optimized

1. **Backend capability and safe fallback.** Linear-attention backends opt in
   through `can_capture_attention_body`; full attention and unsupported live
   batches continue to break/eager safely.
2. **Stable-address GDN metadata.** State/conv indices, cumulative sequence
   lengths, FLA chunk indices and offsets, tracking buffers, and workspaces are
   captured at fixed addresses and refreshed before replay.
3. **Fixed-capacity chunk topology.** A bounded `(sequence, chunk)` plan pads
   unused rows with a zero-length dummy sequence, so live request layouts can
   change without changing captured addresses or launch topology.
4. **Preparation hoisting.** Allocation, `index_put_`, dtype conversion, chunk
   offset generation, and state tracking are removed from every GDN-layer
   break and consolidated in batch preparation or captured kernels.
5. **Two-dimensional graph keys.** Prefill graphs are keyed by packed-token
   capacity and request capacity. The tested sparse table contains only
   `(1024|2048|4096, 1|4)` because `max_prefill_tokens=4096` makes larger live
   request counts unreachable for this workload.

The bucketing design, selection rule, DSpark compatibility, and fallback
semantics are documented in `../BUCKETING_DESIGN.md`.

## When to enable it

This path is useful when all of the following hold:

- the model mixes many GDN/linear-attention layers with fewer full-attention
  layers;
- prefill uses the breakable CUDA graph backend;
- the active packed-token shape is admitted by a captured bucket and is at or
  below `gdn_bcg_tracking_capture_max_tokens`;
- request topology fits the captured request-capacity bucket; and
- metadata/workspace memory for the selected sparse bucket table is acceptable.

The relative gain is largest when host launch/preparation is a material part of
prefill (short-to-medium chunks, many GDN layers, or scheduler pressure). It is
smaller but still positive for 8K-16K here because GPU compute amortizes the
fixed host saving. It should remain disabled or fall back when topology is not
capture-safe, when no shape covers the live batch, or when extra capture memory
outweighs the expected request mix.

## Methodology and caveats

- Model: `/workspace/models/Qwen3.8-27B-FP8`.
- Hardware: 4 x RTX 4090, TP4, Ada SM89.
- Decode graph: disabled; output length: one token, so this is a prefill/TTFT
  test and TPOT is intentionally not reported.
- Inputs: `random-ids`, exact lengths, `random_range_ratio=1`, seed 20260824.
- Requests: four for concurrency 1/4, otherwise equal to concurrency.
- Each cell flushes radix cache and runs one warmup request.
- Scheduler: `chunked_prefill_size=max_prefill_tokens=4096`, maximum 32 running
  requests, context length 32768.

An initial chronological A/B/C sweep used legacy one-dimensional buckets for A
and B, then two-dimensional buckets for C. It showed a nearly constant 27%
cross-run slowdown in C that contradicted the profiler. Live clocks showed no
thermal throttle, and later repetitions demonstrated substantial cross-process
variance. Those raw results are retained in `per_layer_break`, `body_1d`, and
`body_2d`, but they are not used as a direct three-way performance conclusion.

The main comparison was therefore rerun with identical two-dimensional buckets
in `per_layer_break_2d` and `body_2d`. A reverse-order C-after-A repeat at
concurrency 1 again won at every length, by +20.4% to +37.9% throughput, while
also running 7.4% to 22.6% faster than the first C pass. This confirms the
direction but exposes large run-to-run variance; the full-matrix +7.52% is used
as the conservative headline rather than the larger repeat result.

Raw JSON is committed. The 39 MB compressed profiler traces remain locally
under `profiles/` and are intentionally ignored by Git. Run `./summarize.py`
to validate token counts and regenerate the controlled table.
