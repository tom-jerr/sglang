# Hybrid linear-attention BCG experiment results

## Summary

The implementation removes per-GDN prefill graph breaks while preserving
active radix-cache tracking. Its capture threshold is now configurable: the
safe default is 512 tokens, while a profiled Qwen3.8 TP4 deployment can opt in
through 4096-token scheduler chunks. On the tested RTX 4090, the original
small-model grid improved mean TTFT by 5.0% to 60.5%; the later Qwen3.8
1K-to-16K matrix averaged TTFT -9.48% and input throughput +9.37%.

The 385-token correctness request produced exactly the same 48 output token IDs
and text as baseline. A compact trace reduced graph launches from 25 to 7.

The Qwen3.8-27B-FP8 acceptance run also passed TP2 and TP4 startup, allocation,
capture, replay, and inference with native MTP and both full and breakable
decode graphs. MTP was profitable for TP2/C1 but not TP4/C4 on the tested
four-4090 PCIe system.

## Revisions

- Upstream base after final synchronization: `230c052eb`
- Initial hybrid GDN BCG capture: `49af0bdf6`
- Active radix-tracking capture and Ada guard: `6025f6240`
- Branch: `feature/hybrid-linear-attn-bcg-sm89`

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-08-23 to 2026-08-24 |
| GPU | NVIDIA GeForce RTX 4090, SM89, 24,564 MiB |
| GPUs installed / used | 4 / 1 |
| Driver | 595.80 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| Triton | 3.7.1 |
| Model | Local Qwen3.5-0.8B, bfloat16 |
| Model topology | 24 layers: 18 linear attention, 6 full attention |

`nvcc` was not installed in the experiment environment; the CUDA version above
is the runtime bundled with PyTorch.

## Server configuration

Prefill BCG was enabled and decode CUDA graphs were disabled to isolate TTFT
effects.

```bash
CUDA_VISIBLE_DEVICES=0 python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3.5-0.8B \
  --host 127.0.0.1 --port 30000 \
  --mem-fraction-static 0.65 \
  --context-length 4096 \
  --max-running-requests 64 \
  --chunked-prefill-size 1024 \
  --max-prefill-tokens 4096 \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 32 64 128 256 512 1024 \
  --cuda-graph-backend-decode disabled \
  --trust-remote-code
```

## Serving benchmark method

Each point used 48 random prompts, 32 output tokens, an infinite submission
rate, deterministic seed `20260823`, exact random input length, and a cache
flush. Baseline and final runs used the same arguments and warmed kernel cache.

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --base-url http://127.0.0.1:30000 \
  --model /workspace/models/Qwen3.5-0.8B \
  --dataset-name random \
  --random-input-len INPUT_TOKENS \
  --random-output-len 32 \
  --random-range-ratio 1.0 \
  --num-prompts 48 \
  --request-rate inf \
  --max-concurrency CONCURRENCY \
  --seed 20260823 \
  --flush-cache \
  --disable-tqdm
```

## TTFT and TPOT

A negative delta is an improvement.

| Input | Concurrency | Baseline TTFT (ms) | Final TTFT (ms) | TTFT delta | Baseline TPOT (ms) | Final TPOT (ms) | TPOT delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 41.51 | 31.17 | -24.92% | 19.17 | 18.61 | -2.95% |
| 32 | 8 | 63.42 | 44.27 | -30.19% | 19.86 | 19.89 | +0.17% |
| 32 | 32 | 244.83 | 96.66 | -60.52% | 20.48 | 19.76 | -3.52% |
| 256 | 1 | 44.84 | 37.31 | -16.80% | 19.17 | 18.71 | -2.38% |
| 256 | 8 | 80.03 | 72.92 | -8.88% | 20.17 | 19.99 | -0.93% |
| 256 | 32 | 151.98 | 144.35 | -5.02% | 23.16 | 22.86 | -1.29% |
| 1024 | 1 | 47.76 | 46.31 | -3.04% | 19.25 | 18.89 | -1.92% |
| 1024 | 8 | 128.71 | 129.18 | +0.36% | 22.52 | 22.47 | -0.23% |
| 1024 | 32 | 356.87 | 345.32 | -3.24% | 29.70 | 31.44 | +5.83% |

Request-throughput deltas for the same grid were +4.57%, +2.82%, +18.39%,
+3.51%, +1.80%, +1.76%, +2.05%, -0.17%, and -1.61%, respectively.

The 32-token concurrency-32 baseline point is an outlier and uses fewer than
the recommended five prompts per concurrency slot. Treat its 60.5% TTFT result
as directional. The concurrency-1 and concurrency-8 results are more useful
for sizing the expected gain.

TPOT is expected to remain close to baseline because decode CUDA graphs were
disabled and this change only captures prefill. The +5.83% 1024/C32 TPOT result
is a saturated-scheduler regression/noise signal and prevents claiming a decode
improvement.

## Correctness

The accuracy case used a 385-token prompt, active radix-cache tracking, no cache
hit, greedy sampling, and 48 generated tokens.

| Check | Result |
| --- | --- |
| Prompt tokens | 385 |
| Cached tokens | 0 |
| Output token IDs | Exact, 48/48 |
| Output text | Exact |
| Maximum selected-logprob absolute difference | 0.018146 |
| Mean selected-logprob absolute difference | 0.000772 |
| Repeated BCG replay | Deterministic |

The logprob difference is numerical-path variation rather than a token-level
correctness failure. It is smaller than the variation observed between the
server's eager and BCG execution paths in the same environment.

## Compact profiler comparison

Both traces capture one uncached 385-token prefill. Counts are more reliable
than individual runtime-call durations because profiler instrumentation changes
synchronization and launch timing.

| Metric | Baseline | Final | Delta |
| --- | ---: | ---: | ---: |
| `cudaGraphLaunch` | 25 | 7 | -72.0% |
| Direct kernel launches | 255 | 90 | -64.7% |
| CPU operations | 2,678 | 1,071 | -60.0% |
| GPU kernels | 693 | 654 | -5.6% |
| GPU kernel duration | 7.307 ms | 14.368 ms | +96.6% |
| `sglang.vlm.language_model_prefill` host annotation | 42.712 ms | 5.355 ms | -87.5% |
| `aten::empty` calls | 201 | 53 | -73.6% |
| `aten::index_put_` calls | 42 | 6 | -85.7% |
| `aten::arange` calls | 12 | 6 | -50.0% |
| `ChunkGatedDeltaRuleFunction` eager CPU ops | 18 | 0 | -100% |

The 7 graph launches match the expected six full-attention boundaries plus the
final graph segment. The 18 GDN layers no longer create eager graph breaks.

The increased GPU duration is expected for this capture shape: a 385-token
request replays a 512-token bucket and executes fixed-capacity tracking kernels.
The measured TTFT gain therefore comes from eliminating host preparation and
launch gaps, not from reducing GDN arithmetic.

## Automated validation

- 11 GDN/BCG/radix tests passed, including 2 parameterized subtests.
- 5 Mamba scatter tests passed on CUDA, including the new mask-first GDN
  tracking case.
- Targeted Ruff import/undefined-name checks passed.
- `git diff --check` passed.

## Qwen3.8-27B-FP8 acceptance validation

### Checkpoint and topology

| Item | Value |
| --- | --- |
| Repository | `Qwen/Qwen3.8-27B-FP8` |
| Local path | `/workspace/models/Qwen3.8-27B-FP8` |
| Revision | `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| Weight files | 66 safetensors, 30,866,866,928 bytes (28.747 GiB) |
| Weight integrity | All 66 weight CRC32 entries passed |
| Architecture | 64 layers: 48 linear attention, 16 full attention |
| Native MTP | 1 layer |
| Mamba SSM dtype | FP32 |
| Quantization | FP8 blockwise, 128 x 128 |

The official manifest's CRC values for three small metadata files are stale:
the pinned remote files and local files match each other, but not the manifest
entries for `chat_template.jinja`, `generation_config.json`, and
`tokenizer_config.json`. All weights pass, all 66 safetensors open, and the
index resolves 1,606 tensors.

### Capture and allocation matrix

All tested configurations reached server-ready state and replayed their graphs
during live generation.

| TP | MTP | Decode backend | Decode graph batches | Result |
| ---: | --- | --- | --- | --- |
| 2 | Off | full | 1 | Pass: prefill BCG and decode graph replay |
| 4 | Off | full | 1, 2, 4 | Pass: prefill BCG and decode graph replay |
| 2 | Native EAGLE | breakable | 1 | Pass: target verify, draft decode, and draft extend BCG |
| 4 | Native EAGLE | breakable | 1, 2, 4 | Pass: target verify, draft decode, and draft extend BCG |
| 2 | Native EAGLE | full | 1 | Pass: target verify, draft decode, and draft extend full graphs |
| 4 | Native EAGLE | full | 1, 2, 4 | Pass: target verify, draft decode, and draft extend full graphs |

Representative allocation and capture data:

| Metric | TP2 native MTP | TP4 native MTP |
| --- | ---: | ---: |
| Target weights per rank | 14.48 GB | 7.47 GB |
| MTP weights per rank | 2.65 GB | 1.37 GB |
| Maximum Mamba cache entries | 5 | 20 |
| Mamba SSM state per rank | 0.42 GB | 0.74 GB |
| Intermediate MTP SSM state per rank | 0.56 GB | 0.70 GB |
| Target KV tokens | 41,485 | 520,251 |
| Available memory after all BCG captures | 4.87 GB | 4.21 GB |
| Prefill BCG capture, buckets 64/256 | 3.22 s | 3.15 s |
| Breakable target-verify capture | 2.64 s | 5.34 s |
| Breakable draft decode capture | 1.49 s | 1.42 s |
| Breakable draft extend capture | 1.17 s | 1.16 s |

The draft runner logs `Disable prefill CUDA graph because some layers do not
apply Standard GQA`; this applies to draft prefill. The target runner's 64/256
prefill BCG remains active, and live 256-token requests log
`cuda graph: True`. Speculative decode also logs `cuda graph: True`.

### Correctness

Four temperature-zero prompts covered factual English, arithmetic, sequence
continuation, and Chinese generation. English, arithmetic, and sequence output
IDs matched the TP2 no-MTP reference exactly across TP2/TP4 and MTP on/off. The
Chinese prompt diverged after a common 11-token prefix across TP degrees and
MTP modes, but every result was a fluent, factually correct one-sentence
description of gravity. TP2 and TP4 native-MTP produced the same Chinese token
sequence.

This is consistent with a near-tied greedy choice changing under a different
tensor-parallel reduction order. It is not evidence of a graph replay or MTP
correctness failure. The arithmetic spot check also matched exactly under MTP
with full graphs on TP2 and TP4.

### Serving performance

The workload used exact 256-token random inputs, 64 output tokens, seed
`20260824`, a cache flush, and one warmup. TP2 used 8 prompts at concurrency 1;
TP4 used 16 prompts at concurrency 4. These are acceptance measurements, not
confidence intervals.

| TP / concurrency | MTP | Decode graph | Output tok/s | Mean TTFT (ms) | Mean TPOT (ms) | Accept length |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| TP2 / C1 | Off | full | 34.31 | 102.34 | 27.91 | - |
| TP2 / C1 | On | breakable | 66.43 | 200.21 | 12.04 | 2.79 |
| TP2 / C1 | On | full | 60.94 | 209.33 | 13.28 | 2.79 |
| TP4 / C4 | Off | full | 161.28 | 438.80 | 18.13 | - |
| TP4 / C4 | On | breakable | 130.23 | 509.09 | 22.48 | 2.90 |
| TP4 / C4 | On | full | 142.88 | 464.77 | 20.44 | 2.90 |

Relative to no MTP, TP2 breakable MTP improves output throughput by 93.6% and
TPOT by 56.9%, while increasing TTFT by 95.6%. At TP4/C4, the same MTP setting
reduces output throughput by 19.3%, increases TPOT by 24.0%, and increases TTFT
by 16.0%.

Within native MTP, breakable versus full decode graphs is also workload
dependent. Breakable improves TP2 TPOT by 9.3% in this run, but regresses TP4
TPOT by 10.0%. The TP4 result indicates that BCG segment replay and host/device
coordination do not amortize when four-way tensor-parallel communication and
speculative verification already dominate. The TP2 difference is a single-run
observation, not proof that breakable is intrinsically faster than a complete
decode graph; fixed-capacity work and run variance were not isolated.

### Controlled Qwen3.8 prefill optimization A/B

This comparison isolates the branch's GDN prefill-body capture from decode and
MTP. The baseline is `upstream/main@230c052eb`; the optimized case is this
branch at `dda87912c` before adding this result section. Both use the same model,
kernel cache, server arguments, random seed, and graph buckets.

- decode CUDA graphs and MTP are disabled;
- each request generates exactly one token, making the run prefill-dominated;
- every point uses 12 exact-length random prompts at concurrency 1, one warmup,
  and a radix-cache flush;
- configured prefill BCG buckets are 64, 256, 512, and 1024 tokens;
- TP2 uses `mem_fraction_static=0.85`; TP4 uses 0.80; both reserve 20 Mamba
  cache entries and disable custom all-reduce.

A negative TTFT delta and positive input-throughput delta are improvements.

| TP | Input tokens | Baseline TTFT (ms) | Optimized TTFT (ms) | TTFT delta | Baseline input tok/s | Optimized input tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 64 | 196.78 | 180.31 | -8.37% | 323.30 | 352.60 | +9.06% |
| 2 | 256 | 356.68 | 287.10 | -19.51% | 715.22 | 888.47 | +24.22% |
| 2 | 512 | 458.58 | 438.82 | -4.31% | 1,113.50 | 1,163.47 | +4.49% |
| 2 | 1024 | 680.82 | 674.32 | -0.95% | 1,501.30 | 1,515.85 | +0.97% |
| 4 | 64 | 163.76 | 158.28 | -3.35% | 388.53 | 401.54 | +3.35% |
| 4 | 256 | 295.51 | 279.54 | -5.41% | 863.80 | 912.36 | +5.62% |
| 4 | 512 | 502.78 | 477.67 | -4.99% | 1,016.22 | 1,069.44 | +5.24% |
| 4 | 1024 | 896.80 | 878.41 | -2.05% | 1,140.39 | 1,163.95 | +2.07% |

The strongest point is TP2 at 256 tokens: removing the 48 per-GDN eager breaks
reduces mean TTFT by 69.58 ms and raises service-level input throughput by
24.22%. TP4 gains less because four-way collectives occupy more of the critical
path, leaving a smaller host-break fraction to remove.

The 512-token benefit is smaller than the TP2 256-token benefit because
fixed-capacity radix-tracking scatters and bucket work grow with token capacity.
At 1024 tokens the Ada active-tracking guard deliberately keeps the established
per-GDN break path. The remaining 0.95% to 2.05% differences are single-run
variation and possible non-capture implementation effects; they are not
evidence that 1024-token GDN bodies were admitted for capture.

`input tok/s` here is the serving benchmark's total input tokens divided by
wall-clock benchmark duration. With concurrency 1 and one output token it is a
useful prefill service metric, but it is not raw kernel FLOP throughput.

### Long-prefill threshold matrix: 1K to 16K, concurrency 1 to 32

This TP4 matrix compares the same branch with GDN tracking capture disabled
(`threshold=0`) and enabled through the scheduler chunk size
(`threshold=4096`). Both sides use BF16 Mamba state solely to fit the requested
16K/C32 capacity on four 24-GiB GPUs. Decode graphs and MTP are disabled and
each request generates one token.

Common server settings are:

```bash
--tp-size 4 --mem-fraction-static 0.82 --context-length 32768 \
--max-running-requests 32 --max-mamba-cache-size 160 \
--mamba-ssm-dtype bfloat16 \
--chunked-prefill-size 4096 --max-prefill-tokens 4096 \
--cuda-graph-backend-prefill breakable \
--cuda-graph-bs-prefill 1024 2048 4096 \
--cuda-graph-backend-decode disabled --disable-custom-all-reduce
```

The optimized graph capture took 13.1 seconds and 0.56 GB per rank. The
threshold-0 capture took 12.2 seconds and 0.47 GB. Both allocated 160 hybrid
state entries and 543,634 KV tokens. A negative TTFT delta and positive input
throughput delta are improvements.

| Input | C | Baseline TTFT (ms) | Optimized TTFT (ms) | TTFT delta | Baseline input tok/s | Optimized input tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 1 | 994.12 | 942.21 | -5.22% | 1,026.99 | 1,084.04 | +5.55% |
| 1K | 4 | 3,626.69 | 4,054.79 | +11.80% | 961.24 | 984.16 | +2.38% |
| 1K | 8 | 5,925.65 | 4,503.13 | -24.01% | 1,009.81 | 1,189.81 | +17.83% |
| 1K | 16 | 9,446.11 | 8,193.56 | -13.26% | 1,090.74 | 1,180.69 | +8.25% |
| 1K | 32 | 15,219.15 | 14,464.61 | -4.96% | 1,247.96 | 1,241.56 | -0.51% |
| 2K | 1 | 1,931.02 | 1,648.53 | -14.63% | 1,058.87 | 1,240.24 | +17.13% |
| 2K | 4 | 5,281.65 | 4,509.48 | -14.62% | 1,110.79 | 1,241.91 | +11.80% |
| 2K | 8 | 9,051.81 | 7,867.17 | -13.09% | 1,145.77 | 1,269.89 | +10.83% |
| 2K | 16 | 16,149.34 | 14,487.22 | -10.29% | 1,171.14 | 1,259.17 | +7.52% |
| 2K | 32 | 29,708.49 | 25,800.84 | -13.15% | 1,187.85 | 1,326.51 | +11.67% |
| 4K | 1 | 3,638.31 | 3,241.08 | -10.92% | 1,124.93 | 1,262.48 | +12.23% |
| 4K | 4 | 8,939.59 | 8,062.16 | -9.82% | 1,184.98 | 1,271.82 | +7.33% |
| 4K | 8 | 14,801.43 | 14,754.20 | -0.32% | 1,251.05 | 1,248.23 | -0.23% |
| 4K | 16 | 28,996.70 | 27,019.13 | -6.82% | 1,204.74 | 1,276.26 | +5.94% |
| 4K | 32 | 55,990.75 | 52,430.12 | -6.36% | 1,217.69 | 1,288.22 | +5.79% |
| 8K | 1 | 7,057.66 | 6,554.42 | -7.13% | 1,160.15 | 1,249.29 | +7.68% |
| 8K | 4 | 17,579.71 | 16,365.39 | -6.91% | 1,206.32 | 1,252.30 | +3.81% |
| 8K | 8 | 32,139.58 | 29,001.75 | -9.76% | 1,162.04 | 1,278.67 | +10.04% |
| 8K | 16 | 59,854.19 | 52,403.42 | -12.45% | 1,168.23 | 1,327.53 | +13.64% |
| 8K | 32 | 108,319.23 | 97,251.28 | -10.22% | 1,195.35 | 1,336.01 | +11.77% |
| 16K | 1 | 14,255.89 | 12,111.72 | -15.04% | 1,149.06 | 1,352.43 | +17.70% |
| 16K | 4 | 35,938.04 | 32,472.33 | -9.64% | 1,154.20 | 1,267.94 | +9.85% |
| 16K | 8 | 64,108.86 | 56,109.45 | -12.48% | 1,158.52 | 1,311.36 | +13.19% |
| 16K | 16 | 119,244.94 | 106,188.46 | -10.95% | 1,169.17 | 1,309.82 | +12.03% |
| 16K | 32 | 208,944.37 | 194,949.95 | -6.70% | 1,199.56 | 1,332.07 | +11.05% |

The unweighted average across the 25 primary cells is TTFT -9.48% and input
throughput +9.37%. TTFT improves in 24/25 cells, throughput in 23/25, and both
in 22/25. The three non-joint-positive cells were deliberately repeated with
larger samples:

| Input/C | Prompts | Baseline TTFT (ms) | Optimized TTFT (ms) | TTFT delta | Baseline input tok/s | Optimized input tok/s | Throughput delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K/C4 | 16 | 4,289.52 | 3,312.85 | -22.77% | 913.58 | 1,137.50 | +24.51% |
| 1K/C32 | 64 | 22,263.29 | 21,633.16 | -2.83% | 1,143.29 | 1,171.97 | +2.51% |
| 4K/C8 | 16 | 21,979.59 | 19,717.67 | -10.29% | 1,181.81 | 1,286.06 | +8.82% |

Thus every requested input/concurrency cell is positive after replacing the
three short-run boundary observations with its larger-sample repeat. The 1K/C32
margin is only 2.5% and should be treated as the production crossover, not as a
universal guarantee.

#### Numerical qualification of the BF16 capacity configuration

The performance matrix is not an FP32-equivalence result. With one-token greedy
outputs, the 4096-token random-prompt set matched 8/8 exactly, while the
1024-token random-prompt set matched 9/16. A fixed 1024-token replay was
deterministic within the optimized server, selected the same first token as
baseline, and differed in that token's log probability by 0.0173; a longer
greedy continuation then diverged after three common tokens.

These random-token prompts frequently have near-tied logits, but a token
mismatch is still a mismatch. Keep FP32 Mamba state for accuracy-qualified
serving, as in the earlier TP2/TP4 acceptance section. Use the BF16-state C32
numbers only as capacity/performance evidence until a model-level eval
authorizes the dtype override.

Raw JSON, including the expanded repeats and correctness samples, is stored in
`long_prefill_matrix/`.

## Limitations

- The detailed prefill profiler comparison uses Qwen3.5-0.8B on TP1; the
  Qwen3.8 run is serving-level capture/replay validation rather than a second
  Nsight/PyTorch trace comparison.
- Qwen3.8 validation uses the official FP8 checkpoint, not an AWQ checkpoint.
- Results are single-run means, not confidence intervals.
- The concurrency-32 workload has only 48 prompts and is not a full steady-state
  sample.
- The Qwen3.8 serving samples use 8 prompts for TP2 and 16 for TP4. They are
  sufficient for functional acceptance and directional comparisons, not
  capacity-planning claims.
- The controlled prefill A/B uses 12 prompts per point and one run per case;
  deltas should be repeated before setting a production SLO.
- The long-prefill primary grid is one run per cell. Three boundary cells have
  larger-sample repeats, but the grid still does not provide confidence
  intervals.
- BF16 Mamba state was required to fit the TP4 16K/C32 capacity target and did
  not achieve exact output parity on every random-token correctness prompt.
- Missing RTX 4090-specific W8A8 block-FP8 tuning files force default kernel
  configurations, so absolute throughput is not a tuned hardware maximum.
