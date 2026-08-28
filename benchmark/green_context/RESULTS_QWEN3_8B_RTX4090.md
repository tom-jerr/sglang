# Qwen3-8B single-GPU green-context results

## Conclusion

CUDA green contexts help a narrow, latency-sensitive P/D multiplexing case,
but this SGLang integration is not ready to enable by default.

- With a 4K background prefill, green contexts reduced short-decode ITL p99
  by a median 70.4% versus the same PDMux scheduler on ordinary CUDA streams.
- With an 8K background prefill, ITL p95 improved by a median 79.3%. ITL p99
  improved 75.0-78.5% at foreground concurrency 1-8, but only 10.4% and 8.6%
  at concurrency 16 and 32 because an execution-context transition spike
  dominated the extreme tail.
- With a 16K background prefill, ITL p95 improved about 34% at concurrency
  1-8, but ITL p99 was 149-156% worse. At concurrency 16-32, both p95 and p99
  were worse than ordinary-stream PDMux.
- The background prefill TTFT was about 1.73-1.74x the ordinary-stream control
  because the fixed partition gave prefill 80 of the RTX 4090's 128 SMs.
- In homogeneous 4K/8K/16K traffic, median output throughput was 0.69x, 0.64x,
  and 0.62x the ordinary-stream control. Green contexts are therefore a
  latency-isolation tradeoff, not a general throughput optimization.

## Test setup

- GPU: NVIDIA GeForce RTX 4090, 24 GiB, 128 SMs (compute capability 8.9)
- Driver: 595.80
- PyTorch: 2.13.0+cu130; CUDA Runtime reported by cuda-python: 13.3
- Model: `Qwen/Qwen3-8B-AWQ`, FP16 compute, AWQ Marlin, FP8 E5M2 KV cache
- Source base: upstream SGLang commit `74df026877a490720739749c825b8de3a8423dd5`
- Static memory fraction: 0.78, yielding about 178K KV tokens and 4.9 GiB
  free activation workspace
- CUDA graphs and overlap scheduling disabled in all variants
- Fixed P/D partition: 80 prefill SMs / 48 decode SMs

The homogeneous matrix used 128, 4K, 8K, and 16K input tokens at concurrency
1, 2, 4, 8, 16, and 32. The interference matrix started 32-token foreground
requests (64 output tokens), waited for their first token, then injected one
4K, 8K, or 16K background prefill. It covered foreground concurrency 1-32.
Input IDs had distinct first tokens to prevent radix-prefix sharing.

## Net green-context effect

This is the important control: both columns use PDMux split-prefill scheduling;
only the stream backend differs.

| Background | Foreground concurrency | Ordinary ITL p95 | Green ITL p95 | Net change | Ordinary ITL p99 | Green ITL p99 | Net change |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4K | 1-32 | 40.5-90.1 ms | 28.7-36.0 ms | 18.0-67.1% better | 125.0-159.8 ms | 38.6-43.9 ms | 68.8-72.5% better |
| 8K | 1-8 | 147.2-158.3 ms | 31.9-34.9 ms | 77.5-79.4% better | 154.0-165.8 ms | 34.5-39.3 ms | 75.0-78.5% better |
| 8K | 16-32 | 174.3-180.3 ms | 33.5-34.3 ms | 80.3-81.4% better | 182.5-183.6 ms | 163.5-167.9 ms | 8.6-10.4% better |
| 16K | 1-8 | 215.0-215.2 ms | 140.1-142.1 ms | 34.0-34.9% better | 215.7-216.7 ms | 537.4-552.0 ms | 149-156% worse |
| 16K | 16-32 | 236.1-250.4 ms | 495.9-496.4 ms | 98-110% worse | 245.3-252.1 ms | 1187.5-1235.0 ms | 371-403% worse |

Against the unsplit baseline, green contexts still improved interference ITL
p99 by 89-97% for 4K/8K and 52-79% for 16K. That comparison overstates the
hardware-partitioning benefit, because ordinary-stream PDMux already removes
most of the baseline prefill stall.

## Stability findings

Two integration problems appeared during long sweeps:

1. The upstream adaptive configuration creates several green execution-context
   pairs and changes them as decode batch size crosses thresholds. Both the
   Runtime API backend and the ordinary-stream control could hang at repeated
   16/32-concurrency transitions. A fixed 80/48 partition avoided cross-context
   switching and completed each isolated case.
2. Even with one fixed partition, a long multi-case server lifetime could hang
   after many flush/transition cycles. The final 42 matched cases per variant
   were therefore collected in segmented runs with fresh server processes.
   Most cases used two interference repeats; isolated concurrency-16/32
   stability cases used one. All reported requests succeeded, but the smaller
   high-concurrency sample makes their p99 estimates directional rather than a
   production SLO qualification.

An initial `--mem-fraction-static 0.88` run also exhausted temporary activation
workspace during 16K PDMux prefills. The runner now defaults to 0.78.

## Recommendation

Use the Runtime API wrapper as an opt-in experimental backend. For a 4090,
the fixed 80/48 split is useful when 4K-8K prefills interfere with an active
low-latency decode workload and extra prefill TTFT is acceptable. Do not enable
it for homogeneous long-prompt throughput, 16K latency-critical mixes, or a
long-lived adaptive multi-context server until SGLang removes execution-context
switching from the hot path and fixes repeated PDMux transition cleanup.
