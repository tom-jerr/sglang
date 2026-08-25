# Qwen3.8 TP4 long-prefill BCG matrix

This directory contains the raw serving-benchmark output for the
Qwen3.8-27B-FP8 threshold experiment run on four RTX 4090 GPUs.

## Compared configurations

The two servers are identical except for:

- `tp4_baseline`: `--gdn-bcg-tracking-capture-max-tokens 0`
- `tp4_optimized`: `--gdn-bcg-tracking-capture-max-tokens 4096`

Common arguments:

```bash
sglang serve \
  --model-path /workspace/models/Qwen3.8-27B-FP8 \
  --host 127.0.0.1 --port 30000 \
  --tp-size 4 --mem-fraction-static 0.82 \
  --context-length 32768 --max-running-requests 32 \
  --max-mamba-cache-size 160 --mamba-ssm-dtype bfloat16 \
  --chunked-prefill-size 4096 --max-prefill-tokens 4096 \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-bs-prefill 1024 2048 4096 \
  --cuda-graph-backend-decode disabled \
  --disable-custom-all-reduce --trust-remote-code
```

Each primary cell uses exact random input length, one generated token,
temperature zero, seed `20260824`, a cache flush, and one warmup. C1 and C4 use
four prompts; C8/C16/C32 use one prompt per concurrency slot. Files are named
`i<tokens>_c<concurrency>.json`.

The `repeat` files expand the three primary-grid boundary observations:

- 1K/C4: 16 prompts;
- 1K/C32: 64 prompts;
- 4K/C8: 16 prompts.

The `correctness_*` files use seed `20260825`, one-token greedy output,
`--return-logprob`, and `--output-details`. They compare 16 random 1K prompts
and eight random 4K prompts. The benchmark output retains generated text but
does not serialize token logprobs in this version.

## Interpretation constraints

- Input throughput is total input tokens divided by benchmark wall time; it is
  a serving metric, not raw kernel throughput.
- One output token makes E2E latency effectively equal to TTFT and intentionally
  excludes decode/TPOT optimization claims.
- BF16 Mamba state is a capacity-only override. These files do not replace the
  FP32 correctness acceptance recorded in the parent result document.
- Client concurrency 32 and benchmark peak concurrency 32 do not imply that
  FCFS progressive chunking schedules all requests into one GPU batch.

See `../RESULTS.md` for the complete table and `../ANALYSIS.md` for the threshold
and profiler interpretation.
