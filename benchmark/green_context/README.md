# Single-GPU green-context benchmark

This benchmark compares normal SGLang serving, P/D multiplexing on ordinary
CUDA streams, and P/D multiplexing backed by CUDA Runtime API green contexts.
The ordinary-stream PDMux control separates the scheduler's split-prefill
effect from SM spatial isolation. The benchmark targets the case where long
prefills interfere with latency-sensitive decode work on one GPU.

Requirements:

- NVIDIA driver and CUDA Runtime 13.1 or newer
- `cuda-python` exposing `cuda.bindings.runtime`
- A CUDA GPU that supports green contexts
- Qwen3-8B-AWQ (the runner defaults to `/workspace/models/Qwen3-8B-AWQ`)

Run the full three-variant matrix:

```bash
benchmark/green_context/run_qwen3_8b_single_gpu.sh
```

The default matrix covers 128, 4K, 8K, and 16K-token prompts at client
concurrency 1, 2, 4, 8, 16, and 32. It also runs an interference suite: short
requests enter decode first, then a long prefill is injected. JSON results and
server logs are written under `benchmark_results/green_context/`.
When both variants complete, `comparison.md` contains matched p95/p99 latency
tables and percentage improvements.

Useful overrides:

```bash
MODEL_PATH=/models/Qwen3-8B-AWQ \
OUTPUT_DIR=/tmp/green-context-results \
EXTRA_BENCH_ARGS="--suite interference --interference-repeats 5" \
benchmark/green_context/run_qwen3_8b_single_gpu.sh
```

The runner disables CUDA graphs in both variants. This isolates green-context
SM partitioning and avoids multiplying graph memory across PDMux stream groups.
It defaults to `MEM_FRACTION_STATIC=0.78`, leaving enough temporary activation
workspace for 16K prefills, and uses one fixed 80/48-SM P/D partition. Keeping
one partition also avoids reusing PyTorch allocator events across different
CUDA execution contexts while an adaptive split changes thresholds.
The baseline otherwise uses the same scheduling, attention, quantization, and
KV-cache settings as the green-context variant.
