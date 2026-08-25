#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 OUTPUT_DIR" >&2
  exit 2
fi

output_dir=$1
mkdir -p "$output_dir"

for input_len in 1024 2048 4096 8192 16384; do
  for concurrency in 1 4 8 16 32; do
    if (( concurrency <= 4 )); then
      num_prompts=4
    else
      num_prompts=$concurrency
    fi
    output_file="$output_dir/i${input_len}_c${concurrency}.json"
    if [[ -s "$output_file" ]]; then
      echo "skip existing $output_file"
      continue
    fi
    python -m sglang.benchmark.serving \
      --backend sglang \
      --base-url http://127.0.0.1:30000 \
      --dataset-name random-ids \
      --tokenizer /workspace/models/Qwen3.8-27B-FP8 \
      --tokenize-prompt \
      --num-prompts "$num_prompts" \
      --random-input-len "$input_len" \
      --random-output-len 1 \
      --random-range-ratio 1 \
      --request-rate inf \
      --max-concurrency "$concurrency" \
      --seed 20260824 \
      --temperature 0 \
      --flush-cache \
      --warmup-requests 1 \
      --disable-tqdm \
      --output-file "$output_file"
  done
done
