#!/usr/bin/env bash
set -euo pipefail

export SGLANG_SPEC_MIXED_BATCH_INVARIANT=1
export SGLANG_PROFILE_WITH_STACK=false
export SGLANG_PROFILE_RECORD_SHAPES=true

mixed_chunk_args=()
if [[ "${SGLANG_ENABLE_MIXED_CHUNK:-1}" == "1" ]]; then
  mixed_chunk_args+=(--enable-mixed-chunk)
fi

exec python -m sglang.launch_server \
  --model-path /workspace/models/Qwen3-4B \
  --speculative-algorithm EAGLE3 \
  --speculative-draft-model-path /workspace/models/Qwen3-4B_eagle3 \
  --speculative-num-steps 3 \
  --speculative-eagle-topk 1 \
  --speculative-num-draft-tokens 4 \
  --attention-backend triton \
  --speculative-draft-attention-backend triton \
  --chunked-prefill-size 512 \
  "${mixed_chunk_args[@]}" \
  --cuda-graph-backend-prefill breakable \
  --cuda-graph-max-bs-prefill 64 \
  --cuda-graph-max-bs-decode 24 \
  --max-running-requests 24 \
  --mem-fraction-static 0.8 \
  --host 127.0.0.1 \
  --port 30000
