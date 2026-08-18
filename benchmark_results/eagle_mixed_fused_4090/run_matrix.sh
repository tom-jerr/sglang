#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
RESULT_DIR=${RESULT_DIR:-"${SCRIPT_DIR}/results"}
PORT=${PORT:-30000}
mkdir -p "${RESULT_DIR}"

server_pid=""
stop_server() {
  if [[ -n "${server_pid}" ]] && kill -0 "${server_pid}" 2>/dev/null; then
    kill -INT "${server_pid}"
    wait "${server_pid}" || true
  fi
  server_pid=""
}
trap stop_server EXIT

run_one() {
  local label=$1
  local mixed_chunk=$2
  local fused_attention=$3
  local log_path="${RESULT_DIR}/${label}.server.log"
  local result_path="${RESULT_DIR}/${label}.json"
  local mixed_args=()
  if [[ "${mixed_chunk}" == "on" ]]; then
    mixed_args+=(--enable-mixed-chunk)
  fi

  env \
    SGLANG_SPEC_MIXED_BATCH_INVARIANT=1 \
    SGLANG_ENABLE_SPEC_MIXED_FUSED_ATTENTION="$([[ "${fused_attention}" == "on" ]] && echo 1 || echo 0)" \
    python -m sglang.launch_server \
      --model-path /workspace/models/Qwen3-4B \
      --host 127.0.0.1 \
      --port "${PORT}" \
      --dtype bfloat16 \
      --mem-fraction-static 0.80 \
      --max-running-requests 24 \
      --chunked-prefill-size 512 \
      "${mixed_args[@]}" \
      --attention-backend triton \
      --speculative-draft-attention-backend triton \
      --cuda-graph-backend-prefill breakable \
      --cuda-graph-max-bs-prefill 64 \
      --cuda-graph-max-bs-decode 24 \
      --speculative-algorithm EAGLE3 \
      --speculative-draft-model-path /workspace/models/Qwen3-4B_eagle3 \
      --speculative-num-steps 3 \
      --speculative-eagle-topk 1 \
      --speculative-num-draft-tokens 4 \
      --speculative-attention-mode prefill \
      >"${log_path}" 2>&1 &
  server_pid=$!

  local ready=0
  for _ in $(seq 1 120); do
    if curl -fsS --max-time 2 "http://127.0.0.1:${PORT}/get_model_info" >/dev/null 2>&1; then
      ready=1
      break
    fi
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if [[ "${ready}" != "1" ]]; then
    tail -n 120 "${log_path}"
    return 1
  fi

  python "${SCRIPT_DIR}/latency_matrix.py" \
    --url "http://127.0.0.1:${PORT}" \
    --label "${label}" \
    --output "${result_path}" \
    --contexts 512 1024 2048 4096 \
    --samples-per-context 12 \
    --probe-output-len 64 \
    --running-batch-size 4 \
    --running-context 512 \
    --running-output-len 128 \
    --probe-stagger-ms 30 \
    --seed 20260818

  stop_server
}

run_one mixed_off_fused_off off off
run_one mixed_off_fused_on off on
run_one mixed_on_fused_off on off
run_one mixed_on_fused_on on on

python "${SCRIPT_DIR}/compare_matrix.py" \
  --input "mixed_off_fused_off=${RESULT_DIR}/mixed_off_fused_off.json" \
  --input "mixed_off_fused_on=${RESULT_DIR}/mixed_off_fused_on.json" \
  --input "mixed_on_fused_off=${RESULT_DIR}/mixed_on_fused_off.json" \
  --input "mixed_on_fused_on=${RESULT_DIR}/mixed_on_fused_on.json" \
  --output "${RESULT_DIR}/comparison.json"
