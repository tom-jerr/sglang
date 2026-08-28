#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
MODEL_PATH=${MODEL_PATH:-/workspace/models/Qwen3-8B-AWQ}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/benchmark_results/green_context"}
PORT=${PORT:-30000}
VARIANTS=${VARIANTS:-"baseline pdmux_streams green_context"}
SERVER_TIMEOUT=${SERVER_TIMEOUT:-900}
GPU_IDLE_TIMEOUT=${GPU_IDLE_TIMEOUT:-900}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.78}
EXTRA_BENCH_ARGS=${EXTRA_BENCH_ARGS:-}
SERVER_PID=""

mkdir -p "${OUTPUT_DIR}"
export PYTHONPATH="${REPO_ROOT}/python${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

wait_for_server() {
  local deadline=$((SECONDS + SERVER_TIMEOUT))
  until curl --fail --silent "http://127.0.0.1:${PORT}/health" >/dev/null; do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      wait "${SERVER_PID}" || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Server did not become healthy within ${SERVER_TIMEOUT}s" >&2
      return 1
    fi
    sleep 2
  done
}

wait_for_idle_gpu() {
  local deadline=$((SECONDS + GPU_IDLE_TIMEOUT))
  while nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; do
    if (( SECONDS >= deadline )); then
      echo "GPU still has another compute process after ${GPU_IDLE_TIMEOUT}s" >&2
      nvidia-smi --query-compute-apps=pid,process_name,used_memory \
        --format=csv,noheader >&2
      return 1
    fi
    sleep 2
  done
}

run_variant() {
  local variant=$1
  local log_file="${OUTPUT_DIR}/server_${variant}.log"
  local -a green_args=()
  wait_for_idle_gpu
  if [[ "${variant}" == "green_context" ]]; then
    green_args=(
      --enable-pdmux
      --pdmux-config-path "${REPO_ROOT}/benchmark/green_context/pdmux_runtime.yaml"
    )
  elif [[ "${variant}" == "pdmux_streams" ]]; then
    green_args=(
      --enable-pdmux
      --pdmux-config-path "${REPO_ROOT}/benchmark/green_context/pdmux_torch_streams.yaml"
    )
  fi

  python -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --dtype float16 \
    --quantization awq_marlin \
    --kv-cache-dtype fp8_e5m2 \
    --attention-backend triton \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests 64 \
    --chunked-prefill-size -1 \
    --disable-overlap-schedule \
    --cuda-graph-backend-decode disabled \
    --cuda-graph-backend-prefill disabled \
    "${green_args[@]}" \
    >"${log_file}" 2>&1 &
  SERVER_PID=$!
  wait_for_server

  # EXTRA_BENCH_ARGS deliberately supports a plain whitespace-separated list.
  # shellcheck disable=SC2086
  python "${REPO_ROOT}/benchmark/green_context/bench_low_latency.py" \
    --url "http://127.0.0.1:${PORT}" \
    --variant "${variant}" \
    --output "${OUTPUT_DIR}/${variant}.json" \
    ${EXTRA_BENCH_ARGS}

  kill "${SERVER_PID}" 2>/dev/null || true
  wait "${SERVER_PID}" 2>/dev/null || true
  SERVER_PID=""
}

for variant in ${VARIANTS}; do
  run_variant "${variant}"
done

if [[ -f "${OUTPUT_DIR}/baseline.json" && -f "${OUTPUT_DIR}/green_context.json" ]]; then
  analysis_args=()
  if [[ -f "${OUTPUT_DIR}/pdmux_streams.json" ]]; then
    analysis_args=(--pdmux-streams "${OUTPUT_DIR}/pdmux_streams.json")
  fi
  python "${REPO_ROOT}/benchmark/green_context/analyze_results.py" \
    --baseline "${OUTPUT_DIR}/baseline.json" \
    --green-context "${OUTPUT_DIR}/green_context.json" \
    --output "${OUTPUT_DIR}/comparison.md" \
    "${analysis_args[@]}"
fi
