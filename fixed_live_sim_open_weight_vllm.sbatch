#!/usr/bin/env bash
#SBATCH --job-name=fixed-vllm
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail

: "${COHORT_SPLIT:?Set COHORT_SPLIT to train_task_types or heldout_task_types}"
: "${RUN_NAME:?Set a unique RUN_NAME}"

MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ADAPTER=${ADAPTER:-}
WORKERS=${WORKERS:-4}
COMMUNICATION_MODE=${COMMUNICATION_MODE:-full}
VLLM_IMAGE=${VLLM_IMAGE:-/work/umass/shlomo_umass/dbenhamougol_umass/containers/vllm-openai-v0.27.1.sif}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-16384}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.70}
VLLM_MAX_LORA_RANK=${VLLM_MAX_LORA_RANK:-64}
ENABLE_TOWER_CONNECTOR_LORA=${ENABLE_TOWER_CONNECTOR_LORA:-false}
VLLM_TOOL_CALL_PARSER=${VLLM_TOOL_CALL_PARSER:-hermes}
VLLM_REASONING_PARSER=${VLLM_REASONING_PARSER:-}
VLLM_READY_TIMEOUT_S=${VLLM_READY_TIMEOUT_S:-600}
PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

[[ -f "${VLLM_IMAGE}" ]] || { echo "Missing vLLM image: ${VLLM_IMAGE}" >&2; exit 1; }
if [[ -n "${ADAPTER}" && ! -e "${ADAPTER}" ]]; then
  echo "Adapter does not exist: ${ADAPTER}" >&2
  exit 1
fi

# Separate jobs may share a node, so derive a stable per-job port.
VLLM_PORT=${VLLM_PORT:-$((18000 + ${SLURM_JOB_ID:-0} % 10000))}
VLLM_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
if [[ -n "${ADAPTER}" ]]; then
  SERVED_MODEL=${SERVED_MODEL:-robocasa-sft}
else
  SERVED_MODEL=${SERVED_MODEL:-qwen3vl8b-base}
fi

mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs
server_log=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/${SLURM_JOB_NAME:-fixed-vllm}-${SLURM_JOB_ID:-local}-server.log
server=(
  apptainer exec --nv
  --bind /work:/work
  "${VLLM_IMAGE}"
  vllm serve "${MODEL}"
  --host 127.0.0.1
  --port "${VLLM_PORT}"
  --dtype bfloat16
  --max-model-len "${VLLM_MAX_MODEL_LEN}"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}"
  --enable-prefix-caching
  --limit-mm-per-prompt '{"image":3}'
  --mm-processor-kwargs '{"min_pixels":262144,"max_pixels":262144}'
  --mm-processor-cache-gb 2
  --enable-auto-tool-choice
  --tool-call-parser "${VLLM_TOOL_CALL_PARSER}"
  --served-model-name "${SERVED_MODEL}"
)
if [[ -n "${VLLM_REASONING_PARSER}" ]]; then
  server+=(--reasoning-parser "${VLLM_REASONING_PARSER}")
fi
if [[ -n "${ADAPTER}" ]]; then
  server+=(
    --enable-lora
    --max-lora-rank "${VLLM_MAX_LORA_RANK}"
    --lora-modules "${SERVED_MODEL}=${ADAPTER}"
  )
  if [[ "${ENABLE_TOWER_CONNECTOR_LORA}" == "true" ]]; then
    server+=(--enable-tower-connector-lora)
  fi
fi

echo "Starting vLLM model=${MODEL} served_model=${SERVED_MODEL} adapter=${ADAPTER:-none} port=${VLLM_PORT}"
VLLM_USE_FLASHINFER_SAMPLER=0 HF_HOME=/work/umass/shlomo_umass/hf_cache \
  "${server[@]}" >"${server_log}" 2>&1 &
server_pid=$!
cleanup() {
  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + VLLM_READY_TIMEOUT_S))
until "${PYTHON_BIN}" - "${VLLM_BASE_URL}/models" <<'PY'
import json, sys, urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    payload = json.load(response)
assert payload.get("data"), payload
PY
do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "vLLM server exited before readiness; tail follows" >&2
    tail -100 "${server_log}" >&2 || true
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "vLLM server did not become ready within ${VLLM_READY_TIMEOUT_S}s" >&2
    tail -100 "${server_log}" >&2 || true
    exit 1
  fi
  sleep 5
done

echo "vLLM ready; launching ${WORKERS} live-sim clients"
if [[ -n "${DAGGER_LOOP_ARGS:-}" ]]; then
  key_file=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
  [[ -f "${key_file}" ]] || { echo "Missing ${key_file}" >&2; exit 1; }
  set -a
  # shellcheck disable=SC1090
  . "${key_file}"
  set +a
  # shellcheck disable=SC2206
  dagger_loop_args=( ${DAGGER_LOOP_ARGS} )
  VLLM_MODEL="${SERVED_MODEL}" VLLM_BASE_URL="${VLLM_BASE_URL}" \
    exec "${PYTHON_BIN}" -m training.bc_task_vlm.dagger.loop "${dagger_loop_args[@]}"
fi
if [[ -n "${DAGGER_COLLECT_ARGS:-}" ]]; then
  key_file=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
  [[ -f "${key_file}" ]] || { echo "Missing ${key_file}" >&2; exit 1; }
  set -a
  # shellcheck disable=SC1090
  . "${key_file}"
  set +a
  # shellcheck disable=SC2206
  dagger_collect_args=( ${DAGGER_COLLECT_ARGS} )
  VLLM_MODEL="${SERVED_MODEL}" VLLM_BASE_URL="${VLLM_BASE_URL}" \
    exec "${PYTHON_BIN}" -m training.bc_task_vlm.dagger.collect "${dagger_collect_args[@]}"
fi
set +e
BACKEND=vllm \
VLLM_MODEL="${SERVED_MODEL}" \
VLLM_BASE_URL="${VLLM_BASE_URL}" \
WORKERS="${WORKERS}" \
COMMUNICATION_MODE="${COMMUNICATION_MODE}" \
bash fixed_live_sim_eval.sbatch
status=$?
set -e
exit "${status}"
