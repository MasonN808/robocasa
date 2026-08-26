#!/usr/bin/env bash
#SBATCH --job-name=fixed-livesim
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
# Open-weight evaluation defaults to the shared-server path. Callers using a
# hosted or non-model policy must still set BACKEND=gemini/oracle explicitly.
BACKEND=${BACKEND:-vllm}
: "${COHORT_SPLIT:?Set COHORT_SPLIT to train_task_types or heldout_task_types}"
: "${RUN_NAME:?Set a unique RUN_NAME}"

PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
MANIFEST=${MANIFEST:-training/bc_task_vlm/eval_manifests/fixed_live_sim_v1/frozen.json}
DATASET_ROOT=${DATASET_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3}
COHORT_MODE=${COHORT_MODE:-sampled}
EPISODES_PER_TASK=${EPISODES_PER_TASK:-10}
EPISODES_PER_CONFIG=${EPISODES_PER_CONFIG:-1}
MAX_CONFIGS_PER_TASK=${MAX_CONFIGS_PER_TASK:-}
EVALUATION_SEED=${EVALUATION_SEED:-20260817}
SCENE_COMPATIBILITY_CACHE=${SCENE_COMPATIBILITY_CACHE:-}
SCENE_SAMPLING_SEED=${SCENE_SAMPLING_SEED:-20260819}
WORKERS=${WORKERS:-1}
COMMUNICATION_MODE=${COMMUNICATION_MODE:-full}
MODEL=${MODEL:-Qwen/Qwen3-VL-8B-Instruct}
ADAPTER=${ADAPTER:-}
OUTPUT_DIR=${OUTPUT_DIR:-training/bc_task_vlm/eval_runs/fixed_live_sim/${RUN_NAME}/${COHORT_SPLIT}}
EXTRA_ARGS=${EXTRA_ARGS:-}
TASKS=${TASKS:-}
ENABLE_THINKING=${ENABLE_THINKING:-false}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-256}
VLLM_TEMPERATURE=${VLLM_TEMPERATURE:-0.0}
VLLM_TOP_P=${VLLM_TOP_P:-}
VLLM_TOP_K=${VLLM_TOP_K:-}

export HF_HOME=/work/umass/shlomo_umass/hf_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

args=(
  --backend "${BACKEND}"
  --manifest "${MANIFEST}"
  --cohort-split "${COHORT_SPLIT}"
  --cohort-mode "${COHORT_MODE}"
  --episodes-per-task "${EPISODES_PER_TASK}"
  --episodes-per-config "${EPISODES_PER_CONFIG}"
  --evaluation-seed "${EVALUATION_SEED}"
  --dataset-root "${DATASET_ROOT}"
  --output-dir "${OUTPUT_DIR}"
  --partial-history
  --partial-step-index-mode none
  --partial-observation-mode consume-once
  --train-get-image
  --success-criterion fsm
  --uniform-durations
  --no-forced-json
  --map-renderer raster
  --map-dpi 60
  --gl-backend egl
  --no-record-firsts
  --max-new-tokens "${MAX_NEW_TOKENS}"
  --resume
)
if [[ "${BACKEND}" == "vllm" ]]; then
  args+=(--vllm-temperature "${VLLM_TEMPERATURE}")
  if [[ -n "${VLLM_TOP_P}" ]]; then
    args+=(--vllm-top-p "${VLLM_TOP_P}")
  fi
  if [[ -n "${VLLM_TOP_K}" ]]; then
    args+=(--vllm-top-k "${VLLM_TOP_K}")
  fi
fi
if [[ "${ENABLE_THINKING}" == "true" ]]; then
  args+=(--enable-thinking)
fi
# Preserve byte-for-byte command identity for pre-ablation `full` runs so
# strict resume can reuse their existing parallel outputs. `full` is also the
# Python default; only non-default experimental arms need an explicit flag.
if [[ "${COMMUNICATION_MODE}" != "full" ]]; then
  args+=(--communication-mode "${COMMUNICATION_MODE}")
fi
if [[ -n "${MAX_CONFIGS_PER_TASK}" ]]; then
  args+=(--max-configs-per-task "${MAX_CONFIGS_PER_TASK}")
fi
if [[ -n "${SCENE_COMPATIBILITY_CACHE}" ]]; then
  args+=(
    --scene-compatibility-cache "${SCENE_COMPATIBILITY_CACHE}"
    --scene-sampling-seed "${SCENE_SAMPLING_SEED}"
  )
fi
if [[ -n "${TASKS}" ]]; then
  args+=(--tasks "${TASKS}")
fi
if [[ "${BACKEND}" == "hf" ]]; then
  args+=(--model-name-or-path "${MODEL}")
elif [[ "${BACKEND}" == "oracle" ]]; then
  # Expert files are serialized for storage, but their tick_rows are atomic.
  # Replaying flattened steps would split the two-agent opening handshake.
  args+=(--concurrent-expert-replay)
elif [[ "${BACKEND}" == "vllm" ]]; then
  : "${VLLM_MODEL:?Set VLLM_MODEL for the vllm backend}"
  args+=(--vllm-model "${VLLM_MODEL}" --vllm-base-url "${VLLM_BASE_URL:-http://127.0.0.1:8000/v1}")
elif [[ "${BACKEND}" == "gemini" ]]; then
  key_file=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
  [[ -f "${key_file}" ]] || { echo "Missing ${key_file}" >&2; exit 1; }
  set -a
  # shellcheck disable=SC1090
  . "${key_file}"
  set +a
  args+=(--model "${MODEL}")
fi
if [[ -n "${ADAPTER}" ]]; then
  args+=(--adapter-path "${ADAPTER}")
fi
# shellcheck disable=SC2206
extra=( ${EXTRA_ARGS} )
args+=("${extra[@]}")

if [[ "${WORKERS}" -gt 1 ]]; then
  exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_parallel_eval \
    --workers "${WORKERS}" --python "${PYTHON_BIN}" -- "${args[@]}"
fi
exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval "${args[@]}"
