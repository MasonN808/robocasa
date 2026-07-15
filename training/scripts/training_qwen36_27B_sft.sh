#!/usr/bin/env bash
#SBATCH --job-name=qwen36-27b-47task-sft
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dbenhamougol@umass.edu

# Qwen3.6-27B LoRA SFT on the 47 train tasks of the SFT-necessity experiment
# (held-out tasks and the seed-42 10% trajectory holdout stay unseen).
# Smoke test first:  MAX_STEPS=20 sbatch --time=01:00:00 training/scripts/training_qwen36_27B_sft.sh

set -euo pipefail

mkdir -p slurm_logs

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
export UV_CACHE_DIR=/work/hdd/bgjs/dbenhamougoldfajn/.uv_cache
export HF_HOME="${HF_HOME:-/work/hdd/bgjs/.cache/huggingface}"
mkdir -p "${HF_HOME}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export ROBOCASA_VALIDATE_IMAGE_PATHS="${ROBOCASA_VALIDATE_IMAGE_PATHS:-false}"
export PYTHON_BIN="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${repo_root}/.env"
  set +a
fi

model_path="${MODEL_PATH:-Qwen/Qwen3.6-27B}"

# Assemble the 52-task dataset root on node-local /tmp: tasks already
# extracted on /work are symlinked; tar-staged tasks are untarred locally so
# nothing extracted lands on the inode-limited /work quota. The path is
# STABLE (no job id) because the example cache is keyed on the dataset-root
# string; tar preserves mtimes, so caches built on one node validate on any
# other. Pre-clean to avoid trusting a partial extraction from a killed job.
assemble_target="${ASSEMBLE_TARGET:-/tmp/experiment_52_sft}"
if [[ -z "${EXPERIMENT_DATASET_ROOT:-}" ]]; then
  rm -rf "${assemble_target}"
  trap 'rm -rf "${assemble_target}"' EXIT
fi
dataset_root="${EXPERIMENT_DATASET_ROOT:-$(bash "${repo_root}/training/scripts/assemble_dataset_root.sh" "${assemble_target}")}"
selection="${HELD_OUT_SELECTION:-${repo_root}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json}"
output_dir="${OUTPUT_DIR:-training/bc_task_vlm/runs/qwen36-27b-47task-sft}"
max_steps="${MAX_STEPS:-12000}"

train_tasks="$("${PYTHON_BIN}" - <<EOF
import json
print(",".join(json.load(open("${selection}"))["train_tasks"]))
EOF
)"

launch_args=(
  --dataset-root "${dataset_root}"
  --model-name-or-path "${model_path}"
  --processor-name-or-path "${model_path}"
  --sft-format tool_call
  --image-resolution "${IMAGE_RESOLUTION:-512}"
  --train-tasks "${train_tasks}"
  --val-tasks "${train_tasks}"
  --validation-trajectories-per-task 0
  --validation-trajectory-fraction "${VAL_TRAJECTORY_FRACTION:-0.1}"
  --validation-split-seed "${VAL_SPLIT_SEED:-42}"
  --output-dir "${output_dir}"
  --per-device-batch-size "${PER_DEVICE_BATCH_SIZE:-1}"
  --grad-accum "${GRAD_ACCUM:-2}"
  --num-epochs 1
  --max-steps "${max_steps}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --max-length "${MAX_LENGTH:-16384}"
  --max-images-per-sample "${MAX_IMAGES_PER_SAMPLE:-4}"
  --num-workers "${NUM_WORKERS:-8}"
  --bf16
  --gradient-checkpointing
  --attn-implementation sdpa
  --supervise-last-assistant-turn-only
  --report-to wandb
  --wandb-mode "${WANDB_MODE:-online}"
  --wandb-project "${WANDB_PROJECT:-robocasa-bc-task-vlm}"
  --wandb-run-name "${WANDB_RUN_NAME:-qwen36-27b-47task-sft}"
  --wandb-tags bc_task_vlm,qwen3.6,27b,sft,sft_necessity
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES:-128}"
  --logging-steps "${LOGGING_STEPS:-10}"
  --eval-steps "${EVAL_STEPS:-1000}"
  --save-steps "${SAVE_STEPS:-1000}"
  --save-total-limit "${SAVE_TOTAL_LIMIT:-3}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
)

if [[ -n "${RESUME_FROM_CHECKPOINT:-}" ]]; then
  launch_args+=(--resume-from-checkpoint "${RESUME_FROM_CHECKPOINT}")
fi
if [[ "${TRUST_EXAMPLE_CACHE:-false}" == "true" ]]; then
  launch_args+=(--trust-example-cache)
fi

echo "Launching Qwen3.6-27B SFT:"
echo "  dataset_root=${dataset_root}"
echo "  output_dir=${output_dir}"
echo "  max_steps=${max_steps}"
echo "  train_tasks(count)=$(tr ',' '\n' <<<"${train_tasks}" | wc -l)"

training/scripts/launch_multigpu.sh "${launch_args[@]}"
