#!/usr/bin/env bash
#SBATCH --job-name=gemma4-e2b-bc-vlm
#SBATCH --account=bgjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=1
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

export UV_CACHE_DIR=/work/hdd/bgjs/mnakamura/.cache/uv
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export ROBOCASA_VALIDATE_IMAGE_PATHS="${ROBOCASA_VALIDATE_IMAGE_PATHS:-false}"

python_bin="${PYTHON_BIN:-/work/hdd/bgjs/mnakamura/robocasa/.venv/bin/python}"
model_path="${MODEL_PATH:-google/gemma-4-E2B}"
staged_dataset_root="${STAGED_DATASET_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf/robocasa_20260430T030150Z_full_run_selected}"
train_tasks="${TRAIN_TASKS:-all}"
val_tasks="${VAL_TASKS:-all}"
train_example_granularity="${TRAIN_EXAMPLE_GRANULARITY:-decentralized}"
trust_example_cache="${TRUST_EXAMPLE_CACHE:-false}"
validation_trajectories_per_task="${VAL_TRAJECTORIES_PER_TASK:-0}"
validation_trajectory_fraction="${VAL_TRAJECTORY_FRACTION:-0.1}"
validation_split_seed="${VAL_SPLIT_SEED:-42}"
max_length=4096
max_images_per_sample="${MAX_IMAGES_PER_SAMPLE:-4}"
max_history_steps_per_prompt="${MAX_HISTORY_STEPS_PER_PROMPT:-8}"
supervise_last_assistant_turn_only="${SUPERVISE_LAST_ASSISTANT_TURN_ONLY:-true}"
output_dir="${OUTPUT_DIR:-training/bc_task_vlm/runs/gemma4-e2b-20260430-five-task-bc-task-vlm}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-4}"
grad_accum="${GRAD_ACCUM:-2}"
learning_rate="${LEARNING_RATE:-2e-4}"
warmup_ratio="${WARMUP_RATIO:-0.03}"
num_workers="${NUM_WORKERS:-8}"
logging_steps="${LOGGING_STEPS:-10}"
save_steps="${SAVE_STEPS:-500}"
save_total_limit="${SAVE_TOTAL_LIMIT:-8}"
eval_steps="${EVAL_STEPS:-500}"
wandb_mode="${WANDB_MODE:-online}"
wandb_project="${WANDB_PROJECT:-robocasa-bc-task-vlm}"
wandb_run_name="${WANDB_RUN_NAME:-gemma4-e2b-20260430-five-task-sft}"
hf_dataset_repos=(
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_veggie_dip_prep"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_tong_buffet_setup"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_spicy_marinade"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_sweeten_coffee"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_setup_wine_glasses"
)

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is not runnable: ${python_bin}" >&2
  echo "Set PYTHON_BIN to a working training environment with accelerate installed." >&2
  exit 1
fi

export PATH="$(dirname "${python_bin}"):${PATH}"

if [[ "${SKIP_HF_STAGING:-true}" != "true" ]]; then
  stage_args=(
    --output-root "${staged_dataset_root}"
    --split "${HF_DATASET_SPLIT:-train}"
    --workers "${HF_STAGE_WORKERS:-32}"
  )
  if [[ -n "${HF_STAGE_MAX_IN_FLIGHT:-}" ]]; then
    stage_args+=(--max-in-flight "${HF_STAGE_MAX_IN_FLIGHT}")
  fi
  if [[ "${HF_STAGE_RESUME:-false}" == "true" ]]; then
    stage_args+=(--resume --resume-validation "${HF_STAGE_RESUME_VALIDATION:-validated}")
  fi
  if [[ -n "${HF_DATASET_REVISION:-}" ]]; then
    stage_args+=(--revision "${HF_DATASET_REVISION}")
  fi
  for repo_id in "${hf_dataset_repos[@]}"; do
    stage_args+=(--repo-id "${repo_id}")
  done

  "${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
else
  if [[ ! -f "${staged_dataset_root}/hf_stage_manifest.json" ]]; then
    echo "Missing staged HF dataset manifest: ${staged_dataset_root}/hf_stage_manifest.json" >&2
    echo "Stage the dataset on the CPU cluster first:" >&2
    echo "  sbatch training/scripts/stage_hf_sweep_datasets_cpu.sh" >&2
    echo "Or set SKIP_HF_STAGING=false to stage inside this job." >&2
    exit 1
  fi
  echo "Using pre-staged HF dataset at ${staged_dataset_root}"
fi

launch_args=(
  --dataset-root "${staged_dataset_root}" \
  --model-name-or-path "${model_path}" \
  --processor-name-or-path "${model_path}" \
  --train-tasks "${train_tasks}" \
  --val-tasks "${val_tasks}" \
  --validation-trajectories-per-task "${validation_trajectories_per_task}" \
  --validation-trajectory-fraction "${validation_trajectory_fraction}" \
  --validation-split-seed "${validation_split_seed}" \
  --train-example-granularity "${train_example_granularity}" \
  --output-dir "${output_dir}" \
  --per-device-batch-size "${per_device_batch_size}" \
  --grad-accum "${grad_accum}" \
  --num-epochs 1 \
  --learning-rate "${learning_rate}" \
  --max-length "${max_length}" \
  --max-images-per-sample "${max_images_per_sample}" \
  --max-history-steps-per-prompt "${max_history_steps_per_prompt}" \
  --num-workers "${num_workers}" \
  --bf16 \
  --attn-implementation sdpa \
  --report-to wandb \
  --wandb-mode "${wandb_mode}" \
  --wandb-project "${wandb_project}" \
  --wandb-run-name "${wandb_run_name}" \
  --wandb-tags bc_task_vlm,gemma4,e2b,sft,robocasa \
  --eval-generation-max-samples 256 \
  --logging-steps "${logging_steps}" \
  --eval-steps "${eval_steps}" \
  --save-steps "${save_steps}" \
  --save-total-limit "${save_total_limit}" \
  --warmup-ratio "${warmup_ratio}"
)

if [[ "${supervise_last_assistant_turn_only}" == "true" ]]; then
  launch_args+=(--supervise-last-assistant-turn-only)
fi

if [[ "${trust_example_cache}" == "true" ]]; then
  launch_args+=(--trust-example-cache)
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
  launch_args+=(--max-steps "${MAX_STEPS}")
fi

echo "Launching Gemma BC VLM training:"
echo "  output_dir=${output_dir}"
echo "  train_tasks=${train_tasks}"
echo "  val_tasks=${val_tasks}"
echo "  validation_trajectories_per_task=${validation_trajectories_per_task}"
echo "  validation_trajectory_fraction=${validation_trajectory_fraction}"
echo "  save_steps=${save_steps}"
echo "  eval_steps=${eval_steps}"

training/scripts/launch_multigpu.sh "${launch_args[@]}"
