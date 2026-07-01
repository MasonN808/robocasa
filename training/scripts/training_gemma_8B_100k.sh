#!/usr/bin/env bash
#SBATCH --job-name=gemma4-e4b-100k-bc-vlm
#SBATCH --account=bgjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --mem=128g
#SBATCH --time=24:00:00
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
model_path="${MODEL_PATH:-google/gemma-4-E4B}"
staged_dataset_root="${STAGED_DATASET_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf/robocasa_20260430T030150Z_full_100k_first49}"
train_tasks="${TRAIN_TASKS:-all}"
val_tasks="${VAL_TASKS:-all}"
use_example_cache="${USE_EXAMPLE_CACHE:-true}"
trust_example_cache="${TRUST_EXAMPLE_CACHE:-false}"
if [[ "${trust_example_cache}" == "true" ]]; then
  use_example_cache="true"
fi
validation_trajectories_per_task="${VAL_TRAJECTORIES_PER_TASK:-0}"
validation_trajectory_fraction="${VAL_TRAJECTORY_FRACTION:-0.1}"
validation_split_seed="${VAL_SPLIT_SEED:-42}"
max_length="${MAX_LENGTH:-16384}"
max_images_per_sample="${MAX_IMAGES_PER_SAMPLE:-4}"
supervise_last_assistant_turn_only="${SUPERVISE_LAST_ASSISTANT_TURN_ONLY:-true}"
output_dir="${OUTPUT_DIR:-training/bc_task_vlm/runs/gemma4-e4b-100k-first49-bc-task-vlm-2gpu-bs4-lr1e5-evalbs8-workers8-evalmax1000-structtraj100-ddp-timeout7200-ddpunusedtrue}"
per_device_batch_size="${PER_DEVICE_BATCH_SIZE:-4}"
per_device_eval_batch_size="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
grad_accum="${GRAD_ACCUM:-2}"
learning_rate="${LEARNING_RATE:-1e-5}"
warmup_ratio="${WARMUP_RATIO:-0.03}"
num_workers="${NUM_WORKERS:-8}"
example_build_workers="${EXAMPLE_BUILD_WORKERS:-8}"
ddp_timeout_seconds="${DDP_TIMEOUT_SECONDS:-7200}"
logging_steps="${LOGGING_STEPS:-10}"
save_steps="${SAVE_STEPS:-5000}"
save_total_limit="${SAVE_TOTAL_LIMIT:-8}"
eval_steps="${EVAL_STEPS:-5000}"
wandb_mode="${WANDB_MODE:-online}"
wandb_project="${WANDB_PROJECT:-robocasa-bc-task-vlm}"
wandb_run_name="${WANDB_RUN_NAME:-gemma4-e4b-100k-first49-2gpu-bs4-lr1e5-evalbs8-workers8-evalmax1000-structtraj100-ddp-timeout7200-ddpunusedtrue-sft}"
wandb_tags="${WANDB_TAGS:-bc_task_vlm,gemma4,e4b,base,sft,robocasa,100k,2gpu,bs4,lr1e-5,evalbs8,workers8,examplebuild8,example_cache,evalmax1000,structtraj100,nccl_cumem_host0,nccl_net_plugin_forced_none,ddp_timeout7200,ddp_unused_parameters_true}"
eval_max_samples="${EVAL_MAX_SAMPLES:-1000}"
eval_generation_max_samples="${EVAL_GENERATION_MAX_SAMPLES:-1000}"
eval_generation_max_trajectories="${EVAL_GENERATION_MAX_TRAJECTORIES:-100}"
eval_generation_batch_size="${EVAL_GENERATION_BATCH_SIZE:-4}"
hf_stage_max_total_episodes="${HF_STAGE_MAX_TOTAL_EPISODES:-100000}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is not runnable: ${python_bin}" >&2
  echo "Set PYTHON_BIN to a working training environment with accelerate installed." >&2
  exit 1
fi

export PATH="$(dirname "${python_bin}"):${PATH}"

if [[ "${SKIP_HF_STAGING:-true}" != "true" ]]; then
  if [[ -n "${HF_DATASET_REPOS_FILE:-}" ]]; then
    if [[ ! -f "${HF_DATASET_REPOS_FILE}" ]]; then
      echo "HF_DATASET_REPOS_FILE does not exist: ${HF_DATASET_REPOS_FILE}" >&2
      exit 1
    fi
    mapfile -t hf_dataset_repos < <(
      sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "${HF_DATASET_REPOS_FILE}"
    )
  elif [[ -n "${HF_DATASET_REPOS:-}" ]]; then
    IFS=',' read -r -a hf_dataset_repos <<< "${HF_DATASET_REPOS}"
  else
    echo "Inline staging requires HF_DATASET_REPOS_FILE or HF_DATASET_REPOS." >&2
    echo "Recommended: stage on CPU first with training/scripts/stage_hf_sweep_datasets_cpu.sh." >&2
    exit 1
  fi

  stage_args=(
    --output-root "${staged_dataset_root}"
    --split "${HF_DATASET_SPLIT:-train}"
    --load-workers "${HF_STAGE_LOAD_WORKERS:-5}"
    --workers "${HF_STAGE_WORKERS:-32}"
    --progress-interval "${HF_STAGE_PROGRESS_INTERVAL:-25}"
    --max-total-episodes "${hf_stage_max_total_episodes}"
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
    repo_id="${repo_id#"${repo_id%%[![:space:]]*}"}"
    repo_id="${repo_id%"${repo_id##*[![:space:]]}"}"
    if [[ -n "${repo_id}" ]]; then
      stage_args+=(--repo-id "${repo_id}")
    fi
  done

  "${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
else
  if [[ ! -f "${staged_dataset_root}/hf_stage_manifest.json" ]]; then
    echo "Missing staged HF dataset manifest: ${staged_dataset_root}/hf_stage_manifest.json" >&2
    echo "Stage the 100k dataset on the CPU cluster first:" >&2
    echo "  sbatch --export=ALL,HF_DATASET_REPOS_FILE=training/bc_task_vlm/hf_repo_lists/robocasa_20260430T030150Z_full_49.txt,HF_STAGE_MAX_TOTAL_EPISODES=100000,HF_STAGE_RESUME=true training/scripts/stage_hf_sweep_datasets_cpu.sh" >&2
    echo "Or set SKIP_HF_STAGING=false and provide HF_DATASET_REPOS_FILE or HF_DATASET_REPOS." >&2
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
  --output-dir "${output_dir}" \
  --per-device-batch-size "${per_device_batch_size}" \
  --per-device-eval-batch-size "${per_device_eval_batch_size}" \
  --grad-accum "${grad_accum}" \
  --num-epochs 1 \
  --learning-rate "${learning_rate}" \
  --max-length "${max_length}" \
  --max-images-per-sample "${max_images_per_sample}" \
  --num-workers "${num_workers}" \
  --example-build-workers "${example_build_workers}" \
  --ddp-timeout-seconds "${ddp_timeout_seconds}" \
  --bf16 \
  --attn-implementation sdpa \
  --report-to wandb \
  --wandb-mode "${wandb_mode}" \
  --wandb-project "${wandb_project}" \
  --wandb-run-name "${wandb_run_name}" \
  --wandb-tags "${wandb_tags}" \
  --eval-generation-max-samples "${eval_generation_max_samples}" \
  --eval-generation-batch-size "${eval_generation_batch_size}" \
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

if [[ "${use_example_cache}" != "true" ]]; then
  launch_args+=(--no-use-example-cache)
fi

if [[ -n "${eval_max_samples}" ]]; then
  launch_args+=(--eval-max-samples "${eval_max_samples}")
fi

if [[ -n "${eval_generation_max_trajectories}" ]]; then
  launch_args+=(
    --eval-generation-max-trajectories "${eval_generation_max_trajectories}"
  )
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
  launch_args+=(--max-steps "${MAX_STEPS}")
fi

echo "Launching Gemma 8B BC VLM training:"
echo "  model_path=${model_path}"
echo "  staged_dataset_root=${staged_dataset_root}"
echo "  output_dir=${output_dir}"
echo "  train_tasks=${train_tasks}"
echo "  val_tasks=${val_tasks}"
echo "  use_example_cache=${use_example_cache}"
echo "  trust_example_cache=${trust_example_cache}"
echo "  validation_trajectories_per_task=${validation_trajectories_per_task}"
echo "  validation_trajectory_fraction=${validation_trajectory_fraction}"
echo "  max_length=${max_length}"
echo "  max_images_per_sample=${max_images_per_sample}"
echo "  supervise_last_assistant_turn_only=${supervise_last_assistant_turn_only}"
echo "  per_device_batch_size=${per_device_batch_size}"
echo "  per_device_eval_batch_size=${per_device_eval_batch_size}"
echo "  grad_accum=${grad_accum}"
echo "  learning_rate=${learning_rate}"
echo "  warmup_ratio=${warmup_ratio}"
echo "  num_workers=${num_workers}"
echo "  example_build_workers=${example_build_workers}"
echo "  ddp_timeout_seconds=${ddp_timeout_seconds}"
echo "  save_steps=${save_steps}"
echo "  eval_steps=${eval_steps}"
echo "  eval_max_samples=${eval_max_samples:-<unset>}"
echo "  eval_generation_max_samples=${eval_generation_max_samples}"
echo "  eval_generation_max_trajectories=${eval_generation_max_trajectories:-<unset>}"
echo "  eval_generation_batch_size=${eval_generation_batch_size}"

training/scripts/launch_multigpu.sh "${launch_args[@]}"
