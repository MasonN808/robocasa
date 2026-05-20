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
export TOKENIZERS_PARALLELISM=false

python_bin="${PYTHON_BIN:-/work/hdd/bgjs/mnakamura/robocasa/.venv/bin/python}"
model_path="${MODEL_PATH:-google/gemma-4-E2B}"
staged_dataset_root="${STAGED_DATASET_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf/robocasa_20260430T030150Z_full_run_selected}"
train_tasks="${TRAIN_TASKS:-veggie_dip_prep,tong_buffet_setup,spicy_marinade,sweeten_coffee,setup_wine_glasses}"
val_tasks="${VAL_TASKS:-}"
max_length=4096
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

training/scripts/launch_multigpu.sh \
  --dataset-root "${staged_dataset_root}" \
  --model-name-or-path "${model_path}" \
  --processor-name-or-path "${model_path}" \
  --train-tasks "${train_tasks}" \
  --val-tasks "${val_tasks}" \
  --train-example-granularity decentralized \
  --output-dir training/bc_task_vlm/runs/gemma4-e2b-20260430-five-task-bc-task-vlm \
  --per-device-batch-size 4 \
  --grad-accum 2 \
  --num-epochs 1 \
  --learning-rate 2e-4 \
  --max-length "${max_length}" \
  --num-workers 8 \
  --bf16 \
  --attn-implementation sdpa \
  --report-to wandb \
  --wandb-mode online \
  --wandb-project robocasa-bc-task-vlm \
  --wandb-run-name gemma4-e2b-20260430-five-task-sft \
  --wandb-tags bc_task_vlm,gemma4,e2b,sft,robocasa \
  --eval-generation-max-samples 256 \
  --eval-steps 500 \
  --save-steps 500
