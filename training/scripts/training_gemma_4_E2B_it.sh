#!/usr/bin/env bash
#SBATCH --job-name=gemma4-e2b-it-bc-vlm
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
model_path="${MODEL_PATH:-google/gemma-4-E2B-it}"
dataset_root=/work/hdd/bgjs/mnakamura/robocasa/data_generation/task_level/data/image/20260413T205634Z
max_length=4096

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is not runnable: ${python_bin}" >&2
  echo "Set PYTHON_BIN to a working training environment with accelerate installed." >&2
  exit 1
fi

export PATH="$(dirname "${python_bin}"):${PATH}"

training/scripts/launch_multigpu.sh \
  --dataset-root "${dataset_root}" \
  --model-name-or-path "${model_path}" \
  --processor-name-or-path "${model_path}" \
  --train-tasks hot_dog_setup,prepare_sandwich_station \
  --val-tasks prepare_coffee \
  --train-example-granularity decentralized \
  --output-dir training/bc_task_vlm/runs/gemma4-e2b-it-bc-task-vlm \
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
  --wandb-run-name gemma4-e2b-it-task-vlm-test \
  --eval-generation-max-samples 256 \
  --eval-steps 500 \
  --save-steps 500
