#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v1_5-structured-random-sft
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:b200:4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# v1.5 (reasoning probe, --train-reasoning) SFT, trained fresh on the
# structured_random data-generation source instead of verbalized_sampling,
# for a within-source v1-vs-v1.5 comparison mirroring the existing
# verbalized_sampling one. Same recipe/parallelism as
# run_v1_structured_random_full.sbatch, plus --train-reasoning. DATA_ROOT is
# the 21-traj/task capped structured_random training root (matching v1's
# actual training scale) produced by stage_structured_random.sbatch.
#   sbatch --dependency=afterok:<stage_job_id> run_v1_5_structured_random_full.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_train_21
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# --- 4-GPU DDP, v1-matching effective batch (2 x 4 x 1 = 8, same as v1's 2x1x4) ---
export NUM_PROCESSES=4
export PER_DEVICE_BATCH_SIZE=2
export GRAD_ACCUM=1
export ATTN_IMPLEMENTATION=sdpa
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

# --- run identity ---
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v1_5-structured-random
export WANDB_RUN_NAME=qwen3vl-8b-v1_5-structured-random

# --- schedule (v1's exact recipe: 3 epochs, v1 eval/save cadence) ---
export NUM_EPOCHS=3
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export EVAL_GENERATION_MAX_SAMPLES=0  # in-training generation eval hangs on structured_random (rank-2-specific example); real eval runs separately post-training

# --- assemble args (mirrors train_qwen3vl_8b.sh's v1 recipe, minus its
# single-process launch -- launch_multigpu.sh needs the raw args array) ---
source training/scripts/local/_launch_args.sh
launch_args+=(
  --train-reasoning
  --num-epochs "${NUM_EPOCHS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --no-gradient-checkpointing
  # Per-run cache dir -- the default (.cache/bc_task_vlm/examples) is repo-
  # global and races when multiple training jobs run concurrently (this
  # caused a DDP "inconsistent 0 params on one rank" crash).
  --training-samples-cache-dir ".cache/bc_task_vlm/examples_${WANDB_RUN_NAME}"
)

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
