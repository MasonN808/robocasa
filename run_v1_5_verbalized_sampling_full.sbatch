#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v1_5-verbalized-sampling-sft
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

# SFT v1.5 (reasoning probe) ablation vs the v1 baseline, on the
# verbalized_sampling data source -- CLEAN rerun.
# The first v1_5-reasoning run (run_v1_5_reasoning_full.sbatch) trained on
# robocasa_agentsft_subset (30 traj/task, independently re-staged from HF),
# which turned out to diverge from v1's actual training data: confirmed via
# git-blob-sha1 comparison that even identically-named trajectory IDs (e.g.
# add_lemon_to_fish/traj_000000) contain different episode content, and the
# task counts don't even match (21 traj/task for v1 vs 30 for the first
# v1.5 run). This run instead points DATA_ROOT at
# robocasa_local_train_subset_21, pulled directly from
# DorianAtSchool/robocasa_local_train_subset_21 on HF -- the actual raw
# folder v1 was trained on (verbalized_sampling data-generation method) --
# so trajectory IDs and content now match v1 exactly. Everything else is
# v1's exact recipe (effective batch 8, LR 2e-4 cosine, 3 epochs, LoRA
# r=16/alpha=32, 512^2 images) plus --train-reasoning, parallelized over 4
# GPUs (per-device batch 2 x 4 processes x grad-accum 1 = effective batch 8,
# same as v1's single-GPU 2x1x4) purely for wall-clock speed. Baseline:
# DorianAtSchool/qwen3vl-8b-robocasa-sft.
# See training/bc_task_vlm/plans/reasoning_sft_v1_5_plan.md.
#   sbatch run_v1_5_verbalized_sampling_full.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_local_train_subset_21
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
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v1_5-verbalized-sampling
export WANDB_RUN_NAME=qwen3vl-8b-v1_5-verbalized-sampling

# --- schedule (v1's exact recipe: 3 epochs, v1 eval/save cadence, full
# validation set each eval -- v1 never set EVAL_MAX_SAMPLES) ---
export NUM_EPOCHS=3
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export EVAL_GENERATION_MAX_SAMPLES=128

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
  # Per-run cache dir -- the default (.cache/bc_task_vlm/examples) is repo-
  # global and races when multiple training jobs run concurrently (this
  # caused a DDP "inconsistent 0 params on one rank" crash).
  --training-samples-cache-dir ".cache/bc_task_vlm/examples_${WANDB_RUN_NAME}"
)

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
