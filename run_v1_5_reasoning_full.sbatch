#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v1_5-reasoning-sft
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# SFT v1.5 (reasoning probe) ablation vs the v1 baseline: fresh LoRA on
# Qwen3-VL-8B-Instruct, v1's exact recipe (single GPU, per-device batch 2,
# grad-accum 4, sdpa, 512^2 images, 3 epochs) plus --train-reasoning. The
# reasoning prefix is the only variable versus the v1 baseline run
# (DorianAtSchool/qwen3vl-8b-robocasa-sft) so results are directly comparable.
# See training/bc_task_vlm/plans/reasoning_sft_v1_5_plan.md.
#   sbatch run_v1_5_reasoning_full.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# --- single GPU, v1-matching batch config (defaults from _launch_args.sh) ---
export NUM_PROCESSES=1
export PER_DEVICE_BATCH_SIZE=2
export GRAD_ACCUM=4
export ATTN_IMPLEMENTATION=sdpa
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

# --- run identity ---
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v1_5-reasoning
export WANDB_RUN_NAME=qwen3vl-8b-v1_5-reasoning

# --- schedule (v1's exact recipe: 3 epochs, v1 eval/save cadence) ---
export TRAIN_REASONING=1
export NUM_EPOCHS=3
export EVAL_STEPS=500
export SAVE_STEPS=500
export SAVE_TOTAL_LIMIT=2
export EVAL_GENERATION_MAX_SAMPLES=128

exec training/scripts/local/train_qwen3vl_8b.sh
