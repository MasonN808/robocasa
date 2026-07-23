#!/usr/bin/env bash
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa

# v1.5 reasoning-SFT ablation smoke test: v1 base config (agent given, no
# get_image) plus --train-reasoning. Matches v1's compute exactly (single
# GPU, per-device batch 2, grad-accum 4, sdpa) so the reasoning prefix is the
# only variable versus the v1 baseline.

# --- environment ---
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online

# --- single GPU, v1-matching batch config (defaults from _launch_args.sh) ---
export NUM_PROCESSES=1
export PER_DEVICE_BATCH_SIZE=2
export GRAD_ACCUM=4
export ATTN_IMPLEMENTATION=sdpa
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

# --- run identity (throwaway) ---
export OUTPUT_DIR=training/bc_task_vlm/runs/qwen3vl-8b-v1_5-reasoning-smoke
export WANDB_RUN_NAME=qwen3vl-8b-v1_5-reasoning-smoke

# --- schedule (smoke: a handful of steps only) ---
export TRAIN_REASONING=1
export NUM_EPOCHS=3
export EVAL_STEPS=10
export SAVE_STEPS=10
export SAVE_TOTAL_LIMIT=1
export EVAL_GENERATION_MAX_SAMPLES=8
export MAX_STEPS=20

exec training/scripts/local/train_qwen3vl_8b.sh
