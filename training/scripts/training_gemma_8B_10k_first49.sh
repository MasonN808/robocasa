#!/usr/bin/env bash
#SBATCH --job-name=gemma4-e4b-10k-bc-vlm
#SBATCH --account=bgjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-task=2
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

export STAGED_DATASET_ROOT="${STAGED_DATASET_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf/robocasa_20260430T030150Z_full_10k_first49}"
export OUTPUT_DIR="${OUTPUT_DIR:-training/bc_task_vlm/runs/gemma4-e4b-10k-first49-bc-task-vlm-2gpu-bs4-lr1e5-evalbs8-workers8-evalmax1000-structtraj100-ddp-timeout7200-ddpunusedtrue}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-gemma4-e4b-10k-first49-2gpu-bs4-lr1e5-evalbs8-workers8-evalmax1000-structtraj100-ddp-timeout7200-ddpunusedtrue-sft}"
export WANDB_TAGS="${WANDB_TAGS:-bc_task_vlm,gemma4,e4b,base,sft,robocasa,10k,test,2gpu,bs4,lr1e-5,evalbs8,workers8,examplebuild8,example_cache,evalmax1000,structtraj100,nccl_cumem_host0,nccl_net_plugin_forced_none,ddp_timeout7200,ddp_unused_parameters_true}"

export SKIP_HF_STAGING="${SKIP_HF_STAGING:-true}"
export HF_STAGE_MAX_TOTAL_EPISODES="${HF_STAGE_MAX_TOTAL_EPISODES:-10000}"
export VAL_TRAJECTORY_FRACTION="${VAL_TRAJECTORY_FRACTION:-0.1}"
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-4}"
export PER_DEVICE_EVAL_BATCH_SIZE="${PER_DEVICE_EVAL_BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-2}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export EXAMPLE_BUILD_WORKERS="${EXAMPLE_BUILD_WORKERS:-8}"
export DDP_TIMEOUT_SECONDS="${DDP_TIMEOUT_SECONDS:-7200}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export SAVE_STEPS="${SAVE_STEPS:-1000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-6}"
export EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-1000}"
export EVAL_GENERATION_MAX_SAMPLES="${EVAL_GENERATION_MAX_SAMPLES:-1000}"
export EVAL_GENERATION_MAX_TRAJECTORIES="${EVAL_GENERATION_MAX_TRAJECTORIES:-100}"
export EVAL_GENERATION_BATCH_SIZE="${EVAL_GENERATION_BATCH_SIZE:-4}"

exec training/scripts/training_gemma_8B_100k.sh
