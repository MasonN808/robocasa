#!/usr/bin/env bash
#SBATCH --job-name=qwen3vl8b-v2-continue-sft
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256g
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Cluster twin of the local v2-continue run: single-GPU 8B LoRA initialized
# from the v1 adapter, agent-prediction data, wandb online. Reuses the local
# launch path so the two v2 runs cannot drift apart in config:
#   sbatch training/scripts/train_v2_continue_8b_cluster.sh
# Env overrides (must MATCH the local v2-scratch run for compute-matching):
#   TRAJ_PER_TASK (via pre-staged DATA_ROOT), NUM_EPOCHS, MAX_STEPS.
set -euo pipefail
mkdir -p slurm_logs

export UV_CACHE_DIR=/work/hdd/bgjs/dbenhamougoldfajn/.uv_cache
export HF_HOME="${HF_HOME:-/work/hdd/bgjs/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

export PYTHON_BIN="/work/hdd/bgjs/dbenhamougoldfajn/robocasa/.venv/bin/python"
export DATA_ROOT="${DATA_ROOT:-/work/hdd/bgjs/dbenhamougoldfajn/robocasa/training/bc_task_vlm/local_train_subset}"
export OUTPUT_DIR="${OUTPUT_DIR:-training/bc_task_vlm/runs/qwen3vl-8b-agentsft-continue}"
export PREDICT_ACTING_AGENT=1
export INIT_ADAPTER_PATH="${INIT_ADAPTER_PATH:-DorianAtSchool/qwen3vl-8b-robocasa-sft}"
# H200 micro-batch split; effective batch stays 8 (the compute-matching
# invariant shared with the local 5090 v2-scratch run, which uses 1x8).
export PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-8}"
export GRAD_ACCUM="${GRAD_ACCUM:-1}"
export WANDB_MODE=online
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-qwen3vl-8b-agentsft-continue}"

[ -d "${DATA_ROOT}" ] || { echo "DATA_ROOT ${DATA_ROOT} missing - stage first"; exit 1; }
exec bash training/scripts/local/train_qwen3vl_8b.sh
