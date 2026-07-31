#!/usr/bin/env bash
#SBATCH --job-name=smoke-ckpt-ddp
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:b200:4
#SBATCH --mem=512G
#SBATCH --time=00:40:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# 30-step probe of the config we actually want back: gradient checkpointing
# ON at per-device batch 8, with ddp_find_unused_parameters=False so
# checkpointing's re-run of the forward pass can't double-mark params.
#
# Three outcomes worth distinguishing:
#   loss lines appear      -> the fix works; the matrix can use ckpt + batch 8
#   "marked as ready twice"-> find_unused=False did not solve it
#   "Expected to have finished reduction" / hang
#                          -> some LoRA params genuinely go ungradiented, so
#                             find_unused=True is load-bearing and
#                             checkpointing simply cannot be used with it
# Short walltime so it backfills quickly and cannot sit on a node.
#   sbatch smoke_ckpt_ddp.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_local_train_subset_21
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=offline
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export NUM_PROCESSES=4
export PER_DEVICE_BATCH_SIZE=8
export GRAD_ACCUM=2
export ATTN_IMPLEMENTATION=flash_attention_2
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

export OUTPUT_DIR=training/bc_task_vlm/runs/smoke-ckpt-ddp
export WANDB_RUN_NAME=smoke-ckpt-ddp

source training/scripts/local/_launch_args.sh
launch_args+=(
  --predict-acting-agent
  --train-get-image
  --train-reasoning
  --gradient-checkpointing
  --no-ddp-find-unused-parameters
  --ddp-timeout-seconds 600
  --max-steps 30
  --num-epochs 1
  --eval-steps 100000
  --save-steps 100000
  --eval-generation-max-samples 0
  --training-samples-cache-dir ".cache/bc_task_vlm/examples_qwen3vl-8b-v3-vs-cent-reason"
)

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
