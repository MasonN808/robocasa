#!/usr/bin/env bash
#SBATCH --job-name=train-tick150-scale-smoke
#SBATCH --partition=b200-devel
#SBATCH --gres=gpu:b200:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
cd "${repo}"
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered
export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_scale_30_train47_noindex_promptv9
export HELD_OUT_SELECTION=${repo}/training/bc_task_vlm/scale_experiments/tick150_v1/default_47_train_6_heldout.json
export OUTPUT_DIR=${repo}/training/bc_task_vlm/runs/qwen3vl8b-tick150-scale-smoke-noindex-promptv9
export WANDB_RUN_NAME=qwen3vl8b-tick150-scale-smoke-noindex-promptv9
export WANDB_PROJECT=robocasa-bc-task-vlm
export WANDB_MODE=online
export VAL_TASKS=''
export VAL_TRAJECTORY_FRACTION=0
export NUM_PROCESSES=1 PER_DEVICE_BATCH_SIZE=1 GRAD_ACCUM=1
export IMAGE_RESOLUTION=512 MAX_LENGTH=8192 ATTN_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" -c 'import wandb; assert wandb.login(); print("W&B viewer:", wandb.Api().viewer)'
source training/scripts/local/_launch_args.sh
launch_args+=(
  --partial-history --train-get-image --partial-step-index-mode none
  --partial-observation-mode consume-once --gradient-checkpointing
  --no-ddp-find-unused-parameters --num-epochs 2 --max-steps 20
  --save-steps 10 --save-total-limit 2 --eval-generation-max-samples 0
)
exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
