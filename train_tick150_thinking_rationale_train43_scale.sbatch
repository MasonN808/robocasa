#!/usr/bin/env bash
#SBATCH --job-name=train-thinkrat-t43
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=08:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
: "${SCALE:?Set SCALE to 30 or 60}"
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
cd "${repo}"
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export MODEL_PATH=Qwen/Qwen3-VL-8B-Thinking
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered
export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_reasoning_scale${SCALE}_train43_noindex_promptv9
export HELD_OUT_SELECTION=${repo}/training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/43_train_10_heldout.json
run_name=qwen3vl8b-thinking-tick150-s${SCALE}-rationale-train43-held10-ep1-halfckpt-noindex-promptv9
export OUTPUT_DIR=${repo}/training/bc_task_vlm/runs/${run_name}
export WANDB_RUN_NAME=${run_name} WANDB_PROJECT=robocasa-bc-task-vlm WANDB_MODE=online
export VAL_TASKS='' VAL_TRAJECTORY_FRACTION=0
export NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=8 GRAD_ACCUM=2
export IMAGE_RESOLUTION=512 MAX_LENGTH=8192 ATTN_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"${PYTHON_BIN}" -c 'import wandb; assert wandb.login(); print("W&B viewer:", wandb.Api().viewer)'
source training/scripts/local/_launch_args.sh
launch_args+=(
  --partial-history --train-get-image --train-reasoning
  --partial-step-index-mode none --partial-observation-mode consume-once
  --gradient-checkpointing --no-ddp-find-unused-parameters
  --num-epochs 1 --checkpoints-per-epoch 2 --save-total-limit 2
  --eval-generation-max-samples 0
  --wandb-tags "bc_task_vlm,qwen3vl8b-thinking,scale${SCALE},train43,held10,noindex,rationale,ep1"
)
exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
