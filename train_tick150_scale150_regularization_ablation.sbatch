#!/usr/bin/env bash
#SBATCH --job-name=train-150-reg-ablate
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:b200:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=512G
#SBATCH --time=13:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
learning_rates=(1e-4 1e-4 5e-5)
weight_decays=(0.0 0.01 0.01)
arm_names=(lr1e4-wd0 lr1e4-wd001 lr5e5-wd001)
idx="${SLURM_ARRAY_TASK_ID}"

cd "${repo}"
export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export MODEL_PATH=Qwen/Qwen3-VL-8B-Instruct
export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered
export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_scale_150_train47_noindex_promptv9
export HELD_OUT_SELECTION=${repo}/training/bc_task_vlm/scale_experiments/tick150_v1/default_47_train_6_heldout.json
export LEARNING_RATE="${learning_rates[idx]}"
export WEIGHT_DECAY="${weight_decays[idx]}"
run_name="qwen3vl8b-tick150-scale150-ep1-quarterckpt-${arm_names[idx]}-noindex-promptv9"
export OUTPUT_DIR=${repo}/training/bc_task_vlm/runs/${run_name}
export WANDB_RUN_NAME=${run_name} WANDB_PROJECT=robocasa-bc-task-vlm WANDB_MODE=online
export VAL_TASKS='' VAL_TRAJECTORY_FRACTION=0
export NUM_PROCESSES=4 PER_DEVICE_BATCH_SIZE=8 GRAD_ACCUM=2
export IMAGE_RESOLUTION=512 MAX_LENGTH=8192 ATTN_IMPLEMENTATION=flash_attention_2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

"${PYTHON_BIN}" -c 'import wandb; assert wandb.login(); print("W&B viewer:", wandb.Api().viewer)'
source training/scripts/local/_launch_args.sh
launch_args+=(
  --partial-history --train-get-image --partial-step-index-mode none
  --partial-observation-mode consume-once --gradient-checkpointing
  --no-ddp-find-unused-parameters --num-epochs 1 --checkpoints-per-epoch 4
  --save-total-limit 4 --eval-generation-max-samples 0
  --wandb-tags "bc_task_vlm,qwen3vl8b,scale150,tick150,noindex,ep1,regularization-ablation,${arm_names[idx]}"
)
exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
