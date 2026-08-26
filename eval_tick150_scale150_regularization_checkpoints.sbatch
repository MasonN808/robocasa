#!/usr/bin/env bash
#SBATCH --job-name=eval-150-reg
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --array=0-11%5
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail

arms=(lr1e4-wd0 lr1e4-wd001 lr5e5-wd001)
checkpoints=(1780 3559)
epoch_tags=(0p50 1p00)
splits=(train_task_types heldout_task_types)

idx=${SLURM_ARRAY_TASK_ID}
arm_idx=$((idx / 4))
within_arm=$((idx % 4))
checkpoint_idx=$((within_arm / 2))
split_idx=$((within_arm % 2))

arm=${arms[arm_idx]}
checkpoint=${checkpoints[checkpoint_idx]}
epoch_tag=${epoch_tags[checkpoint_idx]}
split=${splits[split_idx]}
run_dir="training/bc_task_vlm/runs/qwen3vl8b-tick150-scale150-ep1-quarterckpt-${arm}-noindex-promptv9"
adapter="${run_dir}/checkpoint-${checkpoint}"
[[ -s "${adapter}/adapter_model.safetensors" ]] || {
  echo "Missing completed adapter ${adapter}" >&2
  exit 1
}

export MANIFEST=training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json
export DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered
export SCENE_COMPATIBILITY_CACHE=training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json
export EPISODES_PER_TASK=10
export EVALUATION_SEED=20260817
export SCENE_SAMPLING_SEED=20260819
export WORKERS=4
export MODEL=Qwen/Qwen3-VL-8B-Instruct
export ADAPTER="${adapter}"
export COMMUNICATION_MODE=full
export COHORT_SPLIT="${split}"
export RUN_NAME="sft_scale150_${arm}_ep${epoch_tag}_fixed10_promptv9"

exec bash fixed_live_sim_open_weight_vllm.sbatch
