#!/usr/bin/env bash
#SBATCH --job-name=eval-v1_5-reasoning
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Evaluates the v1.5 reasoning-SFT adapter (training/bc_task_vlm/runs/
# qwen3vl-8b-v1_5-reasoning) with the exact same standalone harness config
# used for v1's own headline numbers (native tool_call format, no forced
# JSON -- required both because that's what v1 was scored with and because
# forced-JSON can't coexist with a free-text <think> prefix). No reasoning
# -specific eval-code changes needed: Qwen3-VL-8B-Instruct's chat template
# has no native thinking-mode logic (verified: template never references
# "think"), so the model just free-generates the <think>...</think> prefix
# it was supervised to produce, and evaluation.py::split_reasoning already
# strips it before parsing. Runs both headline splits sequentially.
#   sbatch --dependency=afterok:<train_job_id>:<stage_job_id> eval_v1_5_reasoning.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export PYTHONUNBUFFERED=1

adapter_path=training/bc_task_vlm/runs/qwen3vl-8b-v1_5-reasoning
manifest_dir=training/bc_task_vlm/eval_manifests/v1_5_reasoning
run_prefix=training/bc_task_vlm/eval_runs/qwen3vl_8b_v1_5_reasoning

for split in heldout_trajectories heldout_tasks; do
  echo "=== ${split} ==="
  "${PYTHON_BIN}" -m training.bc_task_vlm.eval_standalone \
    --backend hf \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --adapter-path "${adapter_path}" \
    --sft-format tool_call \
    --no-forced-json \
    --image-resolution 512 \
    --batch-size 4 \
    --manifest "${manifest_dir}/eval_manifest_${split}.json" \
    --output-dir "${run_prefix}__${split}" \
    --resume
done

echo "Done."
