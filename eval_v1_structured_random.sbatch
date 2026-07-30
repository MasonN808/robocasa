#!/usr/bin/env bash
#SBATCH --job-name=eval-v1-structured-random
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

# Evaluates the v1 (non-reasoning) baseline trained on structured_random,
# against structured_random's own manifests/data
# (DorianAtSchool/robocasa-structured-random-30) -- the within-source
# counterpart to eval_v1_5_verbalized_sampling.sbatch. Native tool_call
# format, no forced JSON, matching v1's own headline eval config.
#   sbatch --dependency=afterok:<train_job_id> eval_v1_structured_random.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export PYTHONUNBUFFERED=1

adapter_path=training/bc_task_vlm/runs/qwen3vl-8b-v1-structured-random/checkpoint-4362
manifest_dir=training/bc_task_vlm/eval_manifests/structured_random_30
dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_eval_root
run_prefix=training/bc_task_vlm/eval_runs/qwen3vl_8b_v1_structured_random

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
    --dataset-root "${dataset_root}" \
    --output-dir "${run_prefix}__${split}" \
    --resume
done

echo "Done."
