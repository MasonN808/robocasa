#!/usr/bin/env bash
#SBATCH --job-name=eval-v1_5-verbalized-sampling
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

# Evaluates the CLEAN v1.5 reasoning-SFT adapter, trained on the
# verbalized_sampling data source (robocasa_local_train_subset_21, v1's
# actual training data), against v1's own real eval manifests + data
# (DorianAtSchool/robocasa-exp52-eval), instead of the freshly-rebuilt
# substitute manifests used for the first v1_5-reasoning eval. Same harness
# config as v1's own headline numbers: native tool_call format, no forced
# JSON (required both because that's what v1 was scored with and because
# forced-JSON can't coexist with a free-text <think> prefix).
#   sbatch --dependency=afterok:<train_job_id> eval_v1_5_verbalized_sampling.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export PYTHONUNBUFFERED=1

adapter_path=training/bc_task_vlm/runs/qwen3vl-8b-v1_5-verbalized-sampling
manifest_dir=training/bc_task_vlm/eval_manifests/exp52
dataset_root=training/bc_task_vlm/eval_data_subset
run_prefix=training/bc_task_vlm/eval_runs/qwen3vl_8b_v1_5_verbalized_sampling

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
