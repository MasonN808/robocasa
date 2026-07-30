#!/usr/bin/env bash
#SBATCH --job-name=eval-v3-structured-random-offsim
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

# Off-sim (teacher-forced, structured) eval for the v3 (active-observation)
# adapter trained on structured_random. Same harness as the v1/v1.5 evals,
# plus --predict-acting-agent --train-get-image to match training (required:
# eval_standalone.py raises if --train-get-image is set without
# --predict-acting-agent).
#   sbatch --dependency=afterok:<train_job_id> eval_v3_structured_random.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export PYTHONUNBUFFERED=1

adapter_path=training/bc_task_vlm/runs/qwen3vl-8b-v3-structured-random
manifest_dir=training/bc_task_vlm/eval_manifests/structured_random_30
dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_eval_root
run_prefix=training/bc_task_vlm/eval_runs/qwen3vl_8b_v3_structured_random

for split in heldout_trajectories heldout_tasks; do
  echo "=== ${split} ==="
  "${PYTHON_BIN}" -m training.bc_task_vlm.eval_standalone \
    --backend hf \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --adapter-path "${adapter_path}" \
    --predict-acting-agent \
    --train-get-image \
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
