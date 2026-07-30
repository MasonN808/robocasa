#!/usr/bin/env bash
#SBATCH --job-name=eval-v1-verbalized-sampling-baseline
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

# Evaluates the pre-existing v1 (non-reasoning) baseline adapter
# (DorianAtSchool/qwen3vl-8b-robocasa-sft, already published to HF, no
# training needed) against the real exp52 manifests + data, to get its full
# raw structured_eval_metrics.json locally -- EXPERIMENT.md only recorded
# the headline judged-overall numbers (.878/.675), not the full raw metric
# breakdown, and results_table.csv isn't on this cluster. Fills in the
# missing v1-verbalized_sampling cell for the v1-vs-v1.5-per-dataset and
# v1-vs-v1-across-datasets comparisons.
#   sbatch eval_v1_verbalized_sampling_baseline.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export PYTHONUNBUFFERED=1

adapter_path=DorianAtSchool/qwen3vl-8b-robocasa-sft
manifest_dir=training/bc_task_vlm/eval_manifests/exp52
dataset_root=training/bc_task_vlm/eval_data_subset
run_prefix=training/bc_task_vlm/eval_runs/qwen3vl_8b_v1_verbalized_sampling_baseline

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
