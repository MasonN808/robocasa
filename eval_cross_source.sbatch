#!/usr/bin/env bash
#SBATCH --job-name=eval-cross-source
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

# Cross-source generalization eval: scores a model trained on ONE
# data-generation source against the OTHER source's held-out splits. This is
# the controlled version of the "which data source generalizes better"
# question -- the earlier per-source numbers each scored a model on its own
# source's eval set (different reference trajectories, different comm text,
# different comm_step_fraction), so they were not directly comparable.
#
# Parameterized via --export; ADAPTER_PATH may be a local dir or an HF repo id.
#   sbatch --export=ALL,ADAPTER_PATH=...,EVAL_SOURCE=exp52,RUN_PREFIX=... eval_cross_source.sbatch
#   EVAL_SOURCE is one of: exp52 | structured_random
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export PYTHONUNBUFFERED=1

: "${ADAPTER_PATH:?Set ADAPTER_PATH}"
: "${EVAL_SOURCE:?Set EVAL_SOURCE to exp52 or structured_random}"
: "${RUN_PREFIX:?Set RUN_PREFIX}"

case "${EVAL_SOURCE}" in
  exp52)
    manifest_dir=training/bc_task_vlm/eval_manifests/exp52
    dataset_root=training/bc_task_vlm/eval_data_subset
    ;;
  structured_random)
    manifest_dir=training/bc_task_vlm/eval_manifests/structured_random_30
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_eval_root
    ;;
  *)
    echo "Unknown EVAL_SOURCE: ${EVAL_SOURCE}" >&2
    exit 1
    ;;
esac

echo "adapter=${ADAPTER_PATH}  eval_source=${EVAL_SOURCE}  out=${RUN_PREFIX}"

for split in heldout_trajectories heldout_tasks; do
  echo "=== ${split} ==="
  "${PYTHON_BIN}" -m training.bc_task_vlm.eval_standalone \
    --backend hf \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --adapter-path "${ADAPTER_PATH}" \
    --sft-format tool_call \
    --no-forced-json \
    --image-resolution 512 \
    --batch-size 4 \
    --manifest "${manifest_dir}/eval_manifest_${split}.json" \
    --dataset-root "${dataset_root}" \
    --output-dir "${RUN_PREFIX}__${split}" \
    --resume
done

echo "Done."
