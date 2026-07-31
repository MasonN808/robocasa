#!/usr/bin/env bash
#SBATCH --job-name=v3-offsim
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:b200:1
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Off-sim (teacher-forced) eval for one cell of the v3 matrix, both splits.
# Eval flags MUST mirror the flags the cell was trained with, or the prompts
# differ from training and the numbers are meaningless.
#
#   sbatch --export=ALL,SOURCE=structured_random,OBS=partial,REASONING=1 \
#     eval_v3_matrix_offsim.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

: "${SOURCE:?Set SOURCE}"
: "${OBS:?Set OBS}"
REASONING=${REASONING:-0}

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export PYTHONUNBUFFERED=1

case "${SOURCE}" in
  verbalized_sampling)
    src_tag=vs
    manifest_dir=training/bc_task_vlm/eval_manifests/exp52
    dataset_root=training/bc_task_vlm/eval_data_subset
    ;;
  structured_random)
    src_tag=sr
    manifest_dir=training/bc_task_vlm/eval_manifests/structured_random_30
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_eval_root
    ;;
  *) echo "Unknown SOURCE: ${SOURCE}" >&2; exit 1 ;;
esac

# eval_standalone accepts --train-get-image with EITHER --predict-acting-agent
# (centralized) or --partial-history (partial), and rejects the two together.
case "${OBS}" in
  centralized)
    obs_args=(--predict-acting-agent --train-get-image)
    obs_tag=cent
    ;;
  partial)
    obs_args=(
      --partial-history
      --no-predict-acting-agent
      --train-get-image
      --partial-step-index-mode local
      --partial-observation-mode consume-once
    )
    obs_tag=partial
    ;;
  *) echo "Unknown OBS: ${OBS}" >&2; exit 1 ;;
esac

reason_tag=noreason
[[ "${REASONING}" == "1" ]] && reason_tag=reason

run_name="qwen3vl-8b-v3-${src_tag}-${obs_tag}-${reason_tag}"
adapter_path="training/bc_task_vlm/runs/${run_name}"
run_prefix="training/bc_task_vlm/eval_runs/offsim_${run_name}"

echo "=== off-sim eval: ${run_name} ==="
echo "  adapter=${adapter_path}"
echo "  manifests=${manifest_dir}"
echo "  obs_args=${obs_args[*]}"

for split in heldout_trajectories heldout_tasks; do
  echo "--- ${split} ---"
  "${PYTHON_BIN}" -m training.bc_task_vlm.eval_standalone \
    --backend hf \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
    --adapter-path "${adapter_path}" \
    "${obs_args[@]}" \
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
