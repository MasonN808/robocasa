#!/usr/bin/env bash
#SBATCH --job-name=v3-livesim
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Closed-loop live-sim eval for one cell of the v3 matrix, ONE split per job
# (a split is 75 rollouts at roughly 50-120s each, so splitting keeps each job
# inside a sane walltime and lets them backfill in parallel).
#
# Flags mirror training, same as the off-sim script. live_sim_eval.py hardcodes
# agent prediction, so there is no --predict-acting-agent to pass; partial-obs
# is selected with --partial-history plus its two mode flags.
#
# --success-criterion defaults to "fsm", and since the fix in this branch BOTH
# rollout paths honor it -- previously run_trajectory (centralized) hardcoded
# "native AND fsm" while the partial path used fsm alone, which would have
# scored centralized cells on a strictly harder bar than partial ones.
#
#   sbatch --export=ALL,SOURCE=structured_random,OBS=partial,REASONING=1,SPLIT=heldout_tasks \
#     live_sim_v3_matrix.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

: "${SOURCE:?Set SOURCE}"
: "${OBS:?Set OBS}"
: "${SPLIT:?Set SPLIT}"
REASONING=${REASONING:-0}

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
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

case "${OBS}" in
  centralized)
    obs_args=(--train-get-image)
    obs_tag=cent
    ;;
  partial)
    obs_args=(
      --train-get-image
      --partial-history
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

echo "=== live-sim: ${run_name} / ${SPLIT} ==="
echo "  obs_args=${obs_args[*]}"

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
  --backend hf \
  --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path "${adapter_path}" \
  "${obs_args[@]}" \
  --sft-format tool_call \
  --no-forced-json \
  --manifest "${manifest_dir}/eval_manifest_${SPLIT}.json" \
  --dataset-root "${dataset_root}" \
  --output-dir "training/bc_task_vlm/eval_runs/livesim_${run_name}__${SPLIT}" \
  --success-criterion fsm \
  --step-budget-factor 2.0 \
  --gl-backend egl \
  --resume
