#!/usr/bin/env bash
#SBATCH --job-name=qwen-ootb-nidx
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail

: "${SPLIT:?Set SPLIT}"
PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
EVAL_SOURCE=${EVAL_SOURCE:-tick30}
case "${EVAL_SOURCE}" in
  tick30) DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_concurrent ;;
  tick100_from150) DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent ;;
  tick100_from150_v3) DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3 ;;
  tick30_revalidated_v3) DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_revalidated_v3 ;;
  *) echo "Unsupported EVAL_SOURCE=${EVAL_SOURCE}" >&2; exit 1 ;;
esac
MANIFEST="training/bc_task_vlm/eval_manifests/${EVAL_SOURCE}/eval_manifest_${SPLIT}.json"
RUN_SUFFIX=${RUN_SUFFIX:-}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-}

mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore

limit_args=()
if [[ -n "${MAX_TRAJECTORIES}" ]]; then
  limit_args=(--max-trajectories "${MAX_TRAJECTORIES}" --no-record-firsts)
fi

output_dir="training/bc_task_vlm/eval_runs/livesim_ots-noindex-qwen3vl8b-${EVAL_SOURCE}__${SPLIT}${RUN_SUFFIX}"
echo "model=Qwen/Qwen3-VL-8B-Instruct (no adapter) split=${SPLIT} output=${output_dir}"
echo "contract=partial-history,index-none,consume-once,${EVAL_SOURCE},FSM,raster@60dpi"

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
  --backend hf \
  --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --train-get-image \
  --partial-history \
  --partial-step-index-mode none \
  --partial-observation-mode consume-once \
  --no-predict-task-complete \
  --sft-format tool_call \
  --no-forced-json \
  --manifest "${MANIFEST}" \
  --dataset-root "${DATASET_ROOT}" \
  --output-dir "${output_dir}" \
  --success-criterion fsm \
  --step-budget-factor 2.0 \
  --map-renderer raster \
  --map-dpi 60 \
  --gl-backend egl \
  --resume \
  "${limit_args[@]}"
