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
PARTIAL_STEP_INDEX_MODE=${PARTIAL_STEP_INDEX_MODE:-local}
RUN_SUFFIX=${RUN_SUFFIX:-}
MAX_TRAJECTORIES=${MAX_TRAJECTORIES:-}
MAP_RENDERER=${MAP_RENDERER:-raster}
MAP_DPI=${MAP_DPI:-60}
RECORD_FIRSTS=${RECORD_FIRSTS:-1}

case "${PARTIAL_STEP_INDEX_MODE}" in
  none|local) ;;
  *) echo "PARTIAL_STEP_INDEX_MODE must be none or local, got: ${PARTIAL_STEP_INDEX_MODE}" >&2; exit 1 ;;
esac

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

case "${SOURCE}" in
  tick30)
    src_tag=tick30
    manifest_dir=training/bc_task_vlm/eval_manifests/tick30
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_concurrent
    ;;
  tick100)
    src_tag=tick100
    manifest_dir=training/bc_task_vlm/eval_manifests/tick100
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_structured_random_concurrent
    ;;
  tick100_from150)
    src_tag=tick100-from150
    manifest_dir=training/bc_task_vlm/eval_manifests/tick100_from150
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent
    ;;
  tick100_from150_v3)
    src_tag=tick100-from150-v3
    manifest_dir=training/bc_task_vlm/eval_manifests/tick100_from150_v3
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3
    ;;
  tick30_revalidated_v3)
    src_tag=tick30-revalidated-v3
    manifest_dir=training/bc_task_vlm/eval_manifests/tick30_revalidated_v3
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_revalidated_v3
    ;;
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

if [[ "${SOURCE}" == "tick100_from150" && "${PARTIAL_STEP_INDEX_MODE}" != "none" ]]; then
  echo "tick100_from150 evaluation forbids local and global step indices" >&2
  exit 1
fi
if [[ "${SOURCE}" == "tick100_from150_v3" && "${PARTIAL_STEP_INDEX_MODE}" != "none" ]]; then
  echo "tick100_from150_v3 evaluation forbids local and global step indices" >&2
  exit 1
fi
if [[ "${SOURCE}" == "tick30_revalidated_v3" && "${PARTIAL_STEP_INDEX_MODE}" != "none" ]]; then
  echo "tick30_revalidated_v3 evaluation forbids local and global step indices" >&2
  exit 1
fi

case "${OBS}" in
  centralized)
    obs_args=(--train-get-image)
    obs_tag=cent
    ;;
  partial)
    obs_args=(
      --train-get-image
      --partial-history
      --partial-step-index-mode "${PARTIAL_STEP_INDEX_MODE}"
      --partial-observation-mode consume-once
    )
    obs_tag="partial-idx${PARTIAL_STEP_INDEX_MODE}"
    ;;
  *) echo "Unknown OBS: ${OBS}" >&2; exit 1 ;;
esac

reason_tag=noreason
[[ "${REASONING}" == "1" ]] && reason_tag=reason

tc_args=()
tc_tag=""
if [[ "${TASK_COMPLETE:-1}" == "0" ]]; then
  tc_args=(--no-predict-task-complete)
  tc_tag="-notc"
fi

run_name="qwen3vl-8b-v3-${src_tag}-${obs_tag}-${reason_tag}${tc_tag}${RUN_SUFFIX}"
adapter_path="training/bc_task_vlm/runs/${run_name}"
if [[ ! -f "${adapter_path}/adapter_config.json" ]]; then
  echo "Missing completed adapter: ${adapter_path}/adapter_config.json" >&2
  exit 1
fi

# SLURM --export is comma-delimited and so is --tasks, so a task list cannot be
# passed through --export directly (it gets split into bogus env assignments --
# jobs 242698-242700 silently received only the first task and exited as
# "COMPLETED" having done nothing). Pass a FILE path instead.
TASKS_ARG=""
if [[ -n "${TASKS_FILE:-}" && -f "${TASKS_FILE}" ]]; then
  TASKS_ARG="--tasks $(cat "${TASKS_FILE}")"
  echo "  task filter from ${TASKS_FILE}: $(tr ',' '\n' < "${TASKS_FILE}" | wc -l) tasks"
fi

echo "=== live-sim: ${run_name} / ${SPLIT} ==="
echo "  obs_args=${obs_args[*]}"
echo "  adapter=${adapter_path} map=${MAP_RENDERER}@${MAP_DPI}dpi"

limit_args=()
if [[ -n "${MAX_TRAJECTORIES}" ]]; then
  limit_args+=(--max-trajectories "${MAX_TRAJECTORIES}")
fi
record_args=()
if [[ "${RECORD_FIRSTS}" == "0" ]]; then
  record_args+=(--no-record-firsts)
fi

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
  --backend hf \
  --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path "${adapter_path}" \
  "${obs_args[@]}" \
  ${tc_args[@]+"${tc_args[@]}"} \
  --sft-format tool_call \
  --no-forced-json \
  --manifest "${manifest_dir}/eval_manifest_${SPLIT}.json" \
  --dataset-root "${dataset_root}" \
  --output-dir "training/bc_task_vlm/eval_runs/livesim_${run_name}__${SPLIT}${OUT_SUFFIX:-}" \
  --success-criterion fsm \
  --step-budget-factor 2.0 \
  --map-renderer "${MAP_RENDERER}" \
  --map-dpi "${MAP_DPI}" \
  --gl-backend egl \
  --resume \
  "${limit_args[@]}" \
  "${record_args[@]}" \
  ${TASKS_ARG:-}
