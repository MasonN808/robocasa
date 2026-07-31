#!/usr/bin/env bash
#SBATCH --job-name=ots-eval
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Off-the-shelf (no fine-tuning) baselines through the SAME eval path as the
# v3 matrix cells: same manifests, same splits, same tool set.
#
# OBS selects centralized (--predict-acting-agent) or partial
# (--partial-history, no agent prediction); TASK_COMPLETE=0 drops the
# task_complete tool. An untrained model is simply offered these tool sets
# rather than supervised on them.
#
# Caveat worth carrying into any comparison: --partial-history reshapes the
# prompt around one agent's private history, which is a training contract the
# fine-tuned cells learned and an off-the-shelf model never saw. A weak
# partial score here may reflect prompt-format unfamiliarity rather than a
# limit on the model's reasoning.
#
# Off-sim passes --native-tools: an untrained model emits real function-call
# parts, not the JSON-in-text dialect the fine-tuned cells were supervised on.
# Without it Gemini returns empty text and every sample scores 0 -- a harness
# artifact, not a capability result. Live-sim needs no such flag because its
# GeminiPolicy is native-function-calling by construction. The parsed call is
# scored identically either way, but note in any writeup that the baseline was
# met in its own dialect while the tuned cells were met in theirs.
#
# A GPU is requested even though the policy is a hosted API, because live-sim
# renders MuJoCo through EGL. Off-sim does not need it but shares the script.
#
#   sbatch --export=ALL,MODEL=gemini-3-flash-preview,EVAL_SOURCE=exp52,MODE=offsim eval_offtheshelf.sbatch
#   sbatch --export=ALL,MODEL=...,EVAL_SOURCE=structured_random,MODE=livesim,SPLIT=heldout_tasks ...
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

: "${MODEL:?Set MODEL to a Vertex model id}"
: "${EVAL_SOURCE:?Set EVAL_SOURCE to exp52 or structured_random}"
: "${MODE:?Set MODE to offsim or livesim}"

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export GOOGLE_APPLICATION_CREDENTIALS=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/config/gcloud/application_default_credentials.json
export GOOGLE_CLOUD_PROJECT=access-cis260215-538827
# Regional Vertex endpoints 404 here; global is required (EXPERIMENT.md s8).
export GOOGLE_CLOUD_LOCATION=global

case "${EVAL_SOURCE}" in
  exp52)
    manifest_dir=training/bc_task_vlm/eval_manifests/exp52
    dataset_root=training/bc_task_vlm/eval_data_subset
    src_tag=vs
    ;;
  structured_random)
    manifest_dir=training/bc_task_vlm/eval_manifests/structured_random_30
    dataset_root=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_eval_root
    src_tag=sr
    ;;
  *) echo "Unknown EVAL_SOURCE: ${EVAL_SOURCE}" >&2; exit 1 ;;
esac

model_tag=$(echo "${MODEL}" | tr '.' '-' | tr '[:upper:]' '[:lower:]')

OBS=${OBS:-centralized}
# task_complete is off by default everywhere now, so nothing to pass and
# nothing to tag -- centralized and partial share the same terminations.

case "${OBS}" in
  centralized)
    obs_off_args=(--predict-acting-agent --train-get-image)
    # live_sim hardcodes agent prediction, so it takes no such flag
    obs_live_args=(--train-get-image)
    obs_tag=cent
    ;;
  partial)
    obs_off_args=(
      --partial-history --no-predict-acting-agent --train-get-image
      --partial-step-index-mode local --partial-observation-mode consume-once
    )
    obs_live_args=(
      --train-get-image --partial-history
      --partial-step-index-mode local --partial-observation-mode consume-once
    )
    # partial never offers task_complete anyway; keep the tag off to avoid
    # implying an axis that does not exist for this arm
    obs_tag=partial
    ;;
  *) echo "Unknown OBS: ${OBS}" >&2; exit 1 ;;
esac

if [[ "${MODE}" == "offsim" ]]; then
  for split in heldout_trajectories heldout_tasks; do
    echo "=== off-sim ${MODEL} on ${src_tag}/${split} ==="
    "${PYTHON_BIN}" -m training.bc_task_vlm.eval_standalone \
      --backend gemini \
      --model "${MODEL}" \
      --project "${GOOGLE_CLOUD_PROJECT}" \
      --location "${GOOGLE_CLOUD_LOCATION}" \
      "${obs_off_args[@]}" \
      --sft-format tool_call \
      --no-forced-json \
      --native-tools \
      --image-resolution 512 \
      --manifest "${manifest_dir}/eval_manifest_${split}.json" \
      --dataset-root "${dataset_root}" \
      --output-dir "training/bc_task_vlm/eval_runs/offsim_ots-${model_tag}-${obs_tag}-${src_tag}__${split}" \
      --resume
  done
else
  : "${SPLIT:?Set SPLIT for livesim}"
  echo "=== live-sim ${MODEL} on ${src_tag}/${SPLIT} ==="
  exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
    --backend gemini \
    --model "${MODEL}" \
    --project "${GOOGLE_CLOUD_PROJECT}" \
    --location "${GOOGLE_CLOUD_LOCATION}" \
    "${obs_live_args[@]}" \
    --sft-format tool_call \
    --no-forced-json \
    --manifest "${manifest_dir}/eval_manifest_${SPLIT}.json" \
    --dataset-root "${dataset_root}" \
    --output-dir "training/bc_task_vlm/eval_runs/livesim_ots-${model_tag}-${obs_tag}-${src_tag}__${SPLIT}" \
    --success-criterion fsm \
    --step-budget-factor 2.0 \
    --gl-backend egl \
    --resume
fi

echo "Done."
