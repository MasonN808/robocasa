#!/usr/bin/env bash
#SBATCH --job-name=ref-diff
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Reference-difficulty sweep: ALL tasks x 10 trajectories under ONE untrained
# policy, used to assign task difficulty levels and failure-regime tags.
#
# Untrained on purpose: measured difficulty then reflects the task rather than
# one fine-tune's quirks, and it sidesteps malformed generation, which is
# trained-only (7.99% of steps vs 0.00%) and would otherwise contaminate every
# label. Train/eval contamination is meaningless for a model that saw none of
# the training data, so trajectories are sampled from the full pool.
#
# Settings that differ from earlier sweeps, all deliberate:
#   --rejection-mode report-failed   (now the default) adaptation 9.4% -> 94.1%
#   --partial-step-index-mode none   step indices removed
#   partial observability only       centralized is out of scope
#
#   sbatch --export=ALL,SHARD=0,NSHARDS=8 ref_difficulty_sweep.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

: "${SHARD:?Set SHARD}"
: "${NSHARDS:?Set NSHARDS}"
MODEL=${MODEL:-gemini-3-flash-preview}

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1
export PYTHONWARNINGS=ignore
export GOOGLE_APPLICATION_CREDENTIALS=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/config/gcloud/application_default_credentials.json
export GOOGLE_CLOUD_PROJECT=access-cis260215-538827
export GOOGLE_CLOUD_LOCATION=global
unset GEMINI_API_KEY GOOGLE_API_KEY || true

MANIFEST=training/bc_task_vlm/eval_manifests/reference_difficulty/eval_manifest_all_tasks.json
DATA_ROOT=$("${PYTHON_BIN}" -c "import json;print(json.load(open('${MANIFEST}'))['dataset_root'])")

# Deterministic task sharding: task i goes to shard i % NSHARDS.
TASKS=$("${PYTHON_BIN}" - <<PY
import json
m=json.load(open("${MANIFEST}"))
t=sorted(m["trajectory_ids_by_task"])
print(",".join(t[i] for i in range(len(t)) if i % ${NSHARDS} == ${SHARD}))
PY
)
echo "=== shard ${SHARD}/${NSHARDS}: ${TASKS}"

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
  --backend gemini \
  --model "${MODEL}" \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --location "${GOOGLE_CLOUD_LOCATION}" \
  --train-get-image \
  --partial-history \
  --partial-step-index-mode none \
  --partial-observation-mode consume-once \
  --sft-format tool_call \
  --no-forced-json \
  --manifest "${MANIFEST}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "training/bc_task_vlm/eval_runs/refdiff_${MODEL}__shard${SHARD}" \
  --success-criterion fsm \
  --step-budget-factor 2.0 \
  --gl-backend egl \
  --tasks "${TASKS}" \
  --resume
