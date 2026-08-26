#!/usr/bin/env bash
#SBATCH --job-name=p2d-gate
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

# LEVEL-0 VALIDITY GATE: replay each task's own expert trajectory through
# live-sim. If the expert plan cannot satisfy the FSM goal, the task is
# BROKEN, not hard, and must be fixed or excluded before any difficulty
# label is assigned. The oracle bypasses the legality gate by design, so
# this isolates 'achievable in sim' from 'permitted by the spec'.
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
MODEL=oracle

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

MANIFEST=training/bc_task_vlm/eval_manifests/reference_difficulty/eval_manifest_phase2d_occupancy.json
DATA_ROOT=$("${PYTHON_BIN}" -c "import json;print(json.load(open('${MANIFEST}'))['dataset_root'])")

# Deterministic task sharding: task i goes to shard i % NSHARDS.
TASKS=$("${PYTHON_BIN}" - <<PY
import json
m=json.load(open("${MANIFEST}"))
t=sorted(m["trajectory_ids_by_task"])
print(",".join(t[i] for i in range(len(t)) if i % ${NSHARDS} == ${SHARD}))
PY
)
# TASKS may be set explicitly to re-check a subset.
# TASKS_SUBSET_FILE re-checks a subset. A file, not a comma list: SLURM
# --export is itself comma-delimited and would split the list.
if [[ -n "${TASKS_SUBSET_FILE:-}" ]]; then TASKS="$(paste -sd, "${TASKS_SUBSET_FILE}")"; fi
echo "=== shard ${SHARD}/${NSHARDS}: ${TASKS}"

exec "${PYTHON_BIN}" -m training.bc_task_vlm.live_sim_eval \
  --backend oracle \
  --train-get-image \
  --partial-history \
  --partial-step-index-mode none \
  --partial-observation-mode consume-once \
  --concurrent-expert-replay \
  --sft-format tool_call \
  --no-forced-json \
  --manifest "${MANIFEST}" \
  --dataset-root "${DATA_ROOT}" \
  --output-dir "training/bc_task_vlm/eval_runs/oraclegate${OUT_TAG:-}__shard${SHARD}" \
  --success-criterion fsm \
  --step-budget-factor 2.0 --uniform-durations \
  --gl-backend egl \
  --tasks "${TASKS}" \
  --resume
