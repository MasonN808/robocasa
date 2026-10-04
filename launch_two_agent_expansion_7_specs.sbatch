#!/usr/bin/env bash
#SBATCH --job-name=specs-2a-7
#SBATCH --partition=rtx-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

# Use Vertex AI and the same credentials as the existing generation jobs.
unset GOOGLE_API_KEY || true
export GOOGLE_APPLICATION_CREDENTIALS=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/config/gcloud/application_default_credentials.json
export GOOGLE_CLOUD_PROJECT=access-cis260215-538827
export GOOGLE_CLOUD_LOCATION=global
export GOOGLE_GENAI_USE_VERTEXAI=True
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONUNBUFFERED=1

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
RUN=data_generation/task_level/data/pipeline_runs/two_agent_expansion_7_v1/20260826T035429Z_all_batches
TASKS=(
  GatherCuttingTools
  RestockBowls
  PastryDisplay
  PlaceEqualIceCubes
  ArrangeUtensilsByType
  FreezeBottledWaters
  PlaceDishesBySink
)

# Guard against regressing to SimToolExecutor's OSMesa default before making
# any new model calls.
"$PY" -m unittest discover -s tests -p test_sim_normalization.py
"$PY" -m unittest discover -s tests -p test_task_level_phase2.py
"$PY" -m unittest discover -s tests -p test_task_level_phase3.py

"$PY" -m data_generation.task_level.pipeline.cli \
  --phase 2 2.5 \
  --resume "$RUN" \
  --tasks "${TASKS[@]}" \
  --workers 4 \
  --rating-threshold MEDIUM \
  --phase1-sim-normalization \
  --phase2-sim-alignment \
  --phase2-concurrent-canary \
  --phase2-repair-retries 2 \
  --phase2-5-repair-retries 2 \
  --project "$GOOGLE_CLOUD_PROJECT" \
  --location "$GOOGLE_CLOUD_LOCATION"
