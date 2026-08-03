#!/bin/bash
#SBATCH --job-name=regen-pf-r2
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-pf-r2-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-pf-r2-%j.err

# Regenerates trajectories for the 11 tasks that gained `parent_fixture`, so the
# rewritten Rule 2 (a field lookup instead of a fixture-type list) can be checked
# against what the generator actually emits.

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

RUN=/work/umass/shlomo_umass/dbenhamougol_umass/regen_parent_fixture/run2
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

"$PY" -m data_generation.task_level.pipeline.cli \
  --phase 3 --resume "$RUN" \
  --num-runs 10 --workers 4 --max-retries 5 \
  --generation-timeout-sec 600 \
  --tasks ArrangeBreadBowl

echo "PHASE3 DONE"
