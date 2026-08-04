#!/bin/bash
#SBATCH --job-name=tick-rows
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-rows-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-rows-%j.err

# Step 1 of the image-injection rollout: regenerate the two pilot tasks so the
# saved records actually carry `tick_rows`.
#
# Everything before this ran with the rows being dropped twice on the way to
# disk -- once by the validator's normalized_candidate, once by
# build_trajectory_record, both of which rebuild from a fixed field list. The
# flat steps cannot be turned back into rows, so `tick_image_injection_check.py`
# had to reconstruct a schedule with --backfill-rows, which tests the injector
# but not the model's own coordination.
#
# What this run is for: real rows on disk, so step 2 can run the injection
# check with no backfill and mean something.
#
# Same settings as job 266141 (18/18 valid, 3 retries recovered), so the
# validity rate is also a regression check on the persistence change.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_rows_check
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
TASKS="PrepareCoffee AddSugarCubes"

rm -rf "$OUT"

"$PY" -m data_generation.task_level.generation.raw.cli \
  --tasks $TASKS --num-runs 10 --max-retries 5 --max-workers 4 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --summary-path "$OUT/summary.json" \
  --tick-format || echo "exited $?"

echo "TICK ROWS CHECK DONE"
