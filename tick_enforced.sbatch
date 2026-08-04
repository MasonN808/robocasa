#!/bin/bash
#SBATCH --job-name=tick-enf
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-enf-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-enf-%j.err

# Tick generation with the gate ACTUALLY ENFORCED. Job 265805 ran with the
# raw CLI's default (parser.set_defaults(disable_validation=True)), so the
# concurrent validator reported but never raised and nothing was retried --
# "8/20 valid" was first-attempt quality, not a retry-converged rate.
#
# --enable-validation makes a failed candidate raise, which is what drives the
# retry loop in on_demand_generation (validation errors are retryable).
#
# Since 265805 the prompt also gained: releases must carry a message, and the
# release must sit on a strictly later row than the wait.
#
# Fresh output dir: result.json files in an existing one are treated as cached.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_enforced
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
TASKS="PrepareCoffee AddSugarCubes"

rm -rf "$OUT"

"$PY" -m data_generation.task_level.generation.raw.cli \
  --tasks $TASKS --num-runs 10 --max-retries 5 --max-workers 4 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --summary-path "$OUT/summary.json" \
  --tick-format || echo "exited $?"

echo "TICK ENFORCED DONE"
