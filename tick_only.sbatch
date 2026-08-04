#!/bin/bash
#SBATCH --job-name=tick-only
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-only-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-only-%j.err

# Tick-format generation only, same tasks and --num-runs as the tick arm of
# job 265355, so the only difference is the two fixes since:
#   * `about`/`releases` must name an id from initial_state (265355 failed
#     all 20 on invented ids like mug_placed)
#   * tick candidates are gated by ConcurrentTaskValidator, so a collision or
#     a hung wait forces a retry instead of being written as valid
#
# A FRESH output dir is mandatory: result.json files in an existing one are
# treated as cached and returned instead of regenerating.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_only
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
TASKS="PrepareCoffee AddSugarCubes"

rm -rf "$OUT"

"$PY" -m data_generation.task_level.generation.raw.cli \
  --tasks $TASKS --num-runs 10 --max-retries 3 --max-workers 4 \
  --generation-timeout-sec 600 \
  --summary-path "$OUT/summary.json" \
  --tick-format || echo "exited $?"

echo "TICK ONLY DONE"
