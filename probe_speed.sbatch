#!/bin/bash
#SBATCH --job-name=probe-speed
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/probe-speed-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/probe-speed-%j.err

# ONE variable at a time against the job-266619 baseline.
#
#   baseline: PrepareCoffee + AddSugarCubes, 10 runs each, 4 workers,
#             base strategy, thinking null -> 20 traj, $1.3888, ~4 min
#
# Everything below is held identical to that except the single knob named by
# $PROBE. See GENERATION_SPEEDUP.md for the ranked levers and why reasoning
# tokens (21x the output tokens) are the thing worth attacking.
#
# Usage: sbatch --export=PROBE=think probe_speed.sbatch

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

PROBE=${PROBE:-think}
OUT=/work/umass/shlomo_umass/dbenhamougol_umass/probe_speed/$PROBE
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
TASKS="PrepareCoffee AddSugarCubes"

# Held fixed at the baseline unless the probe overrides them.
RUNS=10
WORKERS=4
EXTRA=()

case "$PROBE" in
  think)    EXTRA=(--thinking-level medium) ;;
  verbal)   EXTRA=(--sampling verbalized --verbalized-k 4); RUNS=3 ;;  # k=4 needs the strategy too
  verbal2)  EXTRA=(--sampling verbalized --verbalized-k 2); RUNS=5 ;;  # k=4 broke PrepareCoffee:
                                                                      # Vertex 400, response count
                                                                      # exceeds request limits
  workers)  WORKERS=12 ;;
  combined) EXTRA=(--thinking-level medium --sampling verbalized --verbalized-k 4); RUNS=3; WORKERS=12 ;;
  *) echo "unknown PROBE=$PROBE"; exit 1 ;;
esac

rm -rf "$OUT"
echo "PROBE=$PROBE runs=$RUNS workers=$WORKERS extra=${EXTRA[*]}"
START=$(date +%s)

"$PY" -m data_generation.task_level.generation.raw.cli \
  --tasks $TASKS --num-runs $RUNS --max-retries 5 --max-workers $WORKERS \
  --generation-timeout-sec 600 \
  --enable-validation \
  --summary-path "$OUT/summary.json" \
  --tick-format "${EXTRA[@]}" || echo "exited $?"

echo "PROBE $PROBE DONE in $(( $(date +%s) - START ))s"
