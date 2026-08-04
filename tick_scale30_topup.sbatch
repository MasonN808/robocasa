#!/bin/bash
#SBATCH --job-name=tick30-top
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick30top-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick30top-%A_%a.err

# TOP-UP for the 14 tasks that finished short of 30 (22-29 each, 47 runs
# missing of 1590). Same OUT and the SAME CONFIG as job 267262 -- existing
# results are treated as cached, so this only attempts the pending runs, and
# keeping thinking uncapped avoids mixing two generation configs inside one
# corpus.
#
# These are the hard tail: every missing run already exhausted 5 retries once,
# so expect a low hit rate and a high cost per success. Retrying gives fresh
# samples (new variation keys), not a rerun of the same attempt.
#
# Sharded 9 ways because one sequential pass is 10-25 hours. Each shard owns a
# disjoint slice of the task list and writes into the SHARED output root -- the
# CLI puts each task in its own <OUT>/<task_name>/ directory, so shards never
# touch the same files. There is deliberately NO `rm -rf "$OUT"` here: in an
# array job that would wipe the siblings.
#
# Gate: concurrent FSM, LOCK_STEP only, validation ENFORCED so failures retry.
# --enable-validation is not optional -- the raw CLI defaults it OFF
# (parser.set_defaults(disable_validation=True)) and without it nothing retries
# and first-attempt junk is written to disk.
#
# All 53 are included. The four tasks that looked dead (HotDogSetup,
# PrepareSausageCheese, PrepareCheeseStation, PlateStoreDinner) were 17/17 valid
# in job 266947 -- they only ever failed in a corpus generated with retries off.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

ALL_TASKS=(
  BeverageOrganization
  CandleCleanup
  DisplayMeatVariety
  MatchCupAndDrink
  PlateStoreDinner
  PortionHotDogs
  PortionYogurt
  PrepareCheeseStation
  PrepareCocktailStation
  PrepareDrinkStation
  PrepareSoupServing
  ServeMealJuice
  SetUpSpiceStation
  SetupWineGlasses
)

NSHARDS=4
SHARD=${SLURM_ARRAY_TASK_ID:-0}
TASKS=()
for i in "${!ALL_TASKS[@]}"; do
  if [ $((i % NSHARDS)) -eq "$SHARD" ]; then TASKS+=("${ALL_TASKS[$i]}"); fi
done

echo "shard $SHARD/$NSHARDS -> ${#TASKS[@]} tasks: ${TASKS[*]}"
mkdir -p "$OUT"

"$PY" -m data_generation.task_level.generation.raw.cli \
  --tasks "${TASKS[@]}" --num-runs 30 --max-retries 5 --max-workers 6 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --summary-path "$OUT/summary_topup${SHARD}.json" \
  --tick-format || echo "exited $?"

echo "TICK SCALE30 SHARD $SHARD DONE"
ls "$OUT"/*/trajectories/*.json 2>/dev/null | wc -l
