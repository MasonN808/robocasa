#!/bin/bash
#SBATCH --job-name=tick30
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-8
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick30-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick30-%A_%a.err

# The scale-out: 30 tick trajectories per task, all 53 verified specs.
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
  AddLemonToFish AddSugarCubes AlcoholServingPrep AlignSilverware ArrangeBreadBowl
  ArrangeDrinkware ArrangeTea BeverageOrganization BowlAndCup CandleCleanup
  CheeseMixing ClusterItemsForClearing ColorfulSalsa CondimentCollection CutBuffetPizza
  DateNight DeliverStraw DessertAssembly DisplayMeatVariety DistributeChicken
  GarnishCake GarnishCupcake GarnishPancake GatherMarinadeIngredients HotDogSetup
  LemonSeasoningFish LineUpCondiments MatchCupAndDrink MeatSkewerAssembly OrganizeCondiments
  PlateStoreDinner PortionHotDogs PortionYogurt PrepareCheeseStation PrepareCocktailStation
  PrepareCoffee PrepareDrinkStation PrepareSandwichStation PrepareSausageCheese PrepareSoupServing
  SeasoningSteak ServeMealJuice ServeSteak SetBowlsForSoup SetUpSpiceStation
  SetupBowls SetupButterPlate SetupSodaBowl SetupWineGlasses SpicyMarinade
  SweetenCoffee TongBuffetSetup VeggieDipPrep
)

NSHARDS=9
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
  --summary-path "$OUT/summary_shard${SHARD}.json" \
  --tick-format || echo "exited $?"

echo "TICK SCALE30 SHARD $SHARD DONE"
ls "$OUT"/*/trajectories/*.json 2>/dev/null | wc -l
