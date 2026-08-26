#!/usr/bin/env bash
#SBATCH --job-name=tick70-sr
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-4
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick70-sr-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick70-sr-%A_%a.err

# Generate the additional tick corpus with the measured cheap configuration:
# structured_random, thinking=low, one trajectory per API call. Five shards
# with 12 workers each cap aggregate API concurrency at roughly 60 requests.
#
# The existing corpus has 30 trajectories for 51 tasks and 29 for
# PortionHotDogs and PortionYogurt. Those two receive 71 new samples; all
# others receive 70, yielding exactly 100 trajectories/task after combining.

set -euo pipefail

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

if [[ "${CONFIRM_SPEND:-}" != "yes" ]]; then
  echo "Refusing paid generation without CONFIRM_SPEND=yes." >&2
  exit 2
fi

output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_structured_random_low
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

all_tasks=(
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

num_shards=5
shard="${SLURM_ARRAY_TASK_ID:-0}"
regular_tasks=()
short_tasks=()

for index in "${!all_tasks[@]}"; do
  if (( index % num_shards != shard )); then
    continue
  fi
  task="${all_tasks[index]}"
  if [[ "${task}" == "PortionHotDogs" || "${task}" == "PortionYogurt" ]]; then
    short_tasks+=("${task}")
  else
    regular_tasks+=("${task}")
  fi
done

mkdir -p "${output_root}"

common_args=(
  --random-start-location true
  --sampling structured_random
  --temperature 0.6
  --model gemini-3-flash-preview
  --location global
  --thinking-level low
  --max-workers 12
  --max-retries 5
  --generation-timeout-sec 600
  --enable-validation
  --tick-format
)

echo "shard ${shard}/${num_shards}: ${#regular_tasks[@]} tasks x 70; ${#short_tasks[@]} tasks x 71"

overall_status=0

if (( ${#regular_tasks[@]} > 0 )); then
  if ! "${python_bin}" -m data_generation.task_level.generation.raw.cli \
    --tasks "${regular_tasks[@]}" \
    --num-runs 70 \
    --summary-path "${output_root}/summary_shard${shard}.json" \
    "${common_args[@]}"; then
    # The CLI returns nonzero when some runs exhaust validation retries. That
    # must not prevent the independent 71-run group below from being attempted.
    overall_status=1
  fi
fi

if (( ${#short_tasks[@]} > 0 )); then
  if ! "${python_bin}" -m data_generation.task_level.generation.raw.cli \
    --tasks "${short_tasks[@]}" \
    --num-runs 71 \
    --summary-path "${output_root}/summary_short_shard${shard}.json" \
    "${common_args[@]}"; then
    overall_status=1
  fi
fi

echo "tick structured-random shard ${shard} finished at $(date --iso-8601=seconds)"
exit "${overall_status}"
