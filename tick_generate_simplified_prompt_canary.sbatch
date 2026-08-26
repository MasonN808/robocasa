#!/usr/bin/env bash
#SBATCH --job-name=tick-simple-canary
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root="${OUTPUT_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_simplified_prompt_canary}"
prompt_style="${PROMPT_STYLE:-simplified_v3}"
retry_feedback_style="${RETRY_FEEDBACK_STYLE:-targeted}"
temperature="${TEMPERATURE:-0.6}"
partition_policy="${PARTITION_POLICY:-weighted}"
num_runs="${NUM_RUNS:-2}"
cd "${repo}"

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
[[ "${CONFIRM_SPEND:-}" == "yes" ]]

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

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
shard="${SLURM_ARRAY_TASK_ID}"
tasks=()
for index in "${!all_tasks[@]}"; do
  if (( index % num_shards == shard )); then
    tasks+=("${all_tasks[index]}")
  fi
done

mkdir -p "${output_root}"
echo "${prompt_style}/${retry_feedback_style} partition=${partition_policy} temp=${temperature} canary shard ${shard}: ${#tasks[@]} tasks x ${num_runs} trajectories"

"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks "${tasks[@]}" \
  --num-runs "${num_runs}" \
  --random-start-location true \
  --sampling structured_random \
  --temperature "${temperature}" \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 8 \
  --max-retries 10 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --tick-format \
  --prompt-style "${prompt_style}" \
  --retry-feedback-style "${retry_feedback_style}" \
  --partition-policy "${partition_policy}" \
  --summary-path "${output_root}/summary_shard${shard}.json"

echo "simplified-prompt canary shard ${shard} finished at $(date --iso-8601=seconds)"
