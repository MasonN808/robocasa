#!/usr/bin/env bash
#SBATCH --job-name=tick-gen150-access-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-4%5
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root="${OUTPUT_ROOT:-/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_balanced_access_v1}"
num_runs="${NUM_RUNS:-150}"
preflight="${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json"
cd "${repo}"

[[ "${CONFIRM_SPEND:-}" == "yes" ]]
[[ -s "${preflight}" ]]
"${python_bin}" -c 'import json,sys; p=json.load(open(sys.argv[1])); assert p["num_runs_per_task"] == 150; assert len(p["tasks"]) == 53; assert all(t["balanced"] for t in p["tasks"])' "${preflight}"

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
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
  if (( index % num_shards == shard )); then tasks+=("${all_tasks[index]}"); fi
done

mkdir -p "${output_root}"
"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks "${tasks[@]}" \
  --num-runs "${num_runs}" \
  --random-start-location true \
  --random-access-state true \
  --sampling structured_random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 8 \
  --max-retries 10 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --tick-format \
  --prompt-style simplified_v3 \
  --retry-feedback-style targeted \
  --partition-policy none \
  --summary-path "${output_root}/summary_shard${shard}.json"
