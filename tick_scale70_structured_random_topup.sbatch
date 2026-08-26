#!/usr/bin/env bash
#SBATCH --job-name=tick70-sr-top
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=12
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --array=0-52%5
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick70-sr-top-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick70-sr-top-%A_%a.err

# Isolated top-up: one task per array element, with at most five array elements
# active at once (5 x 12 = 60 API workers globally). A task that exhausts one
# five-retry CLI round is resumed for another round. Failure of one task cannot
# skip or terminate work for any other task.

set -uo pipefail

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
max_rounds="${TOPUP_ROUNDS:-5}"

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

task="${all_tasks[${SLURM_ARRAY_TASK_ID:-0}]}"
task_dir_name="$("${python_bin}" -c \
  'import re, sys; value=re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", sys.argv[1]); print(re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", value).lower())' \
  "${task}")"
task_dir="${output_root}/${task_dir_name}"
summary_path="${task_dir}/summary.json"

target=70
if [[ "${task}" == "PortionHotDogs" || "${task}" == "PortionYogurt" || "${task}" == "BeverageOrganization" ]]; then
  target=71
fi

common_args=(
  --tasks "${task}"
  --num-runs "${target}"
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

count_trajectories() {
  if [[ -d "${task_dir}/trajectories" ]]; then
    find "${task_dir}/trajectories" -maxdepth 1 -type f -name '*.json' | wc -l
  else
    echo 0
  fi
}

for (( round=1; round<=max_rounds; round++ )); do
  count="$(count_trajectories)"
  if (( count >= target )); then
    echo "${task}: complete at ${count}/${target}; no top-up needed"
    exit 0
  fi

  echo "${task}: top-up round ${round}/${max_rounds}, currently ${count}/${target}"
  if [[ -f "${summary_path}" ]]; then
    "${python_bin}" -m data_generation.task_level.generation.raw.cli \
      "${common_args[@]}" --resume "${task_dir}" || true
  else
    mkdir -p "${task_dir}"
    "${python_bin}" -m data_generation.task_level.generation.raw.cli \
      "${common_args[@]}" --summary-path "${summary_path}" || true
  fi
done

count="$(count_trajectories)"
if (( count >= target )); then
  echo "${task}: complete at ${count}/${target}"
  exit 0
fi

echo "${task}: still incomplete at ${count}/${target} after ${max_rounds} top-up rounds" >&2
exit 1
