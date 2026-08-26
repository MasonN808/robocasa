#!/usr/bin/env bash
#SBATCH --job-name=tick-gen150-topup
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --array=0-52%5
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -uo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
[[ "${CONFIRM_SPEND:-}" == "yes" ]] || exit 2
export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true
root=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_partition_none_temp06_v1
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
tasks=(AddLemonToFish AddSugarCubes AlcoholServingPrep AlignSilverware ArrangeBreadBowl ArrangeDrinkware ArrangeTea BeverageOrganization BowlAndCup CandleCleanup CheeseMixing ClusterItemsForClearing ColorfulSalsa CondimentCollection CutBuffetPizza DateNight DeliverStraw DessertAssembly DisplayMeatVariety DistributeChicken GarnishCake GarnishCupcake GarnishPancake GatherMarinadeIngredients HotDogSetup LemonSeasoningFish LineUpCondiments MatchCupAndDrink MeatSkewerAssembly OrganizeCondiments PlateStoreDinner PortionHotDogs PortionYogurt PrepareCheeseStation PrepareCocktailStation PrepareCoffee PrepareDrinkStation PrepareSandwichStation PrepareSausageCheese PrepareSoupServing SeasoningSteak ServeMealJuice ServeSteak SetBowlsForSoup SetUpSpiceStation SetupBowls SetupButterPlate SetupSodaBowl SetupWineGlasses SpicyMarinade SweetenCoffee TongBuffetSetup VeggieDipPrep)
task="${tasks[${SLURM_ARRAY_TASK_ID}]}"
slug="$(${python_bin} -c 'import re,sys; x=re.sub(r"([A-Z]+)([A-Z][a-z])",r"\1_\2",sys.argv[1]); print(re.sub(r"([a-z0-9])([A-Z])",r"\1_\2",x).lower())' "${task}")"
task_dir="${root}/${slug}"
common=(--tasks "${task}" --num-runs 150 --random-start-location true --sampling structured_random --temperature 0.6 --model gemini-3-flash-preview --location global --thinking-level low --max-workers 8 --max-retries 10 --generation-timeout-sec 600 --enable-validation --tick-format --prompt-style simplified_v3 --retry-feedback-style targeted --partition-policy none)
for round in 1 2 3; do
  count="$(find "${task_dir}/trajectories" -maxdepth 1 -type f -name 'traj_*.json' 2>/dev/null | wc -l)"
  (( count >= 150 )) && { echo "${task}: ${count}/150 complete"; exit 0; }
  echo "${task}: resume round ${round}, ${count}/150"
  "${python_bin}" -m data_generation.task_level.generation.raw.cli "${common[@]}" --resume "${task_dir}" || true
done
count="$(find "${task_dir}/trajectories" -maxdepth 1 -type f -name 'traj_*.json' 2>/dev/null | wc -l)"
(( count >= 150 )) || { echo "${task}: incomplete ${count}/150" >&2; exit 1; }
