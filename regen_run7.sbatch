#!/bin/bash
#SBATCH --job-name=regen-r7
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-r7-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-r7-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a; . /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env; set +a
RUN=/work/umass/shlomo_umass/dbenhamougol_umass/regen_parent_fixture/run7
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
T="ArrangeBreadBowl ArrangeTea GatherMarinadeIngredients OrganizeCondiments CondimentCollection PrepareCheeseStation SetupWineGlasses ServeMealJuice PrepareDrinkStation SpicyMarinade SweetenCoffee"
"$PY" -m data_generation.task_level.pipeline.cli --phase 3 --resume "$RUN" \
  --num-runs 10 --workers 4 --max-retries 5 --generation-timeout-sec 600 --tasks $T
"$PY" -m data_generation.task_level.pipeline.cli --phase 4 --resume "$RUN" --workers 4 --tasks $T || true
"$PY" scripts/sweep_trajectories.py --input-dir "$RUN/phase4/pre_image" \
  --output-dir "$RUN/phase4/sweeps" --workers 4 --gl-backend egl --gpu-ids 0
echo "RUN7 DONE"
