#!/bin/bash
#SBATCH --job-name=armA
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/armA-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/armA-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a; . /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env; set +a
RUN=/work/umass/shlomo_umass/dbenhamougol_umass/regen_parent_fixture/armA
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
"$PY" -m data_generation.task_level.pipeline.cli --phase 3 --resume "$RUN" \
  --sampling structured_random --num-runs 10 --workers 6 --max-retries 8 \
  --generation-timeout-sec 600 \
  --tasks ArrangeBreadBowl SetupWineGlasses GatherMarinadeIngredients \
          OrganizeCondiments PrepareCheeseStation SweetenCoffee
echo "ARM A DONE"
