#!/bin/bash
#SBATCH --job-name=regen-full
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-full-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/regen-full-%j.err

# Full pool with wait_for_signal: all 52 tasks x 30 trajectories.
# --max-retries raised from 5: the announce+release+id contract roughly doubled
# wait-rule rejections, and two tasks already sat at 4 retries on the 11-task run.

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a; . /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env; set +a
RUN=/work/umass/shlomo_umass/dbenhamougol_umass/regen_parent_fixture/full
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

"$PY" -m data_generation.task_level.pipeline.cli --phase 3 --resume "$RUN" --sampling structured_random \
  --num-runs 30 --workers 6 --max-retries 8 --generation-timeout-sec 600

"$PY" -m data_generation.task_level.pipeline.cli --phase 4 --resume "$RUN" --workers 6 || true

"$PY" scripts/sweep_trajectories.py --input-dir "$RUN/phase4/pre_image" \
  --output-dir "$RUN/phase4/sweeps" --workers 6 --gl-backend egl --gpu-ids 0
echo "FULL DONE"
