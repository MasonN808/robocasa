#!/bin/bash
#SBATCH --job-name=zerodur
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/zerodur-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/zerodur-%j.err

# Does making observations ZERO-DURATION let genuinely broken plans through?
#
# 47 corpus trajectories were rejected after image injection (20 deadlock,
# 18 inert wait, 7 conflict, 2 post-goal). Every one was valid AS WRITTEN --
# they broke only because the injector charges scheduling time for cameras the
# model never asked for, and an agent's lock-step clock advances per call IT
# makes, so agents with different action counts drift apart.
#
# Zero-duration recovers 46 of 47 with 0 regressions. This is the test that
# matters: replay all 47 in MuJoCo, plus 25 controls that were fine either way.
#
#   47 execute cleanly  -> the rejections were instrumentation artifacts, keep
#   47 fail in sim      -> the rejections were right, drop them instead
#
# Read-only use of live_sim_eval.py. Staging copies steps verbatim.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
BASE=/work/umass/shlomo_umass/dbenhamougol_umass/zero_dur_check
OUT=$BASE/oracle_run

rm -rf "$OUT"

"$PY" -m training.bc_task_vlm.live_sim_eval \
  --backend oracle \
  --manifest "$BASE/manifest.json" \
  --dataset-root "$BASE/data" \
  --output-dir "$OUT" \
  --gl-backend egl \
  --render-size 256 || echo "exited $?"

echo "TICK SIM CHECK DONE"
