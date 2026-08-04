#!/bin/bash
#SBATCH --job-name=tick-sim
#SBATCH --partition=rtx-devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-sim-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-sim-%j.err

# Does concurrent-FSM-valid imply valid SIM execution?
#
# The 18 trajectories accepted by the enforced gate (job 266141), replayed
# step-for-step in MuJoCo with --backend oracle. Nothing is regenerated and no
# step is altered by staging, so what runs here is exactly what the validator
# passed. Read-only use of live_sim_eval.py -- no edits to it.
#
# Three outcomes worth distinguishing in the results:
#   FSM-valid and sim-passes  -> the gate is sound
#   FSM-valid and sim-fails   -> the FSM is missing a physical constraint
#   sim-passes but hung/raced -> see add_sugar_cubes/traj_000005; a race
#                                resolves one way per run and proves little

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
BASE=/work/umass/shlomo_umass/dbenhamougol_umass/tick_sim_check
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
