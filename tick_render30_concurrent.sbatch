#!/bin/bash
#SBATCH --job-name=tick-render30c
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-render30c-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-render30c-%A_%a.err

# Re-render of the whole corpus in CONCURRENT-EXECUTOR order.
#
# The first render (tick_render30) walked each trajectory's flat step list top
# to bottom. That is not the order the executor runs it in: the executor ticks a
# clock and, within a tick, acts in agent-id order. Measured over the corpus,
# 20.5% of observations had a different set of actions applied before them under
# the two orders, in 96.8% of trajectories -- so a fifth of the training images
# showed a world the model will not see at that point. --step-order concurrent
# fixes that; the stored trajectories are untouched, only the run order changes.
#
# This pass also picks up the 38 top-up trajectories that the first render
# predated (see tick_postprocess_all.sbatch).
#
# GL has to be egl in BOTH places, and the two are set differently:
#   - the PARENT imports mujoco at module scope, before any flag is read, so it
#     needs MUJOCO_GL exported.
#   - the WORKERS get MUJOCO_GL from --gl-backend.
# Setting the env AND passing the flag is what makes the two agree.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
IN=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale30_image
OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_concurrent

# no rm -rf: array shards share $OUT and would wipe each other
mkdir -p "$OUT"

"$PY" scripts/sweep_trajectories.py \
  --input-dir "$IN" \
  --output-dir "$OUT" \
  --tasks $(ls "$IN" | awk "NR % 8 == ${SLURM_ARRAY_TASK_ID:-0}") \
  --step-order concurrent \
  --gpu-ids 0 --gl-backend egl || echo "exited $?"

echo "TICK RENDER CONCURRENT DONE"
echo "trajectories: $(find "$OUT" -maxdepth 2 -name 'traj_*' -type d | wc -l)"
echo "images:       $(find "$OUT" -name '*.jpg' | wc -l)"
