#!/bin/bash
#SBATCH --job-name=tick-render
#SBATCH --partition=rtx-devel
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-render-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-render-%j.err

# Step 4: actually render the observations the tick injector inserted.
#
# Nothing has hit the renderer on this path yet. The post-processed records
# validate and their image_paths are internally consistent, but consistent
# references are not pixels -- this is what proves the paths, the view sets and
# the per-agent camera selection survive a real sim execution.
#
# The 20 pilot trajectories (job 266619) after post-processing with
# --revalidate: 20/20 is_valid, 336 get_image steps, tick_rows preserved.
#
# egl, not osmesa: osmesa dies with 'NoneType' object has no attribute
# 'glGetError'.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

export PYTHONWARNINGS=ignore
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
IN=/work/umass/shlomo_umass/dbenhamougol_umass/tick_image
OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render

rm -rf "$OUT"

"$PY" scripts/sweep_trajectories.py \
  --input-dir "$IN" \
  --output-dir "$OUT" \
  --tasks prepare_coffee add_sugar_cubes || echo "exited $?"

echo "TICK RENDER DONE"
find "$OUT" -name '*.png' | wc -l
