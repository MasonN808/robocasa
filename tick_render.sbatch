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
# GL is selected via --gl-backend / --gpu-ids, NOT via exported env vars.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

export PYTHONWARNINGS=ignore
# GL has to be egl in BOTH places, and the two are set differently:
#   - the PARENT imports mujoco at module scope (sweep_trajectories.py:116),
#     before any flag is read, so it needs MUJOCO_GL exported. Job 266761 left
#     it unset, mujoco defaulted to osmesa, and the import died with
#     "'NoneType' object has no attribute 'glGetError'".
#   - the WORKERS get MUJOCO_GL from --gl-backend. Job 266732 exported egl but
#     passed no flag, so workers got osmesa while inheriting
#     PYOPENGL_PLATFORM=egl: "Cannot use OSMesa rendering platform".
# Setting the env AND passing the flag is what makes the two agree.
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
IN=/work/umass/shlomo_umass/dbenhamougol_umass/tick_image
OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render

rm -rf "$OUT"

"$PY" scripts/sweep_trajectories.py \
  --input-dir "$IN" \
  --output-dir "$OUT" \
  --tasks prepare_coffee add_sugar_cubes \
  --gpu-ids 0 --gl-backend egl || echo "exited $?"

echo "TICK RENDER DONE"
find "$OUT" -name '*.png' | wc -l
