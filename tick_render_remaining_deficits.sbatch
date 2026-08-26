#!/usr/bin/env bash
#SBATCH --job-name=render-tick100-remain
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
rendered=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render70_structured_random_concurrent
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
cd "${repo}"
export PYTHONWARNINGS=ignore MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
raw_roots=(
  /work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_deficit_replacements
  /work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_remaining_replacements
  /work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_final_hotdogs
)
processed_roots=(
  /work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_deficit_replacements_image
  /work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_remaining_replacements_image
  /work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_final_hotdogs_image
)
for root_index in 0 1 2; do
  raw="${raw_roots[root_index]}"
  processed="${processed_roots[root_index]}"
  for task_dir in "${raw}"/*/; do
    task="$(basename "${task_dir}")"
    "${python_bin}" -m data_generation.task_level.generation.image.cli \
      --dataset "${task_dir}/summary.json" \
      --output-dataset "${processed}/${task}/summary.json" \
      --revalidate --disable-progress --workers 8
  done
  mapfile -t tasks < <(find "${processed}" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)
  "${python_bin}" scripts/sweep_trajectories.py \
    --input-dir "${processed}" --output-dir "${rendered}" \
    --tasks "${tasks[@]}" --step-order concurrent --gpu-ids 0 --gl-backend egl
done
"${python_bin}" - <<'PY'
from pathlib import Path
base=Path('/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_concurrent')
sup=Path('/work/umass/shlomo_umass/dbenhamougol_umass/tick_render70_structured_random_concurrent')
bad=[]
for task in sorted(p.name for p in base.iterdir() if p.is_dir()):
    b=sum(p.is_dir() for p in (base/task).glob('traj_*'))
    s=sum(p.is_dir() for p in (sup/task).glob('traj_*'))
    if b+s != 100: bad.append((task,b,s))
assert not bad, bad
print('complete rendered preflight: all 53 tasks have exactly 100 trajectories')
PY
