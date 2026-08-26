#!/usr/bin/env bash
#SBATCH --job-name=cabparent-sweep
#SBATCH --partition=rtx-batch
#SBATCH --array=0-74%12
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONUNBUFFERED=1
export NUMBA_CACHE_DIR=/tmp/robocasa_numba_cache_${SLURM_JOB_ID}
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root=training/bc_task_vlm/eval_runs/cabinet_parent_relationship_scene_sweep_v2

layouts=(11 15 18 40 50)
styles=(14 28 34 46 58)
seeds=(42 99 7)
layout_index=$((SLURM_ARRAY_TASK_ID / 15))
remainder=$((SLURM_ARRAY_TASK_ID % 15))
style_index=$((remainder / 3))
seed_index=$((remainder % 3))
layout=${layouts[$layout_index]}
style=${styles[$style_index]}
seed=${seeds[$seed_index]}
out=${output_root}/layout_${layout}_style_${style}_seed_${seed}

exec "${python_bin}" -m training.bc_task_vlm.audit_cabinet_parent_relationships \
  --layout "${layout}" --style "${style}" --seed "${seed}" --output-dir "${out}"
