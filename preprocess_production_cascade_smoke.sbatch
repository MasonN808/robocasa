#!/usr/bin/env bash
#SBATCH --job-name=prep-prod-cascade-smoke
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
cd "${repo}"
exec training/scripts/launch_cpu_preprocess.sh --no-sbatch \
  --dataset-root training/bc_task_vlm/eval_runs/production_cascade_smoke_rendered \
  --output-dir training/bc_task_vlm/eval_runs/production_cascade_smoke_preprocessed \
  --run-name production-cascade-smoke-noindex \
  --train-tasks arrange_bread_bowl,prepare_coffee,set_up_spice_station \
  --val-tasks hot_dog_setup,garnish_cake,meat_skewer_assembly \
  --no-pretokenize --no-use-example-cache \
  --example-build-workers 8 --python-bin "${python_bin}" -- \
  --sft-format tool_call --partial-history --train-get-image \
  --partial-step-index-mode none --partial-observation-mode consume-once \
  --validation-split-mode task-holdout --validation-trajectory-fraction 0 \
  --validation-split-seed 42 --validation-min-trajectories-per-task 0
