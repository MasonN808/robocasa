#!/usr/bin/env bash
#SBATCH --job-name=watch-scale-evals
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:30:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set +e
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
"${python_bin}" -m training.bc_task_vlm.watch_scale_checkpoints \
  --watch-config training/bc_task_vlm/eval_runs/fixed_live_sim_scale_checkpoint_watch/watch_config.json \
  --poll-seconds 60 \
  --max-runtime-hours 23
status=$?
set -e
if [[ ${status} -eq 75 ]]; then
  successor=$(sbatch --parsable watch_scale_checkpoint_evals.sbatch)
  echo "queued successor watcher ${successor}"
  exit 0
fi
exit "${status}"
