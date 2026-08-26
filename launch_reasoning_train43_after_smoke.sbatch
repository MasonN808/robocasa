#!/usr/bin/env bash
#SBATCH --job-name=launch-rsn-t43-full
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=23:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
while true; do
  set +e
  output=$(sbatch --parsable train_tick150_reasoning_scale30_train43.sbatch 2>&1)
  status=$?
  set -e
  if [[ ${status} -eq 0 ]]; then break; fi
  echo "full training submission unavailable; retrying in 5 minutes: ${output}" >&2
  sleep 300
done
full_job=${output%%;*}
eval_launcher=$(sbatch --parsable --dependency="afterok:${full_job}" launch_reasoning_train43_evals.sbatch)
echo "full_training=${full_job} eval_launcher=${eval_launcher%%;*}"
