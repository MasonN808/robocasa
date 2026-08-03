#!/bin/bash
#SBATCH --job-name=proto
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/proto-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/proto-%j.err
set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a; . /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env; set +a
unset GOOGLE_API_KEY || true
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export PYTHONWARNINGS=ignore
for F in baseline timeline ownership; do
  echo "=================== $F ==================="
  "$PY" prototype_generation_formats.py --format $F --n 6 || true
done
echo "PROTO DONE"
