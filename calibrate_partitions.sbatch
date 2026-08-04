#!/bin/bash
#SBATCH --job-name=cal-parts
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/cal-parts-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/cal-parts-%j.err

# Measures how reliably the generator can produce each work partition, so the
# per-split retry budget is set from data rather than guessed. Retries are off
# inside each attempt: we want the raw per-attempt success rate.
#
# Resumable -- a (task, split) pair with a result.json is skipped, so this can
# be run in slices and topped up.

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

OUT=${OUT:-/work/umass/shlomo_umass/dbenhamougol_umass/partition_calibration}
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

"$PY" calibrate_partitions.py --out "$OUT" --attempts "${ATTEMPTS:-5}" ${EXTRA:-}

echo "CALIBRATION DONE"
