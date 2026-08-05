#!/bin/bash
#SBATCH --job-name=cov-ab
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/cov-ab-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/cov-ab-%j.err

# Work-partition coverage, before vs after the deterministic weighting.
#
# Both corpora use the same traj_*/original_trajectory.json layout, so they run
# through ONE probe rather than two -- an earlier comparison used two different
# probes and the percentages were not comparable.
#
# Output MUST land on /work: the scratchpad is node-local, so writing there
# succeeds on the login node and then fails on the compute node.
#
#   sbatch coverage_ab.sbatch
#   python coverage_report.py $OUT/cov_before.json $OUT/cov_after.json

set -u
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
export PYTHONWARNINGS=ignore

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
OUT=${OUT:-/work/umass/shlomo_umass/dbenhamougol_umass/cov_ab}
BEFORE=${BEFORE:-/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_agentsft_subset}
AFTER=${AFTER:-/work/umass/shlomo_umass/dbenhamougol_umass/tick_staged/data}

mkdir -p "$OUT"

echo "=== BEFORE $BEFORE ==="
"$PY" coverage_work_partitions.py --data "$BEFORE" --out "$OUT/cov_before.json" >/dev/null

echo "=== AFTER $AFTER ==="
"$PY" coverage_work_partitions.py --data "$AFTER" --out "$OUT/cov_after.json" >/dev/null

"$PY" coverage_report.py "$OUT/cov_before.json" "$OUT/cov_after.json" \
  --labels before,after

echo COV DONE
