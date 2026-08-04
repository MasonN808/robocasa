#!/bin/bash
#SBATCH --job-name=tick-ab2
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-ab2-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-ab2-%j.err

# Re-run of job 264786 against the corrected TICK_FORMAT_RULES. That run
# measured a prompt that no longer exists: the release obligation, both
# adjacencies (ask->wait, leave->report), the wait-before-the-handoff
# ordering and the resume-after-the-release rule have all been added since.
#
# A FRESH output dir is mandatory -- result.json files in an existing one are
# treated as cached and silently returned instead of regenerating.

cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_ab2
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
TASKS="PrepareCoffee AddSugarCubes"

rm -rf "$OUT"

for ARM in flat tick; do
  FLAG=""
  [ "$ARM" = tick ] && FLAG="--tick-format"
  echo "=== ARM: $ARM ==="
  "$PY" -m data_generation.task_level.generation.raw.cli \
    --tasks $TASKS --num-runs 10 --max-retries 3 --max-workers 4 \
    --generation-timeout-sec 600 \
    --summary-path "$OUT/$ARM/summary.json" \
    $FLAG || echo "ARM $ARM exited $?"
done

echo "TICK AB2 DONE"
