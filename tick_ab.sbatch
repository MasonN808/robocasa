#!/bin/bash
#SBATCH --job-name=tick-ab
#SBATCH --partition=rtx-batch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-ab-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/tick-ab-%j.err

# A/B: can the model place its own coordination when it writes in execution
# order? Same tasks, same count, one run flat (waits derived) and one tick
# (waits written by the model).

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration

set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a

export PYTHONWARNINGS=ignore
unset MUJOCO_GL PYOPENGL_PLATFORM || true

OUT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_ab
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
TASKS="PrepareCoffee AddSugarCubes"

for ARM in flat tick; do
  FLAG=""
  [ "$ARM" = tick ] && FLAG="--tick-format"
  echo "=== ARM: $ARM ==="
  "$PY" -m data_generation.task_level.generation.raw.cli \
    --tasks $TASKS --num-runs 6 --max-retries 3 --max-workers 4 \
    --generation-timeout-sec 600 \
    --summary-path "$OUT/$ARM/summary.json" \
    $FLAG || echo "ARM $ARM exited $?"
done

echo "TICK AB DONE"
