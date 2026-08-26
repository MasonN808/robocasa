#!/usr/bin/env bash
#SBATCH --job-name=gen-final-hotdogs
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
[[ "${CONFIRM_SPEND:-}" == "yes" ]]
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_final_hotdogs
mkdir -p "${root}/portion_hot_dogs"
"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks PortionHotDogs --num-runs 78 --run-indices 77 \
  --random-start-location true --sampling structured_random --temperature 0.6 \
  --model gemini-3-flash-preview --location global --thinking-level low \
  --max-workers 1 --max-retries 15 --generation-timeout-sec 600 \
  --enable-validation --tick-format \
  --summary-path "${root}/portion_hot_dogs/summary.json" || true
"${python_bin}" - <<'PY'
import json
from pathlib import Path
p=Path('/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_final_hotdogs/portion_hot_dogs/trajectories/traj_000077.json')
d=json.loads(p.read_text())
assert d.get('validation',{}).get('is_valid') is True
print('validated final PortionHotDogs replacement:',p)
PY
