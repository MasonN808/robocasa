#!/usr/bin/env bash
#SBATCH --job-name=gen-beverage-repl
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
[[ "${CONFIRM_SPEND:-}" == "yes" ]]
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
out=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_beverage_replacement
mkdir -p "${out}/beverage_organization"
"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks BeverageOrganization \
  --num-runs 71 \
  --run-indices 70 \
  --random-start-location true \
  --sampling structured_random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 1 \
  --max-retries 10 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --tick-format \
  --summary-path "${out}/beverage_organization/summary.json" || true

"${python_bin}" - <<'PY'
import json
from pathlib import Path
p = Path("/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale70_beverage_replacement/beverage_organization/trajectories/traj_000070.json")
payload = json.loads(p.read_text())
assert payload.get("trajectory_id") == "traj_000070"
assert payload.get("validation", {}).get("is_valid") is True
print("validated sparse replacement:", p)
PY
