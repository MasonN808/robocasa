#!/usr/bin/env bash
#SBATCH --job-name=gen-tick100-deficit
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-5%6
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err

set -euo pipefail
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
[[ "${CONFIRM_SPEND:-}" == "yes" ]]

python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick_scale100_deficit_replacements
tasks=(CandleCleanup MatchCupAndDrink PlateStoreDinner PortionHotDogs PrepareCheeseStation PrepareDrinkStation)
starts=(70 70 70 71 70 70)
counts=(3 1 1 4 1 1)
task="${tasks[${SLURM_ARRAY_TASK_ID}]}"
start="${starts[${SLURM_ARRAY_TASK_ID}]}"
count="${counts[${SLURM_ARRAY_TASK_ID}]}"
end=$((start + count - 1))
num_runs=$((end + 1))
indices=()
for ((index=start; index<=end; index++)); do indices+=("${index}"); done
task_dir="$(${python_bin} -c 'import re,sys; x=re.sub(r"([A-Z]+)([A-Z][a-z])",r"\1_\2",sys.argv[1]); print(re.sub(r"([a-z0-9])([A-Z])",r"\1_\2",x).lower())' "${task}")"
mkdir -p "${output_root}/${task_dir}"

"${python_bin}" -m data_generation.task_level.generation.raw.cli \
  --tasks "${task}" \
  --num-runs "${num_runs}" \
  --run-indices "${indices[@]}" \
  --random-start-location true \
  --sampling structured_random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers "${count}" \
  --max-retries 10 \
  --generation-timeout-sec 600 \
  --enable-validation \
  --tick-format \
  --summary-path "${output_root}/${task_dir}/summary.json" || true

"${python_bin}" - "${output_root}/${task_dir}" "${start}" "${end}" <<'PY'
import json, sys
from pathlib import Path
root, start, end = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
for index in range(start, end + 1):
    path = root / "trajectories" / f"traj_{index:06d}.json"
    payload = json.loads(path.read_text())
    assert payload.get("validation", {}).get("is_valid") is True, path
print(f"validated replacements {start}..{end} in {root}")
PY
