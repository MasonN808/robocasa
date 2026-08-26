#!/usr/bin/env bash
#SBATCH --job-name=tick150-cascade-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --array=0-52%24
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
output_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_raw
cd "${repo}"
[[ "${CONFIRM_SPEND:-}" == yes ]]
set -a
. /work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/.env
set +a
export PYTHONPATH="${repo}"
task="$(${python_bin} - "${SLURM_ARRAY_TASK_ID}" <<'PY'
import sys
from data_generation.task_level.tasks import supported_task_names
print(sorted(supported_task_names())[int(sys.argv[1])])
PY
)"
exec "${python_bin}" -m data_generation.task_level.generation.raw.production_cascade \
  --task "${task}" --num-runs 150 --output-root "${output_root}"
