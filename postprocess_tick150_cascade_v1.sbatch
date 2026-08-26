#!/usr/bin/env bash
#SBATCH --job-name=post-tick150-cascade-v1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --array=0-52%24
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
raw_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_raw
image_root=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_image
cd "${repo}"
export PYTHONPATH="${repo}"
readarray -t task_fields < <("${python_bin}" - "${SLURM_ARRAY_TASK_ID}" <<'PY'
import re
import sys
from data_generation.task_level.tasks import supported_task_names
task = sorted(supported_task_names())[int(sys.argv[1])]
dataset_name = re.sub(r"(?<!^)(?=[A-Z])", "_", task).lower()
print(task)
print(dataset_name)
PY
)
task="${task_fields[0]}"
dataset_name="${task_fields[1]}"
exec "${python_bin}" -m data_generation.task_level.generation.image.cli \
  --dataset "${raw_root}/${dataset_name}/summary.json" \
  --output-dataset "${image_root}/${dataset_name}/summary.json" \
  --revalidate --disable-progress --workers 2
