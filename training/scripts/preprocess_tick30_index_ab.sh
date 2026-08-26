#!/usr/bin/env bash
set -euo pipefail

mode="${1:?Usage: $0 none|local}"
case "${mode}" in
  none|local) ;;
  *) echo "mode must be none or local" >&2; exit 1 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

python_bin="${PYTHON_BIN:-/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python}"
selection="data_analysis/analysis/held_out_task_selection/held_out_task_selection.json"
train_tasks="$(${python_bin} -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["train_tasks"]))' "${selection}")"

exec training/scripts/launch_cpu_preprocess.sh \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_concurrent \
  --output-dir "/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick30_train47_toolcall_idx${mode}" \
  --run-name "tick30-train47-toolcall-idx${mode}" \
  --train-tasks "${train_tasks}" \
  --val-tasks same_as_train \
  --no-pretokenize \
  --no-use-example-cache \
  --example-build-workers 16 \
  --sbatch-cpus-per-task 16 \
  --sbatch-account shlomo_umass \
  --python-bin "${python_bin}" \
  -- \
  --sft-format tool_call \
  --partial-history \
  --train-get-image \
  --partial-step-index-mode "${mode}" \
  --partial-observation-mode consume-once \
  --validation-split-mode same-task \
  --validation-trajectory-fraction 0.1 \
  --validation-min-trajectories-per-task 1
