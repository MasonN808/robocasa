#!/usr/bin/env bash
#SBATCH --job-name=prep-tick30-v3-none
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
py=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
selection="${repo}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json"
cd "${repo}"
train_tasks="$(${py} -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["train_tasks"]))' "${selection}")"
exec training/scripts/launch_cpu_preprocess.sh --no-sbatch \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_revalidated_v3 \
  --output-dir /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick30_revalidated_v3_train47_toolcall_idxnone_schemafix \
  --run-name tick30-revalidated-v3-train47-toolcall-idxnone \
  --train-tasks "${train_tasks}" --val-tasks same_as_train --no-pretokenize \
  --no-use-example-cache --example-build-workers 24 --python-bin "${py}" -- \
  --sft-format tool_call --partial-history --train-get-image \
  --partial-step-index-mode none --partial-observation-mode consume-once \
  --validation-split-mode same-task --validation-trajectory-fraction 0.1 \
  --validation-split-seed 42 --validation-min-trajectories-per-task 3
