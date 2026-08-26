#!/usr/bin/env bash
#SBATCH --job-name=prep-tick150-reasoning
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=24
#SBATCH --mem=192G
#SBATCH --time=24:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
split_manifest=${repo}/training/bc_task_vlm/scale_experiments/tick150_v1/default_47_train_6_heldout.json
cd "${repo}"
train_tasks="$(${python_bin} -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["train_tasks"]))' "${split_manifest}")"
exec training/scripts/launch_cpu_preprocess.sh --no-sbatch \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered \
  --output-dir /work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_reasoning_master_train47_noindex_promptv9 \
  --run-name tick150-reasoning-master-train47-noindex-promptv9 \
  --train-tasks "${train_tasks}" --val-tasks '' --no-pretokenize \
  --no-use-example-cache --example-build-workers 24 --python-bin "${python_bin}" -- \
  --sft-format tool_call --partial-history --train-get-image --train-reasoning \
  --partial-step-index-mode none --partial-observation-mode consume-once \
  --validation-split-mode none --validation-trajectory-fraction 0 \
  --validation-min-trajectories-per-task 0
