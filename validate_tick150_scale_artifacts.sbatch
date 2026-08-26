#!/usr/bin/env bash
#SBATCH --job-name=validate-tick150-scales
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
cd "${repo}"
"${python_bin}" -m training.bc_task_vlm.verify_generation_training_eval_prompts \
  --manifest training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json \
  --output training/bc_task_vlm/scale_experiments/tick150_v1/prompt_contract_gate.json
exec "${python_bin}" -m training.bc_task_vlm.validate_scale_artifacts \
  --artifact-pattern '/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick150_scale_{scale}_train47_noindex_promptv9' \
  --manifest-dir training/bc_task_vlm/scale_experiments/tick150_v1
