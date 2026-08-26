#!/usr/bin/env bash
#SBATCH --job-name=fixed-report
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
ROOT=training/bc_task_vlm/eval_runs/fixed_live_sim
ARTIFACT=training/bc_task_vlm/eval_runs/fixed_live_sim_results_artifact
mkdir -p "${ARTIFACT}/summaries"

for run in sft27_3ep sft90_3ep sft27_1ep_lr2e-4 sft90_1ep_lr2e-4; do
  "${PYTHON_BIN}" -m training.bc_task_vlm.summarize_fixed_live_sim \
    --results "${ROOT}/${run}/train_task_types/live_sim_trajectories.jsonl" \
    --results "${ROOT}/${run}/heldout_task_types/live_sim_trajectories.jsonl" \
    --label "${run}" \
    --output-json "${ARTIFACT}/summaries/${run}.json" \
    --output-md "${ARTIFACT}/summaries/${run}.md"
done

"${PYTHON_BIN}" -m training.bc_task_vlm.build_fixed_live_sim_artifact \
  --cohort training/bc_task_vlm/eval_manifests/fixed_live_sim_v1/frozen.json \
  --summary "${ARTIFACT}/summaries/sft27_3ep.json" \
  --summary "${ARTIFACT}/summaries/sft90_3ep.json" \
  --summary "${ARTIFACT}/summaries/sft27_1ep_lr2e-4.json" \
  --summary "${ARTIFACT}/summaries/sft90_1ep_lr2e-4.json" \
  --output "${ARTIFACT}/report.html"
