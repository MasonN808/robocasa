#!/usr/bin/env bash
#SBATCH --job-name=artifact-noidx-scale
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
cd "${repo}"
/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python training/bc_task_vlm/build_noindex_training_scale_artifact.py
cd /home/dbenhamougol_umass/.codex/plugins/cache/openai-curated-remote/data-analytics/0.2.8-13ceeea1f599
PATH=/tmp/node-v22.17.0-linux-x64/bin:${PATH} npm run report:deliver -- \
  --input "${repo}/training/bc_task_vlm/eval_runs/noindex_training_scale_comparison_artifact/artifact.json" \
  --output "${repo}/training/bc_task_vlm/eval_runs/noindex_training_scale_comparison_artifact/report.html"
