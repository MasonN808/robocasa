#!/usr/bin/env bash
#SBATCH --job-name=judge-all-comms
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# LLM-judge pass over every standalone eval run (4 same-source + 4
# cross-source models x 2 splits = 16 run dirs). Raw
# comm_exact_call_accuracy is a style-match artifact -- the cross-source eval
# showed up to a 17x home-field advantage purely from phrasing conventions --
# so any communication claim needs these judged numbers instead.
#
# Runs on the cpu partition rather than the login node: the judge is
# dependency-light (stdlib + google-genai only, via
# training/bc_task_vlm/prediction_io.py, which exists so this doesn't drag in
# torch/transformers), but the login-node session has a 5GB memory cgroup that
# OOM-kills long-running processes. cpu nodes have internet egress (proven by
# the HF staging jobs).
#
# Writes comm_judge.jsonl (per-step verdicts) + comm_judge_metrics.json into
# each run dir. Resumable: cached verdicts are reused when predicted_args
# still match, so re-running after an interruption is cheap.
#   sbatch judge_all_comms.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

VENV=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/vertex_venv
export GOOGLE_APPLICATION_CREDENTIALS=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/config/gcloud/application_default_credentials.json
export GOOGLE_CLOUD_PROJECT=access-cis260215-538827
# Vertex Gemini needs the global endpoint here; regional endpoints 404
# (recorded in training/bc_task_vlm/EXPERIMENT.md section 8).
export GOOGLE_CLOUD_LOCATION=global
# google-auth warns loudly that this venv's python 3.9 is EOL; harmless here.
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

runs_root=training/bc_task_vlm/eval_runs
run_dir_args=()
for d in \
  qwen3vl_8b_v1_verbalized_sampling_baseline \
  qwen3vl_8b_v1_5_verbalized_sampling \
  qwen3vl_8b_v1_structured_random \
  qwen3vl_8b_v1_5_structured_random \
  xqwen3vl_8b_v1_verbalized_ON_structured_random \
  xqwen3vl_8b_v1_5_verbalized_ON_structured_random \
  xqwen3vl_8b_v1_structured_random_ON_exp52 \
  xqwen3vl_8b_v1_5_structured_random_ON_exp52 \
; do
  for split in heldout_trajectories heldout_tasks; do
    run_dir_args+=(--run-dir "${runs_root}/${d}__${split}")
  done
done

echo "Judging ${#run_dir_args[@]} args ($(( ${#run_dir_args[@]} / 2 )) run dirs)"

"${VENV}/bin/python" -m training.bc_task_vlm.judge_communications \
  "${run_dir_args[@]}" \
  --judge-model gemini-3-flash-preview \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --location "${GOOGLE_CLOUD_LOCATION}" \
  --concurrency 8

echo "Done."
