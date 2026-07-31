#!/usr/bin/env bash
#SBATCH --job-name=judge-v3-matrix
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# LLM-judge over every off-sim run of the v3 matrix (8 cells x 2 splits).
# Required for any communication claim: raw comm_exact_call_accuracy measures
# match to the training source's phrasing conventions, not quality -- the
# cross-source eval showed up to a 17x own-vs-cross gap on that metric alone.
#
# Runs on cpu: the judge is stdlib + google-genai only (via
# training/bc_task_vlm/prediction_io.py, which exists so this does not drag in
# torch/transformers). Resumable -- cached verdicts are reused when
# predicted_args still match.
#   sbatch --dependency=afterok:<offsim ids> judge_v3_matrix.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

VENV=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/vertex_venv
export GOOGLE_APPLICATION_CREDENTIALS=/work/umass/shlomo_umass/dbenhamougol_umass/vertex_env/config/gcloud/application_default_credentials.json
export GOOGLE_CLOUD_PROJECT=access-cis260215-538827
# Vertex Gemini needs the global endpoint here; regional endpoints 404
# (EXPERIMENT.md section 8).
export GOOGLE_CLOUD_LOCATION=global
# google-auth warns that this venv's python 3.9 is EOL; harmless.
export PYTHONWARNINGS=ignore
export PYTHONUNBUFFERED=1

run_dir_args=()
for src in vs sr; do
  for obs in cent partial; do
    for r in noreason reason; do
      for split in heldout_trajectories heldout_tasks; do
        d="training/bc_task_vlm/eval_runs/offsim_qwen3vl-8b-v3-${src}-${obs}-${r}__${split}"
        if [[ -f "${d}/structured_eval_predictions.jsonl" ]]; then
          run_dir_args+=(--run-dir "${d}")
        else
          echo "skip (no predictions): ${d}"
        fi
      done
    done
  done
done

if [[ ${#run_dir_args[@]} -eq 0 ]]; then
  echo "No prediction files found; nothing to judge." >&2
  exit 1
fi

echo "Judging $(( ${#run_dir_args[@]} / 2 )) run dirs"

"${VENV}/bin/python" -m training.bc_task_vlm.judge_communications \
  "${run_dir_args[@]}" \
  --judge-model gemini-3-flash-preview \
  --project "${GOOGLE_CLOUD_PROJECT}" \
  --location "${GOOGLE_CLOUD_LOCATION}" \
  --concurrency 8

echo "Done."
