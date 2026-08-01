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
add_dir() {
  if [[ -f "$1/structured_eval_predictions.jsonl" ]]; then
    run_dir_args+=(--run-dir "$1")
  else
    echo "skip (no predictions): $1"
  fi
}

# Centralized cells live in the "-notc" runs -- the task_complete-era ones are
# superseded and deliberately not sourced by build_v3_artifact.py either, so
# judging them would produce numbers the artifact never reads.
for src in vs sr; do
  for obs in cent partial; do
    suffix=""
    [[ "${obs}" == "cent" ]] && suffix="-notc"
    for r in noreason reason; do
      for split in heldout_trajectories heldout_tasks; do
        add_dir "training/bc_task_vlm/eval_runs/offsim_qwen3vl-8b-v3-${src}-${obs}-${r}${suffix}__${split}"
      done
    done
  done
done

# Untrained baselines, judged on the same footing. JUDGE_BASELINES=0 skips them.
if [[ "${JUDGE_BASELINES:-1}" == "1" ]]; then
  for model in gemini-3-flash-preview gemini-robotics-er-2-preview; do
    for src in vs sr; do
      for obs in cent partial; do
        for split in heldout_trajectories heldout_tasks; do
          add_dir "training/bc_task_vlm/eval_runs/offsim_ots-${model}-${obs}-${src}__${split}"
        done
      done
    done
  done
fi

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
