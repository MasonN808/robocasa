#!/usr/bin/env bash
#SBATCH --job-name=robocasa-hf-stage
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128g
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

python_bin="${PYTHON_BIN:-/work/hdd/bgjs/mnakamura/robocasa/.venv_d/bin/python}"
staged_dataset_root="${STAGED_DATASET_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf/robocasa_20260430T030150Z_full_run_selected}"
stage_workers="${HF_STAGE_WORKERS:-32}"
progress_interval="${HF_STAGE_PROGRESS_INTERVAL:-25}"

default_hf_dataset_repos=(
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_veggie_dip_prep"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_tong_buffet_setup"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_spicy_marinade"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_sweeten_coffee"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_setup_wine_glasses"
)

if [[ -n "${HF_DATASET_REPOS:-}" ]]; then
  IFS=',' read -r -a hf_dataset_repos <<< "${HF_DATASET_REPOS}"
else
  hf_dataset_repos=("${default_hf_dataset_repos[@]}")
fi

normalized_hf_dataset_repos=()
for repo_id in "${hf_dataset_repos[@]}"; do
  repo_id="${repo_id#"${repo_id%%[![:space:]]*}"}"
  repo_id="${repo_id%"${repo_id##*[![:space:]]}"}"
  if [[ -n "${repo_id}" ]]; then
    normalized_hf_dataset_repos+=("${repo_id}")
  fi
done
hf_dataset_repos=("${normalized_hf_dataset_repos[@]}")

if [[ "${#hf_dataset_repos[@]}" -lt 1 ]]; then
  echo "No Hugging Face dataset repos configured." >&2
  exit 1
fi

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is not runnable: ${python_bin}" >&2
  echo "Set PYTHON_BIN to a working environment with datasets, Pillow, and huggingface_hub installed." >&2
  exit 1
fi

if [[ ! "${stage_workers}" =~ ^[0-9]+$ || "${stage_workers}" -lt 1 ]]; then
  echo "HF_STAGE_WORKERS must be a positive integer; got ${stage_workers}" >&2
  exit 1
fi

if [[ ! "${progress_interval}" =~ ^[0-9]+$ ]]; then
  echo "HF_STAGE_PROGRESS_INTERVAL must be a non-negative integer; got ${progress_interval}" >&2
  exit 1
fi

export PATH="$(dirname "${python_bin}"):${PATH}"

"${python_bin}" -c "import datasets, huggingface_hub, PIL" >/dev/null 2>&1 || {
  echo "Missing staging dependencies in ${python_bin}." >&2
  echo "Install training/bc_task_vlm/requirements.txt into that environment." >&2
  exit 1
}

stage_args=(
  --output-root "${staged_dataset_root}"
  --split "${HF_DATASET_SPLIT:-train}"
  --workers "${stage_workers}"
  --progress-interval "${progress_interval}"
)

if [[ -n "${HF_STAGE_MAX_IN_FLIGHT:-}" ]]; then
  stage_args+=(--max-in-flight "${HF_STAGE_MAX_IN_FLIGHT}")
fi

if [[ -n "${HF_DATASET_REVISION:-}" ]]; then
  stage_args+=(--revision "${HF_DATASET_REVISION}")
fi

if [[ "${HF_TRUST_REMOTE_CODE:-false}" == "true" ]]; then
  stage_args+=(--trust-remote-code)
fi

for repo_id in "${hf_dataset_repos[@]}"; do
  if [[ -n "${repo_id}" ]]; then
    stage_args+=(--repo-id "${repo_id}")
  fi
done

echo "Staging HF datasets on CPU"
echo "Output root: ${staged_dataset_root}"
echo "Workers: ${stage_workers}"
if [[ -n "${HF_STAGE_MAX_IN_FLIGHT:-}" ]]; then
  echo "Max in-flight episodes: ${HF_STAGE_MAX_IN_FLIGHT}"
fi
printf 'Repos:'
printf ' %s' "${hf_dataset_repos[@]}"
printf '\n'
printf 'Running:'
printf ' %q' "${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
printf '\n'

"${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
