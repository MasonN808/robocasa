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
default_staged_dataset_root="/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf/robocasa_20260430T030150Z_full_run_selected"
if [[ -n "${HF_DATASET_REPOS_FILE:-}" || -n "${HF_STAGE_MAX_TOTAL_EPISODES:-}" ]]; then
  default_staged_dataset_root="/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf/robocasa_20260430T030150Z_full_100k_first49"
fi
staged_dataset_root="${STAGED_DATASET_ROOT:-${default_staged_dataset_root}}"
hf_cache_root="${HF_STAGE_CACHE_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/hf_cache}"
load_workers="${HF_STAGE_LOAD_WORKERS:-5}"
stage_workers="${HF_STAGE_WORKERS:-32}"
progress_interval="${HF_STAGE_PROGRESS_INTERVAL:-25}"
resume_existing="${HF_STAGE_RESUME:-false}"
resume_validation="${HF_STAGE_RESUME_VALIDATION:-validated}"

default_hf_dataset_repos=(
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_veggie_dip_prep"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_tong_buffet_setup"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_spicy_marinade"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_sweeten_coffee"
  "DorianAtSchool/robocasa_20260430T030150Z_full_run_setup_wine_glasses"
)

if [[ -n "${HF_DATASET_REPOS_FILE:-}" ]]; then
  if [[ ! -f "${HF_DATASET_REPOS_FILE}" ]]; then
    echo "HF_DATASET_REPOS_FILE does not exist: ${HF_DATASET_REPOS_FILE}" >&2
    exit 1
  fi
  mapfile -t hf_dataset_repos < <(
    sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "${HF_DATASET_REPOS_FILE}"
  )
elif [[ -n "${HF_DATASET_REPOS:-}" ]]; then
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

if [[ ! "${load_workers}" =~ ^[0-9]+$ || "${load_workers}" -lt 1 ]]; then
  echo "HF_STAGE_LOAD_WORKERS must be a positive integer; got ${load_workers}" >&2
  exit 1
fi

if [[ ! "${progress_interval}" =~ ^[0-9]+$ ]]; then
  echo "HF_STAGE_PROGRESS_INTERVAL must be a non-negative integer; got ${progress_interval}" >&2
  exit 1
fi

if [[ "${resume_existing}" != "true" && "${resume_existing}" != "false" ]]; then
  echo "HF_STAGE_RESUME must be true or false; got ${resume_existing}" >&2
  exit 1
fi

if [[ "${resume_validation}" != "validated" && "${resume_validation}" != "unchecked" ]]; then
  echo "HF_STAGE_RESUME_VALIDATION must be validated or unchecked; got ${resume_validation}" >&2
  exit 1
fi

if [[ -n "${HF_STAGE_MAX_TOTAL_EPISODES:-}" && ( ! "${HF_STAGE_MAX_TOTAL_EPISODES}" =~ ^[0-9]+$ || "${HF_STAGE_MAX_TOTAL_EPISODES}" -lt 1 ) ]]; then
  echo "HF_STAGE_MAX_TOTAL_EPISODES must be a positive integer; got ${HF_STAGE_MAX_TOTAL_EPISODES}" >&2
  exit 1
fi

export PATH="$(dirname "${python_bin}"):${PATH}"

if [[ -z "${HF_HUB_CACHE:-}" && -n "${HUGGINGFACE_HUB_CACHE:-}" ]]; then
  hf_hub_cache="${HUGGINGFACE_HUB_CACHE}"
else
  hf_hub_cache="${HF_HUB_CACHE:-${hf_cache_root}/hub}"
fi
hf_datasets_cache="${HF_DATASETS_CACHE:-${hf_cache_root}/datasets}"
hf_assets_cache="${HF_ASSETS_CACHE:-${hf_cache_root}/assets}"
hf_modules_cache="${HF_MODULES_CACHE:-${hf_cache_root}/modules}"

export HF_HUB_CACHE="${hf_hub_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${hf_hub_cache}}"
export HF_DATASETS_CACHE="${hf_datasets_cache}"
export HF_ASSETS_CACHE="${hf_assets_cache}"
export HF_MODULES_CACHE="${hf_modules_cache}"

mkdir -p \
  "${HF_HUB_CACHE}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${HF_DATASETS_CACHE}" \
  "${HF_ASSETS_CACHE}" \
  "${HF_MODULES_CACHE}"

"${python_bin}" -c "import datasets, huggingface_hub, PIL" >/dev/null 2>&1 || {
  echo "Missing staging dependencies in ${python_bin}." >&2
  echo "Install training/bc_task_vlm/requirements.txt into that environment." >&2
  exit 1
}

stage_args=(
  --output-root "${staged_dataset_root}"
  --split "${HF_DATASET_SPLIT:-train}"
  --load-workers "${load_workers}"
  --workers "${stage_workers}"
  --progress-interval "${progress_interval}"
)

if [[ -n "${HF_STAGE_MAX_IN_FLIGHT:-}" ]]; then
  stage_args+=(--max-in-flight "${HF_STAGE_MAX_IN_FLIGHT}")
fi

if [[ -n "${HF_STAGE_MAX_TOTAL_EPISODES:-}" ]]; then
  stage_args+=(--max-total-episodes "${HF_STAGE_MAX_TOTAL_EPISODES}")
fi

if [[ "${resume_existing}" == "true" ]]; then
  stage_args+=(--resume --resume-validation "${resume_validation}")
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
echo "HF datasets cache: ${HF_DATASETS_CACHE}"
echo "HF hub cache: ${HF_HUB_CACHE}"
echo "Load workers: ${load_workers}"
echo "Workers: ${stage_workers}"
echo "Resume existing: ${resume_existing}"
if [[ -n "${HF_DATASET_REPOS_FILE:-}" ]]; then
  echo "Repo list file: ${HF_DATASET_REPOS_FILE}"
fi
if [[ "${resume_existing}" == "true" ]]; then
  echo "Resume validation: ${resume_validation}"
fi
if [[ -n "${HF_STAGE_MAX_IN_FLIGHT:-}" ]]; then
  echo "Max in-flight episodes: ${HF_STAGE_MAX_IN_FLIGHT}"
fi
if [[ -n "${HF_STAGE_MAX_TOTAL_EPISODES:-}" ]]; then
  echo "Max total episodes: ${HF_STAGE_MAX_TOTAL_EPISODES}"
fi
printf 'Repos:'
printf ' %s' "${hf_dataset_repos[@]}"
printf '\n'
printf 'Running:'
printf ' %q' "${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
printf '\n'

"${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
