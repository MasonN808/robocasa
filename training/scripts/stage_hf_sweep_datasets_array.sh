#!/usr/bin/env bash
#SBATCH --job-name=robocasa-hf-stage-array
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err

set -euo pipefail

mkdir -p slurm_logs

python_bin="${PYTHON_BIN:-/work/hdd/bgjs/mnakamura/robocasa/.venv_d/bin/python}"
repo_list_file="${HF_DATASET_REPOS_FILE:?Set HF_DATASET_REPOS_FILE to a newline-delimited repo list.}"
shard_root_base="${HF_STAGE_SHARD_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/staged_hf_shards/robocasa_20260430T030150Z_full_49}"
hf_cache_root="${HF_STAGE_CACHE_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/training/bc_task_vlm/hf_cache}"
load_workers="${HF_STAGE_LOAD_WORKERS:-1}"
stage_workers="${HF_STAGE_WORKERS:-${SLURM_CPUS_PER_TASK:-16}}"
progress_interval="${HF_STAGE_PROGRESS_INTERVAL:-25}"
resume_existing="${HF_STAGE_RESUME:-true}"
resume_validation="${HF_STAGE_RESUME_VALIDATION:-validated}"
array_task_id="${SLURM_ARRAY_TASK_ID:-0}"
array_task_count="${HF_STAGE_NUM_SHARDS:-${SLURM_ARRAY_TASK_COUNT:-1}}"

if [[ ! -f "${repo_list_file}" ]]; then
  echo "HF_DATASET_REPOS_FILE does not exist: ${repo_list_file}" >&2
  exit 1
fi

if [[ ! "${array_task_id}" =~ ^[0-9]+$ ]]; then
  echo "SLURM_ARRAY_TASK_ID must be a non-negative integer; got ${array_task_id}" >&2
  exit 1
fi

if [[ ! "${array_task_count}" =~ ^[0-9]+$ || "${array_task_count}" -lt 1 ]]; then
  echo "HF_STAGE_NUM_SHARDS/SLURM_ARRAY_TASK_COUNT must be a positive integer; got ${array_task_count}" >&2
  exit 1
fi

if [[ "${array_task_id}" -ge "${array_task_count}" ]]; then
  echo "Shard index ${array_task_id} is outside shard count ${array_task_count}" >&2
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

if [[ "${resume_existing}" != "true" && "${resume_existing}" != "false" ]]; then
  echo "HF_STAGE_RESUME must be true or false; got ${resume_existing}" >&2
  exit 1
fi

if [[ "${resume_validation}" != "validated" && "${resume_validation}" != "unchecked" ]]; then
  echo "HF_STAGE_RESUME_VALIDATION must be validated or unchecked; got ${resume_validation}" >&2
  exit 1
fi

mapfile -t all_repos < <(
  sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "${repo_list_file}"
)

shard_repos=()
for index in "${!all_repos[@]}"; do
  if (( index % array_task_count == array_task_id )); then
    shard_repos+=("${all_repos[index]}")
  fi
done

if [[ "${#shard_repos[@]}" -lt 1 ]]; then
  echo "Shard ${array_task_id}/${array_task_count} has no repos to stage."
  exit 0
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

shard_name="$(printf 'shard_%03d' "${array_task_id}")"
shard_output_root="${HF_STAGE_SHARD_OUTPUT_ROOT:-${shard_root_base}/${shard_name}}"
manifest_path="${shard_output_root}/hf_stage_manifest.json"

stage_args=(
  --output-root "${shard_output_root}"
  --manifest-path "${manifest_path}"
  --split "${HF_DATASET_SPLIT:-train}"
  --load-workers "${load_workers}"
  --workers "${stage_workers}"
  --progress-interval "${progress_interval}"
  --trajectory-id-prefix "${shard_name}_"
)

if [[ -n "${HF_STAGE_MAX_IN_FLIGHT:-}" ]]; then
  stage_args+=(--max-in-flight "${HF_STAGE_MAX_IN_FLIGHT}")
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

for repo_id in "${shard_repos[@]}"; do
  stage_args+=(--repo-id "${repo_id}")
done

echo "Staging HF dataset shard"
echo "Shard: ${array_task_id}/${array_task_count}"
echo "Output root: ${shard_output_root}"
echo "Manifest: ${manifest_path}"
echo "HF datasets cache: ${HF_DATASETS_CACHE}"
echo "HF hub cache: ${HF_HUB_CACHE}"
echo "Load workers: ${load_workers}"
echo "Workers: ${stage_workers}"
printf 'Repos:'
printf ' %s' "${shard_repos[@]}"
printf '\n'
printf 'Running:'
printf ' %q' "${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
printf '\n'

"${python_bin}" training/scripts/stage_hf_sweep_datasets.py "${stage_args[@]}"
