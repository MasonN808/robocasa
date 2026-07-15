#!/usr/bin/env bash
#SBATCH --job-name=task-vlm-eval-hf
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --mem=256g
#SBATCH --time=08:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# GPU wrapper for the standalone HF eval. All arguments are forwarded to
# training.bc_task_vlm.eval_standalone, e.g.:
#   sbatch training/scripts/eval_standalone_hf.sh \
#     --backend hf --model-name-or-path Qwen/Qwen3.6-27B \
#     --manifest training/bc_task_vlm/eval_manifests/exp1/eval_manifest_heldout_tasks.json \
#     --output-dir training/bc_task_vlm/eval_runs/qwen36_base__heldout_task --resume

set -euo pipefail

mkdir -p slurm_logs

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
export UV_CACHE_DIR=/work/hdd/bgjs/dbenhamougoldfajn/.uv_cache
export HF_HOME="${HF_HOME:-/work/hdd/bgjs/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
mkdir -p "${HF_HOME}"

if [[ -f "${repo_root}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${repo_root}/.env"
  set +a
fi

exec "${python_bin}" -m training.bc_task_vlm.eval_standalone "$@"
