#!/usr/bin/env bash
#SBATCH --job-name=sample-cos-qwen3
#SBATCH --account=bgjs-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --mem=128g
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

python_bin="${PYTHON_BIN:-/work/hdd/bgjs/mnakamura/robocasa/.venv/bin/python}"
input_root="${SAMPLING_INPUT_ROOT:-data_generation/task_level/data_k3/raw/sampling_methods}"
output_dir="${SAMPLING_ANALYSIS_OUTPUT_DIR:-data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3}"
embedding_provider="${EMBEDDING_PROVIDER:-vllm}"
embedding_model="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
embedding_batch_size="${EMBEDDING_BATCH_SIZE:-256}"
max_trajectories_per_task="${MAX_TRAJECTORIES_PER_TASK:-30}"
load_workers="${LOAD_WORKERS:-16}"
embedding_cache="${EMBEDDING_CACHE:-auto}"
embedding_progress_interval="${EMBEDDING_PROGRESS_INTERVAL:-1}"
dtype="${DTYPE:-float16}"
skip_import_preflight="${SKIP_IMPORT_PREFLIGHT:-false}"
disable_flash_attn_discovery="${DISABLE_FLASH_ATTN_DISCOVERY:-false}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is not runnable: ${python_bin}" >&2
  echo "Set PYTHON_BIN to an environment with vLLM and numpy installed." >&2
  exit 1
fi

export PATH="$(dirname "${python_bin}"):${PATH}"
export TOKENIZERS_PARALLELISM=false

if [[ "${disable_flash_attn_discovery}" == "true" ]]; then
  export ROBOCASA_DISABLE_FLASH_ATTN_DISCOVERY=1
  export PYTHONPATH="${PWD}/data_analysis/vllm_no_flash_attn${PYTHONPATH:+:${PYTHONPATH}}"
  echo "Disabling optional flash_attn discovery for vLLM."
fi

if [[ "${skip_import_preflight}" == "true" ]]; then
  echo "Skipping Python import preflight because SKIP_IMPORT_PREFLIGHT=true."
else
  "${python_bin}" - <<'PY' || {
import importlib.metadata as metadata

import numpy
import vllm

print(f"numpy {numpy.__version__}")
print(f"vllm {metadata.version('vllm')}")
PY
    echo "Failed to import vLLM/numpy from ${python_bin}." >&2
    echo "Set PYTHON_BIN to a clean environment where vLLM imports successfully." >&2
    exit 1
  }
fi

args=(
  data_analysis/analyze_sampling_methods.py
  --input-root "${input_root}"
  --output-dir "${output_dir}"
  --embedding-provider "${embedding_provider}"
  --embedding-model "${embedding_model}"
  --embedding-batch-size "${embedding_batch_size}"
  --embedding-cache "${embedding_cache}"
  --embedding-progress-interval "${embedding_progress_interval}"
  --max-trajectories-per-task "${max_trajectories_per_task}"
  --load-workers "${load_workers}"
  --dtype "${dtype}"
)

if [[ -n "${TENSOR_PARALLEL_SIZE:-}" ]]; then
  args+=(--tensor-parallel-size "${TENSOR_PARALLEL_SIZE}")
fi

if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
  args+=(--max-model-len "${MAX_MODEL_LEN}")
fi

if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
  args+=(--gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}")
fi

if [[ -n "${EMBEDDING_INSTRUCTION:-}" ]]; then
  args+=(--embedding-instruction "${EMBEDDING_INSTRUCTION}")
fi

printf 'Running:'
printf ' %q' "${python_bin}" "${args[@]}"
printf '\n'

"${python_bin}" "${args[@]}"
