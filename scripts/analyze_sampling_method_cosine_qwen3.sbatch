#!/usr/bin/env bash
#SBATCH --job-name=sample-cos-qwen3
#SBATCH --account=bgjs-delta-gpu
#SBATCH --partition=gpuA100x4
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

default_python_bin="/sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/bin/python"
if [[ ! -x "${default_python_bin}" ]]; then
  default_python_bin="/work/hdd/bgjs/mnakamura/robocasa/.venv_d/bin/python"
fi
python_bin="${PYTHON_BIN:-${default_python_bin}}"
input_root="${SAMPLING_INPUT_ROOT:-data_generation/task_level/data_k3/raw/sampling_methods}"
output_dir="${SAMPLING_ANALYSIS_OUTPUT_DIR:-data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3}"
embedding_provider="${EMBEDDING_PROVIDER:-transformers}"
embedding_model="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
embedding_batch_size="${EMBEDDING_BATCH_SIZE:-16}"
max_trajectories_per_task="${MAX_TRAJECTORIES_PER_TASK:-30}"
load_workers="${LOAD_WORKERS:-16}"
embedding_cache="${EMBEDDING_CACHE:-auto}"
embedding_progress_interval="${EMBEDDING_PROGRESS_INTERVAL:-1}"
dtype="${DTYPE:-float16}"
skip_import_preflight="${SKIP_IMPORT_PREFLIGHT:-false}"
disable_flash_attn_discovery="${DISABLE_FLASH_ATTN_DISCOVERY:-false}"
hf_cache_root="${HF_CACHE_ROOT:-/work/hdd/bgjs/mnakamura/robocasa/hf_cache}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is not runnable: ${python_bin}" >&2
  echo "Set PYTHON_BIN to an environment with torch, transformers, and numpy installed." >&2
  exit 1
fi

export PATH="$(dirname "${python_bin}"):${PATH}"
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-${hf_cache_root}}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HF_HOME}/xdg}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" "${HF_XET_CACHE}" "${XDG_CACHE_HOME}"

printf 'Using Hugging Face cache root: %s\n' "${HF_HOME}"

if [[ "${disable_flash_attn_discovery}" == "true" ]]; then
  export ROBOCASA_DISABLE_FLASH_ATTN_DISCOVERY=1
  export PYTHONPATH="${PWD}/data_analysis/vllm_no_flash_attn${PYTHONPATH:+:${PYTHONPATH}}"
  echo "Disabling optional flash_attn discovery for vLLM."
fi

if [[ "${skip_import_preflight}" == "true" ]]; then
  echo "Skipping Python import preflight because SKIP_IMPORT_PREFLIGHT=true."
else
  case "${embedding_provider}" in
    vllm)
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
      ;;
    transformers)
      "${python_bin}" - <<'PY' || {
import importlib.util

import numpy

missing = [
    module_name
    for module_name in ("torch", "transformers")
    if importlib.util.find_spec(module_name) is None
]
if missing:
    raise ModuleNotFoundError(", ".join(missing))

print(f"numpy {numpy.__version__}")
print("torch available")
print("transformers available")
PY
        echo "Failed to find Transformers/Torch/numpy from ${python_bin}." >&2
        echo "Set PYTHON_BIN to an environment with torch, transformers, and numpy installed." >&2
        exit 1
      }
      ;;
    vllm-openai | none)
      "${python_bin}" - <<'PY' || {
import numpy

print(f"numpy {numpy.__version__}")
PY
        echo "Failed to import numpy from ${python_bin}." >&2
        echo "Set PYTHON_BIN to an environment with numpy installed." >&2
        exit 1
      }
      ;;
    *)
      echo "Unsupported embedding provider: ${embedding_provider}" >&2
      exit 1
      ;;
  esac
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
