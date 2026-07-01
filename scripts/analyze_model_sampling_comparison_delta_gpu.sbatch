#!/usr/bin/env bash
#SBATCH --job-name=model-sample-cos
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
plot_python_bin="${PLOT_PYTHON_BIN:-${PWD}/.venv_d/bin/python}"
output_root="${OUTPUT_ROOT:-${PWD}/data_analysis/model_sampling_comparison}"
plot_output_dir="${PLOT_OUTPUT_DIR:-${PWD}/data_analysis/plots/model_sampling_comparison}"
gemini_flash_metrics="${GEMINI_FLASH_METRICS:-${PWD}/data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3/task_metrics.csv}"
embedding_provider="${EMBEDDING_PROVIDER:-transformers}"
embedding_model="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-4B}"
embedding_batch_size="${EMBEDDING_BATCH_SIZE:-16}"
embedding_cache="${EMBEDDING_CACHE:-auto}"
embedding_progress_interval="${EMBEDDING_PROGRESS_INTERVAL:-1}"
max_trajectories_per_task="${MAX_TRAJECTORIES_PER_TASK:-30}"
load_workers="${LOAD_WORKERS:-16}"
dtype="${DTYPE:-float16}"
hf_cache_root="${HF_CACHE_ROOT:-${PWD}/hf_cache}"

read -r -a methods <<< "${METHODS:-base structured_random}"

model_specs=(
  "gemini_3_1_pro|Gemini-3.1-Pro"
  "gpt_5_4_mini_azure|GPT-5.4-Mini"
)

if [[ ! -x "${python_bin}" ]]; then
  echo "Python executable is not runnable: ${python_bin}" >&2
  exit 1
fi

if [[ ! -x "${plot_python_bin}" ]]; then
  plot_python_bin="${python_bin}"
fi

if [[ ! -x "${plot_python_bin}" ]]; then
  echo "Plot Python executable is not runnable: ${plot_python_bin}" >&2
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
mkdir -p "${HF_HOME}" "${HF_HUB_CACHE}" "${TRANSFORMERS_CACHE}" \
  "${HF_XET_CACHE}" "${XDG_CACHE_HOME}"

case "${embedding_provider}" in
  transformers)
    "${python_bin}" - <<'PY'
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
    ;;
  none | vllm | vllm-openai)
    "${python_bin}" - <<'PY'
import numpy
print(f"numpy {numpy.__version__}")
PY
    ;;
  *)
    echo "Unsupported EMBEDDING_PROVIDER: ${embedding_provider}" >&2
    exit 1
    ;;
esac

"${plot_python_bin}" - <<'PY'
import PIL
print(f"Pillow {PIL.__version__}")
PY

for spec in "${model_specs[@]}"; do
  IFS="|" read -r model_slug model_label <<< "${spec}"
  input_root="${output_root}/raw/${model_slug}/sampling_methods"
  analysis_dir="${output_root}/analysis/${model_slug}"
  if [[ ! -d "${input_root}" ]]; then
    echo "Missing generated sampling root for ${model_label}: ${input_root}" >&2
    exit 1
  fi

  args=(
    data_analysis/analyze_sampling_methods.py
    --input-root "${input_root}"
    --output-dir "${analysis_dir}"
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

  printf 'Analyzing %s from %s\n' "${model_label}" "${input_root}"
  printf 'Running:'
  printf ' %q' "${python_bin}" "${args[@]}"
  printf '\n'
  "${python_bin}" "${args[@]}"
done

plot_args=(
  data_analysis/plot_base_structured_model_comparison.py
  --output-dir "${plot_output_dir}"
  --methods "${methods[@]}"
  --model-input "Gemini-3-Flash=${gemini_flash_metrics}"
)

if [[ ! -f "${gemini_flash_metrics}" ]]; then
  echo "Missing Gemini-3-Flash metrics: ${gemini_flash_metrics}" >&2
  exit 1
fi

for spec in "${model_specs[@]}"; do
  IFS="|" read -r model_slug model_label <<< "${spec}"
  plot_args+=(
    --model-input
    "${model_label}=${output_root}/analysis/${model_slug}/task_metrics.csv"
  )
done

printf 'Plotting comparison:\n'
printf ' %q' "${plot_python_bin}" "${plot_args[@]}"
printf '\n'
"${plot_python_bin}" "${plot_args[@]}"
