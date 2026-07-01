#!/usr/bin/env bash
#SBATCH --job-name=model-sample-gen
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/hdd/bgjs/mnakamura/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

set -euo pipefail

mkdir -p slurm_logs

repo_root="${ROBOCASA_REPO_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
repo_root="$(cd "${repo_root}" && pwd)"
if [[ ! -f "${repo_root}/data_generation/task_level/generation/raw/cli.py" ]]; then
  echo "Could not resolve RoboCasa repository root from: ${repo_root}" >&2
  echo "Set ROBOCASA_REPO_ROOT to /work/hdd/bgjs/mnakamura/robocasa." >&2
  exit 1
fi
default_python_bin="${repo_root}/.venv_d/bin/python"
if [[ ! -x "${default_python_bin}" ]]; then
  default_python_bin="python"
fi

python_bin="${PYTHON_BIN:-${default_python_bin}}"
output_root="${OUTPUT_ROOT:-${repo_root}/data_analysis/model_sampling_comparison}"
num_runs="${NUM_RUNS:-30}"
max_workers="${MAX_WORKERS:-4}"
max_retries="${MAX_RETRIES:-5}"
temperature="${TEMPERATURE:-0.6}"
thinking_level="${THINKING_LEVEL:-low}"
random_start_location="${RANDOM_START_LOCATION:-true}"
generation_timeout_sec="${GENERATION_TIMEOUT_SEC:-300}"
parallelize_tasks="${PARALLELIZE_TASKS:-false}"
validation_mode="${VALIDATION_MODE:-disable}"
parallelize_models="${PARALLELIZE_MODELS:-true}"

read -r -a task_args <<< "${TASKS:-verified}"
read -r -a methods <<< "${METHODS:-base structured_random}"

model_specs=(
  "gemini_3_1_pro|Gemini-3.1-Pro|google-genai|${GEMINI_MODEL:-gemini-3.1-pro-preview}|${GEMINI_LOCATION:-${GOOGLE_CLOUD_LOCATION:-global}}|${GEMINI_TEMPERATURE:-${TEMPERATURE:-0.6}}"
  "gpt_5_4_mini_azure|GPT-5.4-Mini|azure-openai|${AZURE_MODEL:-gpt-5.4-mini}|${AZURE_LOCATION:-global}|${AZURE_TEMPERATURE:-1}"
)

if [[ "${validation_mode}" == "enable" ]]; then
  validation_arg="--enable-validation"
elif [[ "${validation_mode}" == "disable" ]]; then
  validation_arg="--disable-validation"
else
  echo "VALIDATION_MODE must be either enable or disable." >&2
  exit 1
fi

if [[ "${parallelize_models}" != "true" && "${parallelize_models}" != "false" ]]; then
  echo "PARALLELIZE_MODELS must be either true or false." >&2
  exit 1
fi

for positive_integer in num_runs max_workers max_retries generation_timeout_sec; do
  value="${!positive_integer}"
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${positive_integer} must be a positive integer." >&2
    exit 1
  fi
done

for decimal_value in temperature; do
  value="${!decimal_value}"
  if ! [[ "${value}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "${decimal_value} must be a non-negative number." >&2
    exit 1
  fi
done

for spec in "${model_specs[@]}"; do
  IFS="|" read -r model_slug _ _ _ _ model_temperature <<< "${spec}"
  if ! [[ "${model_temperature}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "${model_slug} temperature must be a non-negative number." >&2
    exit 1
  fi
done

ensure_verified_task_resume_dirs() {
  local method_dir="$1"
  "${python_bin}" - "$method_dir" <<'PY'
from pathlib import Path
import sys

from data_generation.task_level.tasks.specs import supported_verified_task_names
from data_generation.utils import camel_to_snake_case

method_dir = Path(sys.argv[1])
method_dir.mkdir(parents=True, exist_ok=True)
for task_name in supported_verified_task_names():
    (method_dir / camel_to_snake_case(task_name)).mkdir(parents=True, exist_ok=True)
PY
}

write_run_config() {
  "${python_bin}" - "$output_root" "$num_runs" "$max_workers" "$max_retries" \
    "$temperature" "$thinking_level" "$random_start_location" \
    "$generation_timeout_sec" "$validation_mode" "${task_args[*]}" \
    "${methods[*]}" "${model_specs[@]}" <<'PY'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

(
    output_root,
    num_runs,
    max_workers,
    max_retries,
    temperature,
    thinking_level,
    random_start_location,
    generation_timeout_sec,
    validation_mode,
    tasks,
    methods,
    *model_specs,
) = sys.argv[1:]

models = []
for spec in model_specs:
    slug, label, sdk, model, location, model_temperature = spec.split("|", 5)
    models.append(
        {
            "slug": slug,
            "label": label,
            "sdk": sdk,
            "model": model,
            "location": location,
            "temperature": float(model_temperature),
            "sampling_root": str(
                Path(output_root) / "raw" / slug / "sampling_methods"
            ),
            "analysis_dir": str(Path(output_root) / "analysis" / slug),
        }
    )

payload = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "tasks": tasks.split(),
    "methods": methods.split(),
    "num_runs_per_task": int(num_runs),
    "max_workers": int(max_workers),
    "max_retries": int(max_retries),
    "temperature": float(temperature),
    "thinking_level": thinking_level,
    "random_start_location": random_start_location,
    "generation_timeout_sec": int(generation_timeout_sec),
    "validation_mode": validation_mode,
    "models": models,
}
path = Path(output_root) / "run_config.json"
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
print(path)
PY
}

failed_runs=()

write_run_config

run_model_spec() {
  local spec="$1"
  local model_slug model_label sdk model location model_temperature sampling_root failed_methods
  IFS="|" read -r model_slug model_label sdk model location model_temperature <<< "${spec}"
  sampling_root="${output_root}/raw/${model_slug}/sampling_methods"
  failed_methods=()

  for method in "${methods[@]}"; do
    local method_dir summary_path exit_code
    local -a cli_args
    method_dir="${sampling_root}/${method}"
    summary_path="${method_dir}/summary.json"
    cli_args=(
      -m data_generation.task_level.generation.raw.cli
      --tasks "${task_args[@]}"
      --num-runs "${num_runs}"
      --random-start-location "${random_start_location}"
      --sampling "${method}"
      --temperature "${model_temperature}"
      --model "${model}"
      --sdk "${sdk}"
      --location "${location}"
      --thinking-level "${thinking_level}"
      --max-workers "${max_workers}"
      --max-retries "${max_retries}"
      --generation-timeout-sec "${generation_timeout_sec}"
      "${validation_arg}"
    )

    if [[ "${parallelize_tasks}" == "true" ]]; then
      cli_args+=(--parallelize-tasks)
    fi

    if [[ -d "${method_dir}" ]]; then
      ensure_verified_task_resume_dirs "${method_dir}"
      cli_args+=(--resume "${method_dir}")
    else
      mkdir -p "${method_dir}"
      cli_args+=(--summary-path "${summary_path}")
    fi

    printf 'Generating %s / %s at temperature %s into %s\n' \
      "${model_label}" "${method}" "${model_temperature}" "${method_dir}"
    if (
      cd "${repo_root}"
      "${python_bin}" "${cli_args[@]}"
    ); then
      printf 'Completed %s / %s\n' "${model_label}" "${method}"
    else
      exit_code="$?"
      failed_methods+=("${method}:${exit_code}")
      printf 'Generation failed for %s / %s with exit code %s; continuing.\n' \
        "${model_label}" "${method}" "${exit_code}" >&2
    fi
  done

  if [[ ${#failed_methods[@]} -gt 0 ]]; then
    echo "One or more methods failed for ${model_label}:" >&2
    for failed_method in "${failed_methods[@]}"; do
      echo "  ${model_slug}:${failed_method}" >&2
    done
    return 1
  fi
}

if [[ "${parallelize_models}" == "true" ]]; then
  declare -a model_pids=()
  declare -a model_pid_labels=()
  printf 'Launching %s model generation branches in parallel.\n' "${#model_specs[@]}"
  for spec in "${model_specs[@]}"; do
    IFS="|" read -r model_slug model_label _ <<< "${spec}"
    printf 'Starting model branch: %s\n' "${model_label}"
    run_model_spec "${spec}" &
    model_pids+=("$!")
    model_pid_labels+=("${model_slug}")
  done

  for index in "${!model_pids[@]}"; do
    if wait "${model_pids[$index]}"; then
      printf 'Model branch completed: %s\n' "${model_pid_labels[$index]}"
    else
      exit_code="$?"
      failed_runs+=("${model_pid_labels[$index]}:${exit_code}")
      printf 'Model branch failed: %s exit %s\n' \
        "${model_pid_labels[$index]}" "${exit_code}" >&2
    fi
  done
else
  for spec in "${model_specs[@]}"; do
    IFS="|" read -r model_slug _ <<< "${spec}"
    if run_model_spec "${spec}"; then
      printf 'Model branch completed: %s\n' "${model_slug}"
    else
      exit_code="$?"
      failed_runs+=("${model_slug}:${exit_code}")
    fi
  done
fi

if [[ ${#failed_runs[@]} -gt 0 ]]; then
  echo "One or more generation runs failed:" >&2
  for failed_run in "${failed_runs[@]}"; do
    echo "  ${failed_run}" >&2
  done
  exit 1
fi
