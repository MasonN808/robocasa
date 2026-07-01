#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PYTHON_BIN:-}" ]]; then
  python_bin="${PYTHON_BIN}"
elif [[ -x ".venv/bin/python" ]]; then
  python_bin=".venv/bin/python"
else
  python_bin="python"
fi

check_training_environment() {
  local import_log nccl_lib nccl_requirement site_packages
  local -a env_lines

  if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "Python executable is not runnable: ${python_bin}" >&2
    echo "Set PYTHON_BIN to the training environment that has torch and accelerate installed." >&2
    exit 1
  fi

  import_log="$(mktemp)"
  if ! mapfile -t env_lines < <("${python_bin}" -c 'import importlib.metadata as m
from pathlib import Path
import sys

try:
    torch_dist = m.distribution("torch")
    m.version("accelerate")
except m.PackageNotFoundError as exc:
    print(f"Missing required training package: {exc.name}", file=sys.stderr)
    raise SystemExit(1)

site_packages = Path(torch_dist.locate_file("")).resolve()
nccl_requirement = ""
for req in m.requires("torch") or []:
    req = req.split(";", 1)[0].strip()
    if req.startswith("nvidia-nccl-"):
        nccl_requirement = req
        break

print(site_packages)
print(nccl_requirement)
' 2>"${import_log}"); then
    echo "Training Python metadata check failed: ${python_bin}" >&2
    cat "${import_log}" >&2
    rm -f "${import_log}"
    exit 1
  fi
  rm -f "${import_log}"

  site_packages="${env_lines[0]:-}"
  nccl_requirement="${env_lines[1]:-}"
  if [[ "${nccl_requirement}" == nvidia-nccl-cu13* ]]; then
    nccl_lib="${site_packages}/nvidia/nccl/lib/libnccl.so.2"
    if [[ ! -f "${nccl_lib}" ]]; then
      echo "Missing NCCL runtime library required by torch: ${nccl_lib}" >&2
      echo "Reinstall the NCCL wheel required by torch:" >&2
      echo "  uv --cache-dir /tmp/robocasa_uv_cache pip install --python ${python_bin} --force-reinstall --no-deps ${nccl_requirement}" >&2
      exit 1
    fi

    if command -v nm >/dev/null 2>&1 && ! grep -q " ncclDevCommDestroy$" < <(nm -D --defined-only "${nccl_lib}" 2>/dev/null); then
      echo "Detected a PyTorch/NCCL library mismatch." >&2
      echo "Torch requires ${nccl_requirement}, but ${nccl_lib} does not export ncclDevCommDestroy." >&2
      echo "If both CUDA 12 and CUDA 13 NCCL wheels are installed, they can overwrite the same libnccl.so.2 path." >&2
      echo "Reinstall the NCCL wheel required by torch after any other nvidia-nccl-* wheel:" >&2
      echo "  uv --cache-dir /tmp/robocasa_uv_cache pip install --python ${python_bin} --force-reinstall --no-deps ${nccl_requirement}" >&2
      exit 1
    fi
  fi
}

count_visible_gpus() {
  local gpu_list_var raw_value item count

  for gpu_list_var in CUDA_VISIBLE_DEVICES SLURM_STEP_GPUS SLURM_JOB_GPUS; do
    raw_value="${!gpu_list_var:-}"
    if [[ -z "${raw_value}" ]]; then
      continue
    fi

    count=0
    IFS=',' read -r -a gpu_ids <<< "${raw_value}"
    for item in "${gpu_ids[@]}"; do
      item="${item//[[:space:]]/}"
      if [[ -n "${item}" && "${item}" != "-1" ]]; then
        count=$((count + 1))
      fi
    done

    printf '%s\n' "${count}"
    return 0
  done

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | awk 'NF {count += 1} END {print count + 0}'
    return 0
  fi

  printf '0\n'
}

check_training_environment

NUM_PROCESSES="${NUM_PROCESSES:-$(count_visible_gpus)}"

if [[ "${NUM_PROCESSES}" -lt 1 ]]; then
  echo "No GPUs detected from CUDA_VISIBLE_DEVICES/SLURM allocation/nvidia-smi." >&2
  echo "Set NUM_PROCESSES explicitly or make GPUs visible before launching accelerate." >&2
  exit 1
fi

echo "Launching with NUM_PROCESSES=${NUM_PROCESSES}"

if [[ "${NUM_PROCESSES}" -eq 1 ]]; then
  "${python_bin}" -m training.bc_task_vlm.main "$@"
else
  # Avoid NCCL cuMem host allocation crashes seen on some Slurm/GH nodes.
  export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
  # This launcher is single-node. Avoid loading a site OFI net plugin that can
  # fail during initialization before NCCL falls back to intra-node transports.
  if [[ "${ROBOCASA_KEEP_NCCL_NET_PLUGIN:-false}" != "true" ]]; then
    export NCCL_NET_PLUGIN=none
  fi
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  echo "NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE}"
  echo "NCCL_NET_PLUGIN=${NCCL_NET_PLUGIN}"
  echo "NCCL_DEBUG=${NCCL_DEBUG}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "SLURM_STEP_GPUS=${SLURM_STEP_GPUS:-<unset>}"
  echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS:-<unset>}"
  "${python_bin}" -m accelerate.commands.launch \
    --config_file training/bc_task_vlm/accelerate_multigpu.yaml \
    --num_processes "${NUM_PROCESSES}" \
    -m training.bc_task_vlm.main \
    "$@"
fi
