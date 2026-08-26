#!/usr/bin/env bash

# Submit a Slurm job, retrying only the transient per-user GRES/QOS rejection
# seen when a just-finished GPU job has not yet released its accounting slot.
submit_with_qos_retry() {
  local attempt=1
  local max_attempts=${SLURM_SUBMIT_MAX_ATTEMPTS:-180}
  local retry_seconds=${SLURM_SUBMIT_RETRY_SECONDS:-60}
  local output status

  while true; do
    set +e
    output=$(sbatch --parsable "$@" 2>&1)
    status=$?
    set -e
    if (( status == 0 )); then
      printf '%s\n' "${output%%;*}"
      return 0
    fi
    if [[ "${output}" != *QOSMaxGRESPerUser* && "${output}" != *"violates accounting/QOS policy"* ]]; then
      printf '%s\n' "${output}" >&2
      return "${status}"
    fi
    if (( attempt >= max_attempts )); then
      printf 'Slurm GPU quota still unavailable after %d attempts: %s\n' "${attempt}" "${output}" >&2
      return "${status}"
    fi
    printf 'GPU quota has not released yet; retrying submission in %ss (attempt %d/%d).\n' \
      "${retry_seconds}" "${attempt}" "${max_attempts}" >&2
    sleep "${retry_seconds}"
    ((attempt += 1))
  done
}
