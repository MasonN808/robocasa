#!/usr/bin/env bash
#SBATCH --job-name=pull-vllm-container
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail
VLLM_VERSION=${VLLM_VERSION:-0.12.0}
target=/work/umass/shlomo_umass/dbenhamougol_umass/containers/vllm-openai-v${VLLM_VERSION}.sif
mkdir -p "$(dirname "${target}")" /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs
if [[ ! -f "${target}" ]]; then
  apptainer pull "${target}" "docker://vllm/vllm-openai:v${VLLM_VERSION}"
fi
apptainer inspect "${target}" >/dev/null
echo "Validated ${target}"
