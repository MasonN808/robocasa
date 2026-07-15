#!/usr/bin/env bash
#SBATCH --job-name=traj-gen-smoke
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64g
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=dbenhamougoldfajn@umass.edu
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/slurm-%j.out

set -euo pipefail

venv_path="/work/hdd/bgjs/dbenhamougoldfajn/robocasa/.venv"
if [[ -d "$venv_path" ]]; then
  source "$venv_path/bin/activate"
else
  echo "Warning: venv not found at $venv_path" >&2
fi

export HF_HOME="/work/hdd/bgjs/.cache/huggingface"

echo "Job started at: $(date)"
start_time=$(date +%s)

time python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 5 \
  --random-start-location true \
  --parallelize-tasks \
  --sampling verbalized \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 16 \
  --max-retries 5 \
  --enable-validation

end_time=$(date +%s)
echo "Job finished at: $(date)"
echo "Total runtime: $((end_time - start_time)) seconds"

echo "Trajectory generation smoke test completed!"