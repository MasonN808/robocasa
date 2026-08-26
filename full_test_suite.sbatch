#!/bin/bash
#SBATCH --job-name=pytest-all
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/pytest-all-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/pytest-all-%j.err

# The suite does not fit in the login node's memory cap -- it dies at ~23% with
# SIGKILL, which reads exactly like a hang and is not a test result.
#
# Two files are excluded for upstream breakage unrelated to this work:
#   test_dataset_playback.py  -- no module robocasa.scripts.playback_dataset
#   test_env_determinism.py   -- robosuite dropped load_controller_config

set -u
cd /work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
export PYTHONWARNINGS=ignore
# Do NOT export MUJOCO_GL here. Forcing osmesa on the cpu partition makes every
# mujoco import die in OpenGL ("'NoneType' object has no attribute glGetError")
# and takes 20 test files down at collection. Left unset, the tests import fine.

PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

"$PY" -m pytest tests/ -q -p no:cacheprovider \
  --ignore=tests/test_dataset_playback.py \
  --ignore=tests/test_env_determinism.py \
  -rf || echo "pytest exited $?"

echo PYTEST DONE
