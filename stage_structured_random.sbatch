#!/usr/bin/env bash
#SBATCH --job-name=stage-structured-random
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# Downloads the structured_random raw trajectory data from HF and builds a
# 21-traj/task capped training root (matching v1's actual training scale --
# robocasa_local_train_subset_21 -- so the v1-vs-v1.5 ablation on this
# source isn't confounded by a bigger training pool). The eval manifests
# (training/bc_task_vlm/eval_manifests/structured_random_30/) are already
# staged in this repo; this job only needs the raw trajectories.
#
# STRUCTURED_RANDOM_REPO_ID is a best guess (the manifests-only repo) --
# override it once the actual raw-data repo id is known, e.g.:
#   sbatch --export=STRUCTURED_RANDOM_REPO_ID=DorianAtSchool/robocasa-structured-random-data stage_structured_random.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/hf
export HF_HOME=/work/umass/shlomo_umass/hf_cache
# Xet backend (HF's default accelerated transfer path) has been flaky here
# (429s, then a 500 from the CDN) -- plain HTTP is slower but more reliable.
export HF_HUB_DISABLE_XET=1

repo_id="${STRUCTURED_RANDOM_REPO_ID:-DorianAtSchool/robocasa-structured-random-30}"
raw_root=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_raw
train_root=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_train_21
selection=data_analysis/analysis/held_out_task_selection/held_out_task_selection.json

echo "Downloading ${repo_id} -> ${raw_root}"
mkdir -p "${raw_root}"
"${HF_BIN}" download "${repo_id}" --repo-type dataset \
  --local-dir "${raw_root}" --max-workers 2

echo "Resolving task-directory root under ${raw_root}"
resolved_root="$("${PYTHON_BIN}" resolve_dataset_root.py --search-root "${raw_root}" --selection "${selection}")"
echo "Resolved data root: ${resolved_root}"

# Stable symlink so eval sbatch scripts don't need to know the (possibly
# nested) resolved path -- always /work/.../data/structured_random_eval_root.
eval_root_link=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_eval_root
ln -sfn "${resolved_root}" "${eval_root_link}"
echo "Eval dataset root symlink: ${eval_root_link} -> ${resolved_root}"

echo "Building ${train_root} (21 traj/task, train tasks only)"
"${PYTHON_BIN}" cap_trajectories_per_task.py \
  --source-root "${resolved_root}" \
  --output-root "${train_root}" \
  --selection "${selection}" \
  --trajectories-per-task 21

echo "Done. Train root: ${train_root}; eval/dataset root: ${resolved_root}"
