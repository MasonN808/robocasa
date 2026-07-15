#!/usr/bin/env bash
#SBATCH --job-name=robocasa-build-manifests
#SBATCH --account=bgjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --chdir=/work/hdd/bgjs/dbenhamougoldfajn/robocasa
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err

# Builds the SFT-necessity eval manifests against the FULL dataset (assembled
# on node-local /tmp from local tasks + staged tars), so the trajectory-holdout
# split matches exactly what training sees. Afterwards exports just the
# trajectories referenced by any manifest to
# training/bc_task_vlm/eval_data_subset/ (~5 GB), so Gemini/pilot evals can
# run anywhere without re-assembling the 180 GB root.

set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
manifest_dir="${MANIFEST_DIR:-${repo_root}/training/bc_task_vlm/eval_manifests/exp52}"
subset_root="${EVAL_SUBSET_ROOT:-${repo_root}/training/bc_task_vlm/eval_data_subset}"
selection="${HELD_OUT_SELECTION:-${repo_root}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json}"

assemble_target="/tmp/experiment_52_${SLURM_JOB_ID:-$$}"
trap 'rm -rf "${assemble_target}"' EXIT
dataset_root="$(bash "${repo_root}/training/scripts/assemble_dataset_root.sh" "${assemble_target}")"

"${python_bin}" -m training.bc_task_vlm.build_eval_manifest \
  --dataset-root "${dataset_root}" \
  --selection "${selection}" \
  --trajectories-per-split "${TRAJECTORIES_PER_SPLIT:-75}" \
  --pilot-trajectories "${PILOT_TRAJECTORIES:-12}" \
  --output-dir "${manifest_dir}"

echo "Exporting referenced trajectories to ${subset_root}"
"${python_bin}" - "$dataset_root" "$manifest_dir" "$subset_root" <<'EOF'
import json
import shutil
import sys
from pathlib import Path

dataset_root, manifest_dir, subset_root = map(Path, sys.argv[1:4])
referenced: set[tuple[str, str]] = set()
for manifest_path in sorted(manifest_dir.glob("eval_manifest_*.json")):
    manifest = json.loads(manifest_path.read_text())
    for sample in manifest["samples"]:
        referenced.add((sample["task_name"], sample["trajectory_id"]))

copied = skipped = 0
for task_name, trajectory_id in sorted(referenced):
    source = dataset_root / task_name / trajectory_id
    target = subset_root / task_name / trajectory_id
    if target.exists():
        skipped += 1
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    copied += 1
print(f"eval subset: {copied} trajectories copied, {skipped} already present")
EOF

echo "Done. Manifests in ${manifest_dir}; eval images in ${subset_root}"
