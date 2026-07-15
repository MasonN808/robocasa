#!/usr/bin/env bash
# LOCAL. Measures training throughput on THIS box, then reports how many
# trajectories/task give 2-3 epochs in an 8-10h window. Run this before
# train_qwen3vl_8b.sh so the subset size matches your GPU's real speed.
#
# Trains ~40 steps and exits (no checkpoint kept).
#
# Usage:
#   DATA_ROOT=~/robocasa_local_train_subset \
#     bash training/scripts/local/probe_throughput.sh
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_launch_args.sh"

probe_steps="${PROBE_STEPS:-40}"
echo "=== throughput probe: ${probe_steps} steps on ${model_path} (eff batch ${eff_batch}) ==="
start=$(date +%s)
"${python_bin}" -m training.bc_task_vlm.main \
  "${launch_args[@]}" \
  --num-epochs 1 --max-steps "${probe_steps}" \
  --eval-steps 100000 --save-steps 100000 --save-total-limit 1
elapsed=$(( $(date +%s) - start ))

echo ""
report='
import sys
elapsed, steps, eff_batch = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
samples = steps * eff_batch
sps = samples / max(elapsed, 1)
warm = samples / max(elapsed - 60, 1)  # crude subtraction of load/compile
n_tasks, steps_per_traj = 47, 15
print("probe: %d samples in %ds -> ~%.2f samples/s (warm est ~%.2f/s)"
      % (samples, elapsed, sps, warm))
print("suggested TRAJ_PER_TASK (re-stage if different):")
for hours in (8, 10):
    for epochs in (2, 3):
        tpt = warm * hours * 3600 / (epochs * steps_per_traj * n_tasks)
        print("  %d epochs in %dh -> ~%.0f traj/task" % (epochs, hours, tpt))
'
"${python_bin}" -c "${report}" "${elapsed}" "${probe_steps}" "${eff_batch}"
