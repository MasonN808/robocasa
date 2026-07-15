#!/usr/bin/env bash
# LOCAL GPU BOX (e.g. RTX 5090, 32 GB). LoRA SFT of Qwen3-VL-8B on a staged
# N-traj/task subset (see stage_subset.sh), targeting a couple of epochs in an
# overnight window.
#
# Prereqs on the local box: a Python env with torch (CUDA build for your GPU),
# transformers, peft, accelerate, xgrammar; the repo checked out; and the
# subset rsync'd locally. Point DATA_ROOT at it.
#
# Throughput on a fresh box is unknown, so run the probe FIRST:
#   PROBE=1 DATA_ROOT=~/robocasa_local_train_subset bash training/scripts/local/train_qwen3vl_8b.sh
# It trains ~40 steps, prints measured samples/sec, and tells you how many
# traj/task give ~3 epochs in your target window. Then launch the real run:
#   DATA_ROOT=~/robocasa_local_train_subset bash training/scripts/local/train_qwen3vl_8b.sh
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"

model_path="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
data_root="${DATA_ROOT:?Set DATA_ROOT to the staged subset dir}"
selection="${HELD_OUT_SELECTION:-${repo_root}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json}"
output_dir="${OUTPUT_DIR:-${repo_root}/training/bc_task_vlm/runs/qwen3vl-8b-local-sft}"

# Keep HF downloads off the system partition.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export ROBOCASA_VALIDATE_IMAGE_PATHS="${ROBOCASA_VALIDATE_IMAGE_PATHS:-false}"

per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
grad_accum="${GRAD_ACCUM:-4}"
num_epochs="${NUM_EPOCHS:-3}"
image_resolution="${IMAGE_RESOLUTION:-512}"

read_train_tasks='import json,sys; print(",".join(json.load(open(sys.argv[1]))["train_tasks"]))'
train_tasks="$("${python_bin}" -c "${read_train_tasks}" "${selection}")"

launch_args=(
  --dataset-root "${data_root}"
  --model-name-or-path "${model_path}"
  --processor-name-or-path "${model_path}"
  --sft-format tool_call
  --image-resolution "${image_resolution}"
  --train-tasks "${train_tasks}"
  --val-tasks "${train_tasks}"
  --validation-trajectories-per-task 0
  --validation-trajectory-fraction "${VAL_TRAJECTORY_FRACTION:-0.1}"
  --validation-split-seed "${VAL_SPLIT_SEED:-42}"
  --output-dir "${output_dir}"
  --per-device-batch-size "${per_device_batch}"
  --grad-accum "${grad_accum}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --max-length "${MAX_LENGTH:-8192}"
  --max-images-per-sample "${MAX_IMAGES_PER_SAMPLE:-4}"
  --num-workers "${NUM_WORKERS:-8}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --bf16
  --gradient-checkpointing
  --attn-implementation sdpa
  --supervise-last-assistant-turn-only
  --logging-steps "${LOGGING_STEPS:-10}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --wandb-mode "${WANDB_MODE:-offline}"
  --wandb-run-name "${WANDB_RUN_NAME:-qwen3vl-8b-local-sft}"
)

if [[ -n "${MAX_STEPS:-}" ]]; then
  launch_args+=(--max-steps "${MAX_STEPS}")
fi

if [[ "${PROBE:-0}" == "1" ]]; then
  echo "=== throughput probe: ~40 steps, then exits ==="
  eff_batch=$((per_device_batch * grad_accum))
  start=$(date +%s)
  "${python_bin}" -m training.bc_task_vlm.main \
    "${launch_args[@]}" --num-epochs 1 --max-steps 40 \
    --eval-steps 100000 --save-steps 100000 --save-total-limit 1
  elapsed=$(( $(date +%s) - start ))
  echo ""
  echo "=== probe done in ${elapsed}s for ~$((40 * eff_batch)) samples ==="
  probe_report='
import sys
elapsed, eff_batch = int(sys.argv[1]), int(sys.argv[2])
samples = 40 * eff_batch
sps = samples / max(elapsed, 1)
warm = samples / max(elapsed - 60, 1)  # crude subtraction of load/compile
n_tasks, steps_per_traj = 47, 15
print("measured ~%.2f samples/s (warm est ~%.2f/s)" % (sps, warm))
for hours in (8, 10):
    for epochs in (2, 3):
        tpt = warm * hours * 3600 / (epochs * steps_per_traj * n_tasks)
        print("  for %d epochs in %dh -> ~%.0f traj/task" % (epochs, hours, tpt))
'
  "${python_bin}" -c "${probe_report}" "${elapsed}" "${eff_batch}"
  exit 0
fi

echo "Training ${model_path} (LoRA) on ${data_root}"
echo "  epochs=${num_epochs} per_device=${per_device_batch} grad_accum=${grad_accum} res=${image_resolution}"
exec "${python_bin}" -m training.bc_task_vlm.main \
  "${launch_args[@]}" \
  --num-epochs "${num_epochs}" \
  --eval-steps "${EVAL_STEPS:-500}" \
  --save-steps "${SAVE_STEPS:-500}" \
  --save-total-limit "${SAVE_TOTAL_LIMIT:-2}" \
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES:-128}"
