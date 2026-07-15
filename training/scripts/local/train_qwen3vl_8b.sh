#!/usr/bin/env bash
# LOCAL GPU BOX (e.g. RTX 5090, 32 GB). LoRA SFT of Qwen3-VL-8B on a staged
# N-traj/task subset, targeting a couple of epochs in an overnight window.
#
# Run probe_throughput.sh first to size TRAJ_PER_TASK to this box, then:
#   DATA_ROOT=~/robocasa_local_train_subset \
#     bash training/scripts/local/train_qwen3vl_8b.sh
#
# Defaults: Qwen3-VL-8B, LoRA r=16, 512² images, bf16, gradient checkpointing,
# sdpa attention (no flash-attn), 3 epochs, effective batch 8. Override via env
# (MODEL_PATH, NUM_EPOCHS, IMAGE_RESOLUTION, PER_DEVICE_BATCH_SIZE, MAX_STEPS…).
# Set MAX_STEPS to cap wall-clock regardless of epochs; training stops at
# whichever of --num-epochs / MAX_STEPS comes first.
set -euo pipefail

source "$(dirname "${BASH_SOURCE[0]}")/_launch_args.sh"

num_epochs="${NUM_EPOCHS:-3}"
if [[ -n "${MAX_STEPS:-}" ]]; then
  launch_args+=(--max-steps "${MAX_STEPS}")
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
