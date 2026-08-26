#!/usr/bin/env bash
# Sourced by probe_throughput.sh and train_qwen3vl_8b.sh. Defines the shared
# training configuration so the two entry points can't drift apart.
#
# After sourcing, the caller has: python_bin, model_path, data_root,
# output_dir, per_device_batch, grad_accum, eff_batch, image_resolution, and
# the launch_args[] base array (everything except epoch/step/eval cadence).
#
# Not meant to be run directly.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-python}"

model_path="${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
data_root="${DATA_ROOT:?Set DATA_ROOT to the staged subset dir}"
selection="${HELD_OUT_SELECTION:-${repo_root}/data_analysis/analysis/held_out_task_selection/held_out_task_selection.json}"
output_dir="${OUTPUT_DIR:-${repo_root}/training/bc_task_vlm/runs/qwen3vl-8b-local-sft}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export ROBOCASA_VALIDATE_IMAGE_PATHS="${ROBOCASA_VALIDATE_IMAGE_PATHS:-false}"

per_device_batch="${PER_DEVICE_BATCH_SIZE:-2}"
grad_accum="${GRAD_ACCUM:-4}"
eff_batch=$((per_device_batch * grad_accum))
image_resolution="${IMAGE_RESOLUTION:-512}"

read_train_tasks='import json,sys; print(",".join(json.load(open(sys.argv[1]))["train_tasks"]))'
train_tasks="$("${python_bin}" -c "${read_train_tasks}" "${selection}")"
val_tasks="${VAL_TASKS-${train_tasks}}"

launch_args=(
  --dataset-root "${data_root}"
  --model-name-or-path "${model_path}"
  --processor-name-or-path "${model_path}"
  --sft-format tool_call
  --image-resolution "${image_resolution}"
  --train-tasks "${train_tasks}"
  --val-tasks "${val_tasks}"
  --validation-trajectories-per-task 0
  --validation-trajectory-fraction "${VAL_TRAJECTORY_FRACTION:-0.1}"
  --validation-split-seed "${VAL_SPLIT_SEED:-42}"
  --output-dir "${output_dir}"
  --per-device-batch-size "${per_device_batch}"
  --grad-accum "${grad_accum}"
  --train-sampling-strategy "${TRAIN_SAMPLING_STRATEGY:-random}"
  --learning-rate "${LEARNING_RATE:-2e-4}"
  --weight-decay "${WEIGHT_DECAY:-0.0}"
  --max-length "${MAX_LENGTH:-8192}"
  --max-images-per-sample "${MAX_IMAGES_PER_SAMPLE:-4}"
  --num-workers "${NUM_WORKERS:-8}"
  --lora-r "${LORA_R:-16}"
  --lora-alpha "${LORA_ALPHA:-32}"
  --bf16
  --gradient-checkpointing
  --attn-implementation "${ATTN_IMPLEMENTATION:-sdpa}"
  --supervise-last-assistant-turn-only
  --logging-steps "${LOGGING_STEPS:-10}"
  --warmup-ratio "${WARMUP_RATIO:-0.03}"
  --wandb-mode "${WANDB_MODE:-offline}"
  --wandb-run-name "${WANDB_RUN_NAME:-qwen3vl-8b-local-sft}"
)

if [[ -n "${PREPROCESSED_DATA_DIR:-}" ]]; then
  launch_args+=(--preprocessed-data-dir "${PREPROCESSED_DATA_DIR}")
fi

# Optional: a smaller eval micro-batch than the train micro-batch, to keep the
# full-vocab eval loss within VRAM without slowing training. Unset by default
# (eval batch falls back to the train batch), so v1 behavior is unchanged.
if [[ -n "${PER_DEVICE_EVAL_BATCH_SIZE:-}" ]]; then
  launch_args+=(--per-device-eval-batch-size "${PER_DEVICE_EVAL_BATCH_SIZE}")
fi
