#!/usr/bin/env bash
#SBATCH --job-name=v3-matrix
#SBATCH --partition=b200-batch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:b200:4
#SBATCH --mem=512G
#SBATCH --time=12:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

# One script for the whole v3 matrix:
#   data source  : verbalized_sampling | structured_random
#   observability: centralized (--predict-acting-agent) | partial (--partial-history)
#   reasoning    : off | on (--train-reasoning)
#
# 4 GPUs rather than 8: whole-node 8-GPU requests could not backfill on
# b200-batch (656 pending / 79 running, every node mix|alloc; SLURM projected
# a ~7h wait). Effective batch is held constant at 64 by doubling grad-accum
# (4 procs x 8 per-device x 2 accum == 8 x 8 x 1).
#
# Partial-observability uses --partial-history WITHOUT --predict-acting-agent:
# dataset.py forbids partial_history+predict_agent ("the caller is the actor"
# -- there is no agent to predict when each example is one agent's own view),
# but explicitly supports partial_history+train_get_image.
#
# Select a cell with --export, e.g.:
#   sbatch --export=ALL,SOURCE=structured_random,OBS=partial,REASONING=1 run_v3_matrix.sbatch
set -euo pipefail
mkdir -p /work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs

: "${SOURCE:?Set SOURCE to verbalized_sampling or structured_random}"
: "${OBS:?Set OBS to centralized or partial}"
REASONING=${REASONING:-0}
PARTIAL_STEP_INDEX_MODE=${PARTIAL_STEP_INDEX_MODE:-local}
SMOKE=${SMOKE:-0}
RUN_SUFFIX=${RUN_SUFFIX:-}
TWO_GPU_BENCHMARK_STEPS=${TWO_GPU_BENCHMARK_STEPS:-0}
NUM_EPOCHS=${NUM_EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-2e-4}

# Non-default ablations must never reuse a baseline output/W&B name.  Keep a
# caller-provided suffix, but also encode the two training knobs automatically.
if [[ "${NUM_EPOCHS}" != "3" || "${LEARNING_RATE}" != "2e-4" ]]; then
  lr_tag=${LEARNING_RATE//./p}
  RUN_SUFFIX="${RUN_SUFFIX}-ep${NUM_EPOCHS}-lr${lr_tag}"
fi

case "${PARTIAL_STEP_INDEX_MODE}" in
  none|local) ;;
  *) echo "PARTIAL_STEP_INDEX_MODE must be none or local, got: ${PARTIAL_STEP_INDEX_MODE}" >&2; exit 1 ;;
esac

export PYTHON_BIN=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
export HF_HOME=/work/umass/shlomo_umass/hf_cache
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# Fail before allocating model weights if online W&B authentication or API
# connectivity is broken. A successful import alone is not enough: smoke and
# full runs must be visible remotely, not left in an offline spool directory.
"${PYTHON_BIN}" -c 'import wandb; assert wandb.login(), "wandb.login() failed"; print("W&B viewer:", wandb.Api().viewer)'

case "${SOURCE}" in
  verbalized_sampling)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/robocasa_local_train_subset_21
    src_tag=vs
    ;;
  structured_random)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/data/structured_random_train_21
    src_tag=sr
    ;;
  tick30)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_concurrent
    src_tag=tick30
    ;;
  tick100)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_structured_random_concurrent
    src_tag=tick100
    ;;
  tick100_from150)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent
    src_tag=tick100-from150
    ;;
  tick100_from150_v3)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3
    src_tag=tick100-from150-v3
    ;;
  tick30_revalidated_v3)
    export DATA_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick_render30_revalidated_v3
    src_tag=tick30-revalidated-v3
    ;;
  *) echo "Unknown SOURCE: ${SOURCE}" >&2; exit 1 ;;
esac

obs_args=()
case "${OBS}" in
  centralized)
    obs_args=(--predict-acting-agent --train-get-image)
    obs_tag=cent
    ;;
  partial)
    obs_args=(
      --partial-history
      --train-get-image
      --partial-step-index-mode "${PARTIAL_STEP_INDEX_MODE}"
      --partial-observation-mode consume-once
    )
    obs_tag="partial-idx${PARTIAL_STEP_INDEX_MODE}"
    ;;
  *) echo "Unknown OBS: ${OBS}" >&2; exit 1 ;;
esac

# TASK_COMPLETE=0 drops the synthetic task_complete tool + terminal example
# from centralized cells, making them structurally comparable to partial-obs
# (which never offers task_complete). Centralized cells measurably over-use it:
# they declared completion at 0.27-0.45 while succeeding at 0.05-0.40, and each
# premature declaration ends the episode and forfeits remaining budget.
tc_args=()
tc_tag=""
if [[ "${TASK_COMPLETE:-1}" == "0" ]]; then
  tc_args=(--no-predict-task-complete)
  tc_tag="-notc"
fi

reason_args=()
reason_tag=noreason
if [[ "${REASONING}" == "1" ]]; then
  reason_args=(--train-reasoning)
  reason_tag=reason
fi

# Gradient checkpointing is disabled for CENTRALIZED cells (partial-obs takes
# the opposite branch below -- read both). It interacts badly with
# ddp_find_unused_parameters=True (hardcoded in _build_training_arguments)
# whenever some batches leave LoRA params untouched, and it fails two
# different ways:
#   - hard crash, "Expected to mark a variable ready only once"
#     (structured_random v1/v1.5 2026-07-29; v3+reasoning job 239494)
#   - silent hang, no output at all, until the NCCL timeout
#     (v3 no-reasoning job 239493 -- 11 min of silence where the first loss
#     line was due at ~69s, so a hang, not slow progress)
# The hang is why this is applied unconditionally rather than only to the
# combinations already seen to crash: a cell that merely hangs looks like a
# healthy queued job until it burns its whole walltime. Non-reentrant
# checkpointing was tried and did NOT fix it.
#
# Without checkpointing activation memory rises, so per-device batch halves
# (8->4) and grad-accum doubles (2->4), holding effective batch at 64.
ckpt_args=(--no-gradient-checkpointing)
NO_CKPT=1

# ...except partial-observability, which needs the opposite treatment.
# The only partial-obs run known to have trained successfully
# (qwen3vl-8b-v3-partial-observability) used checkpointing ON with
# grad_accum=1; this script's no-checkpointing + accum-8 config hung both
# verbalized partial cells (239685/239686: zero loss lines, silent from the
# second trainer.train() logged, while both centralized cells on the same
# code were happily logging loss).
#
# Suspected mechanism: with find_unused_parameters=True, DDP re-derives the
# unused-param set every forward. consume_once hands an image to exactly one
# following call, so a whole accumulation micro-batch can be image-free --
# ranks then disagree about whether the vision tower participated and the
# allreduce deadlocks. accum=1 gives that disagreement no window to open,
# which is what the working run had.
if [[ "${OBS}" == "partial" ]]; then
  # accum=1 did NOT help: 239866/239867/239868 all crashed "marked as ready
  # twice" at accum=1 with checkpointing on, so accumulation was never the
  # variable. That error is precisely the find_unused_parameters=True +
  # checkpointing incompatibility, and smoke test 239769 showed
  # --no-ddp-find-unused-parameters clears it (checkpointing ON, batch 8,
  # 30 steps, loss logging normally).
  #
  # This likely also explains the earlier no-checkpointing HANG: with
  # find_unused_parameters=True, ranks independently derive which params went
  # ungradiented, and consume_once makes whole micro-batches image-free, so
  # ranks can disagree and the allreduce deadlocks. Turning the detection off
  # removes both failure modes.
  #
  # Residual risk, stated plainly: if a partial batch genuinely leaves params
  # without gradients, find_unused=False raises "Expected to have finished
  # reduction in the prior iteration" instead. That is a loud, immediate
  # failure, not a silent wrong result.
  ckpt_args=(--gradient-checkpointing --no-ddp-find-unused-parameters)
  NO_CKPT=0
fi

if [[ "${SOURCE}" == "tick30" && "${OBS}" == "partial" ]]; then
  if [[ "${PARTIAL_STEP_INDEX_MODE}" == "none" ]]; then
    export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick30_train47_toolcall_idxnone
  else
    export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick30_train47_toolcall_idxlocal
  fi
fi
if [[ "${SOURCE}" == "tick100" && "${OBS}" == "partial" ]]; then
  if [[ "${PARTIAL_STEP_INDEX_MODE}" != "none" ]]; then
    echo "tick100 currently has only the validated no-index artifact" >&2
    exit 1
  fi
  export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick100_train47_toolcall_idxnone
fi
if [[ "${SOURCE}" == "tick100_from150" && "${OBS}" == "partial" ]]; then
  if [[ "${PARTIAL_STEP_INDEX_MODE}" != "none" ]]; then
    echo "tick100_from150 currently has only the validated no-index artifact" >&2
    exit 1
  fi
  export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick100_from150_train47_toolcall_idxnone
fi
if [[ "${SOURCE}" == "tick100_from150_v3" && "${OBS}" == "partial" ]]; then
  if [[ "${PARTIAL_STEP_INDEX_MODE}" != "none" ]]; then
    echo "tick100_from150_v3 forbids local and global step indices" >&2
    exit 1
  fi
  export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick100_from150_v3_train47_toolcall_idxnone_schemafix
fi
if [[ "${SOURCE}" == "tick30_revalidated_v3" && "${OBS}" == "partial" ]]; then
  if [[ "${PARTIAL_STEP_INDEX_MODE}" != "none" ]]; then
    echo "tick30_revalidated_v3 forbids local and global step indices" >&2
    exit 1
  fi
  export PREPROCESSED_DATA_DIR=/work/umass/shlomo_umass/dbenhamougol_umass/preprocessed/tick30_revalidated_v3_train47_toolcall_idxnone_schemafix
fi

run_name="qwen3vl-8b-v3-${src_tag}-${obs_tag}-${reason_tag}${tc_tag}"
if [[ "${SMOKE}" == "1" ]]; then
  run_name="${run_name}-smoke"
fi
if [[ "${TWO_GPU_BENCHMARK_STEPS}" -gt 0 ]]; then
  run_name="${run_name}-2gpu-bench${TWO_GPU_BENCHMARK_STEPS}"
fi
run_name="${run_name}${RUN_SUFFIX}"
export OUTPUT_DIR="training/bc_task_vlm/runs/${run_name}"
export WANDB_RUN_NAME="${run_name}"

# --- 4 GPUs, effective batch 64 (== the 8-GPU config's 8x8x1) ---
# Without checkpointing every activation is retained, and v3 samples carry
# extra get_image frames, so per-device batch has to come all the way down to
# 2: batch 4 OOM'd on a 178GB B200 (job 239629 -- 173GB in use, tried to
# allocate another 8.18GB). accum 8 keeps the effective batch at 64.
export NUM_PROCESSES=4
if [[ "${NO_CKPT}" == "1" ]]; then
  export PER_DEVICE_BATCH_SIZE=2
  export GRAD_ACCUM=8
else
  # partial-obs: accum MUST be 1. ckpt+accum2 crashed "marked as ready twice"
  # (job 239825) exactly as ckpt+accum8 did; the one partial-obs run that ever
  # completed used accum=1. With accumulation, HF wraps non-final micro-steps
  # in no_sync, changing when DDP prepares for backward, and checkpointing's
  # recomputation hooks then fire where DDP does not expect them. accum=1
  # removes that window. batch 16 keeps the effective batch at 64 on 4 GPUs.
  export PER_DEVICE_BATCH_SIZE=8
  export GRAD_ACCUM=2
fi
if [[ "${SMOKE}" == "1" ]]; then
  export NUM_PROCESSES=1
  export PER_DEVICE_BATCH_SIZE=1
  export GRAD_ACCUM=1
  # This job allocation contains four GPUs, but a one-process smoke run must
  # see only one of them. Otherwise Transformers wraps the VLM in DataParallel
  # and independently shards image placeholders and flattened image features,
  # producing false image-token/image-feature mismatch failures.
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
  else
    export CUDA_VISIBLE_DEVICES=0
  fi
fi
if [[ "${TWO_GPU_BENCHMARK_STEPS}" -gt 0 ]]; then
  # Preserve the production effective batch of 64:
  # 2 ranks x 8 samples x 4 accumulation steps.
  export NUM_PROCESSES=2
  export PER_DEVICE_BATCH_SIZE=8
  export GRAD_ACCUM=4
fi
# Recommended by the OOM message itself; costs nothing when memory is ample.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# The working partial-obs run allowed 5400s; 1800 is tight for a slow first
# collective and turns a recoverable stall into a dead job.
export DDP_TIMEOUT_SECONDS=5400
export ATTN_IMPLEMENTATION=flash_attention_2
export IMAGE_RESOLUTION=512
export MAX_LENGTH=8192

export NUM_EPOCHS
export LEARNING_RATE
export EVAL_STEPS=2000
export SAVE_STEPS=2000
export SAVE_TOTAL_LIMIT=2
export EVAL_MAX_SAMPLES=512
export EVAL_GENERATION_MAX_SAMPLES=0
if [[ "${SMOKE}" == "1" ]]; then
  export EVAL_STEPS=5
  export SAVE_STEPS=5
  export SAVE_TOTAL_LIMIT=1
  export EVAL_MAX_SAMPLES=16
fi
if [[ "${TWO_GPU_BENCHMARK_STEPS}" -gt 0 ]]; then
  export EVAL_STEPS="${TWO_GPU_BENCHMARK_STEPS}"
  export SAVE_STEPS="${TWO_GPU_BENCHMARK_STEPS}"
  export SAVE_TOTAL_LIMIT=1
  export EVAL_MAX_SAMPLES=32
fi

echo "=== ${run_name} ==="
echo "  DATA_ROOT=${DATA_ROOT}"
echo "  obs=${obs_args[*]}"
echo "  reasoning=${reason_args[*]:-none}  ckpt=${ckpt_args[*]:-default}"
echo "  epochs=${NUM_EPOCHS}  learning_rate=${LEARNING_RATE}"

source training/scripts/local/_launch_args.sh
launch_args+=(
  "${obs_args[@]}"
  "${reason_args[@]}"
  ${tc_args[@]+"${tc_args[@]}"}
  "${ckpt_args[@]}"
  --ddp-timeout-seconds "${DDP_TIMEOUT_SECONDS}"
  --num-epochs "${NUM_EPOCHS}"
  --eval-steps "${EVAL_STEPS}"
  --save-steps "${SAVE_STEPS}"
  --save-total-limit "${SAVE_TOTAL_LIMIT}"
  --eval-max-samples "${EVAL_MAX_SAMPLES}"
  --eval-generation-max-samples "${EVAL_GENERATION_MAX_SAMPLES}"
  --training-samples-cache-dir ".cache/bc_task_vlm/examples_${run_name}"
)

if [[ "${SMOKE}" == "1" ]]; then
  launch_args+=(--max-steps 10)
fi
if [[ "${TWO_GPU_BENCHMARK_STEPS}" -gt 0 ]]; then
  launch_args+=(--max-steps "${TWO_GPU_BENCHMARK_STEPS}")
fi

exec training/scripts/launch_multigpu.sh "${launch_args[@]}"
