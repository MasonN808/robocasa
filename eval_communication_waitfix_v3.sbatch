#!/usr/bin/env bash
#SBATCH --job-name=comm-waitfix-v3
#SBATCH --partition=rtx-batch
#SBATCH --array=0-11%6
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=03:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail

repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
cd "${repo}"
index=${SLURM_ARRAY_TASK_ID:?Run this launcher as a Slurm array}
launch_id=${SLURM_ARRAY_JOB_ID:?Run this launcher as a Slurm array}
(( index >= 0 && index < 12 )) || { echo "Invalid array index ${index}" >&2; exit 2; }
modes=(unguided minimal intermediate)
splits=(train_task_types heldout_task_types)
model_index=$((index / 6))
mode_index=$(((index % 6) / 2))
export COMMUNICATION_MODE=${modes[mode_index]}
export COHORT_SPLIT=${splits[index % 2]}

# Preserve the original 47/6 episode selections. Results may subsequently be
# regrouped into 43/10, as with the original communication comparison.
export MANIFEST=${repo}/training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json
export DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered
export SCENE_COMPATIBILITY_CACHE=${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json
export COHORT_MODE=sampled EPISODES_PER_TASK=10 EPISODES_PER_CONFIG=1
export EVALUATION_SEED=20260817 SCENE_SAMPLING_SEED=20260819 WORKERS=4
export MAX_NEW_TOKENS=256 ENABLE_THINKING=false VLLM_TEMPERATURE=0.0
export REJECTION_MODE=report-failed
export ADAPTER='' EXTRA_ARGS='' TASKS='' MAX_CONFIGS_PER_TASK=''
export DAGGER_LOOP_ARGS='' DAGGER_COLLECT_ARGS=''
export VLLM_REASONING_PARSER='' VLLM_TOOL_CALL_PARSER=hermes
export VLLM_RAW_TEXT_MODE=false RECOVER_MISSING_OPEN_TOOL_TAG=false
export VLLM_IMAGE=/work/umass/shlomo_umass/dbenhamougol_umass/containers/vllm-openai-v0.27.1.sif
export VLLM_MAX_MODEL_LEN=16384 VLLM_GPU_MEMORY_UTILIZATION=0.70
export PYTHONUNBUFFERED=1
python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python

# Fail before API use if queued jobs no longer match the reviewed prompt code.
"${python_bin}" - <<'PY'
import hashlib
import json
from pathlib import Path

snapshot = json.loads(Path('training/bc_task_vlm/reports/communication_waitfix_v3/prompt_snapshots.json').read_text())
for source, expected in snapshot['source_sha256'].items():
    actual = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f'Prompt/runtime source changed after review: {source}')
print('Reviewed prompt/runtime source hashes verified.')
PY

if (( model_index == 0 )); then
  export MODEL=Qwen/Qwen3-VL-8B-Instruct
  export RUN_NAME=ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_${COMMUNICATION_MODE}_waitfix_v3_job${launch_id}
  launcher=fixed_live_sim_open_weight_vllm.sbatch
else
  export BACKEND=gemini MODEL=gemini-3-flash-preview
  export RUN_NAME=ootb_gemini3flash_fixed10_promptv9_comm_${COMMUNICATION_MODE}_waitfix_v3_job${launch_id}
  launcher=fixed_live_sim_eval.sbatch
fi
export OUTPUT_DIR=${repo}/training/bc_task_vlm/eval_runs/fixed_live_sim/${RUN_NAME}/${COHORT_SPLIT}
echo "model=${MODEL} mode=${COMMUNICATION_MODE} split=${COHORT_SPLIT} workers=${WORKERS} output=${OUTPUT_DIR}"
exec bash "${launcher}"
