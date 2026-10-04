#!/usr/bin/env bash
#SBATCH --job-name=thinkrat43-four
#SBATCH --partition=rtx-batch
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%A_%a.err
set -euo pipefail
scales=(60 90 120)
checkpoints=(1298 1945 2593)
scale=${scales[${SLURM_ARRAY_TASK_ID:?}]}
checkpoint=${checkpoints[${SLURM_ARRAY_TASK_ID}]}
repo=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
export MODEL=Qwen/Qwen3-VL-8B-Thinking
export ADAPTER=${repo}/training/bc_task_vlm/runs/qwen3vl8b-thinking-tick150-s${scale}-rationale-train43-held10-ep1-halfckpt-lr1e4-wd001-noindex-promptv9/checkpoint-${checkpoint}
export MANIFEST=${repo}/training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json
export COHORT_SPLIT=heldout_task_types COHORT_MODE=sampled
export TASKS=arrange_bread_bowl,distribute_chicken,prepare_cheese_station,serve_meal_juice
export RUN_NAME=thinking_rationale_s${scale}_native43_four_tasks_ep1p0_job${SLURM_ARRAY_JOB_ID}
export OUTPUT_DIR=${repo}/training/bc_task_vlm/eval_runs/fixed_live_sim/${RUN_NAME}/${COHORT_SPLIT}
export DATASET_ROOT=/work/umass/shlomo_umass/dbenhamougol_umass/tick53x150_state_grounded_cascade_v1_rendered
export SCENE_COMPATIBILITY_CACHE=${repo}/training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json
export EPISODES_PER_TASK=10 EPISODES_PER_CONFIG=1 WORKERS=4
export EVALUATION_SEED=20260817 SCENE_SAMPLING_SEED=20260819
export ENABLE_THINKING=true MAX_NEW_TOKENS=2048
export VLLM_TEMPERATURE=0.6 VLLM_TOP_P=0.95 VLLM_TOP_K=20
export VLLM_REASONING_PARSER=qwen3 VLLM_TOOL_CALL_PARSER=qwen3_xml
export COMMUNICATION_MODE=full RECOVER_MISSING_OPEN_TOOL_TAG=false
export VLLM_RAW_TEXT_MODE=false ENABLE_TOWER_CONNECTOR_LORA=false
export VLLM_IMAGE=/work/umass/shlomo_umass/dbenhamougol_umass/containers/vllm-openai-v0.27.1.sif
export VLLM_MAX_MODEL_LEN=16384 VLLM_GPU_MEMORY_UTILIZATION=0.70
export VLLM_ENABLE_AUTO_TOOL_CHOICE=true
export EXTRA_ARGS='' REJECTION_MODE='' MAX_CONFIGS_PER_TASK=''
export DAGGER_LOOP_ARGS='' DAGGER_COLLECT_ARGS=''
/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python - <<'PY'
import json, os
from pathlib import Path
ck = Path(os.environ['ADAPTER'])
state = json.loads((ck / 'trainer_state.json').read_text())
assert abs(state['epoch'] - 1.0) < 1e-8, state['epoch']
assert (ck / 'adapter_model.safetensors').is_file()
print('Verified one-epoch adapter:', ck, flush=True)
PY
exec bash fixed_live_sim_open_weight_vllm.sbatch
