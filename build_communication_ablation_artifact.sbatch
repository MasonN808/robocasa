#!/usr/bin/env bash
#SBATCH --job-name=comm-ablation-report
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err

set -euo pipefail

python_bin=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
root=training/bc_task_vlm/eval_runs/fixed_live_sim
artifact=training/bc_task_vlm/eval_runs/fixed_live_sim_communication_ablation_artifact
cohort=training/bc_task_vlm/eval_manifests/fixed_live_sim_configuration_v1/frozen.json
mkdir -p "${artifact}/summaries"

models=(gemini3flash qwen3vl8b)
modes=(full minimal unguided none)

run_dir() {
  local model=$1
  local mode=$2
  if [[ "${model}" == gemini3flash ]]; then
    if [[ "${mode}" == full ]]; then
      echo ootb_gemini3flash_fixed10_promptv9
    else
      echo "ootb_gemini3flash_fixed10_promptv9_comm_${mode}_v2"
    fi
  elif [[ "${mode}" == full ]]; then
    echo ootb_qwen3vl8b_fixed10_promptv9_vllm
  else
    echo "ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_${mode}_v2"
  fi
}

label_for() {
  local model=$1
  local mode=$2
  local model_label='Base Qwen 3 VL 8B'
  [[ "${model}" == gemini3flash ]] && model_label='Gemini 3 Flash'
  printf '%s · %s communication' "${model_label}" "${mode}"
}

summary_args=()
for model in "${models[@]}"; do
  for mode in "${modes[@]}"; do
    run=$(run_dir "${model}" "${mode}")
    train_results="${root}/${run}/train_task_types/aggregate/live_sim_trajectories.jsonl"
    held_results="${root}/${run}/heldout_task_types/aggregate/live_sim_trajectories.jsonl"
    summary="${artifact}/summaries/${model}_${mode}.json"
    [[ -s "${train_results}" ]] || { echo "Missing ${train_results}" >&2; exit 1; }
    [[ -s "${held_results}" ]] || { echo "Missing ${held_results}" >&2; exit 1; }
    "${python_bin}" -m training.bc_task_vlm.summarize_fixed_live_sim \
      --results "${train_results}" \
      --results "${held_results}" \
      --label "$(label_for "${model}" "${mode}")" \
      --output-json "${summary}" \
      --output-md "${artifact}/summaries/${model}_${mode}.md"
    jq -e '
      .num_records == 530
      and .splits.train_task_types.num_episodes == 470
      and .splits.heldout_task_types.num_episodes == 60
      and .schema_version == 4
    ' "${summary}" >/dev/null
    summary_args+=(--summary "${summary}")
  done
done

# Add the scale-30 SFT checkpoint selected for the main comparison, while also
# retaining all four checkpoints for the dedicated epoch analysis.
epoch_summary_args=()
for epoch_tag in 0p5 1p0 1p5 2p0; do
  case "${epoch_tag}" in
    0p5) epoch_label=0.5 ;;
    1p0) epoch_label=1.0 ;;
    1p5) epoch_label=1.5 ;;
    2p0) epoch_label=2.0 ;;
  esac
  run="sft_scale30_ep${epoch_tag}_fixed10_promptv9"
  train_results="${root}/${run}/train_task_types/aggregate/live_sim_trajectories.jsonl"
  held_results="${root}/${run}/heldout_task_types/aggregate/live_sim_trajectories.jsonl"
  summary="${artifact}/summaries/sft_scale30_ep${epoch_tag}.json"
  [[ -s "${train_results}" ]] || { echo "Missing ${train_results}" >&2; exit 1; }
  [[ -s "${held_results}" ]] || { echo "Missing ${held_results}" >&2; exit 1; }
  "${python_bin}" -m training.bc_task_vlm.summarize_fixed_live_sim \
    --results "${train_results}" \
    --results "${held_results}" \
    --label "SFT Qwen 3 VL 8B (30 trajectories/task, epoch ${epoch_label}) · full communication" \
    --output-json "${summary}" \
    --output-md "${artifact}/summaries/sft_scale30_ep${epoch_tag}.md"
  jq --arg epoch "${epoch_label}" '. + {epoch: $epoch}' "${summary}" > "${summary}.tmp"
  mv "${summary}.tmp" "${summary}"
  jq -e '
    .num_records == 530
    and .splits.train_task_types.num_episodes == 470
    and .splits.heldout_task_types.num_episodes == 60
    and .schema_version == 4
  ' "${summary}" >/dev/null
  epoch_summary_args+=(--epoch-summary "${summary}")
  if [[ "${epoch_tag}" == 1p0 ]]; then
    summary_args+=(--summary "${summary}")
  fi
done

"${python_bin}" -m training.bc_task_vlm.build_fixed_live_sim_artifact \
  --cohort "${cohort}" \
  "${summary_args[@]}" \
  "${epoch_summary_args[@]}" \
  --output "${artifact}/report.html"

echo "Communication-ablation artifact: ${artifact}/report.html"
