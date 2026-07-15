#!/usr/bin/env bash
# Stage 2.5 pilot ablations on the pilot manifest (Gemini backend; the HF
# forced-json ablation runs separately on GPU via eval_standalone_hf.sh).
# Grid: forced JSON vs free-form, thinking budget 0 vs model default,
# temperature 0 vs 0.7, few-shot 0 vs 1.
set -euo pipefail

repo_root="/work/hdd/bgjs/dbenhamougoldfajn/robocasa"
cd "${repo_root}"
python_bin="${PYTHON_BIN:-${repo_root}/.venv/bin/python}"
manifest="${PILOT_MANIFEST:-training/bc_task_vlm/eval_manifests/exp52/eval_manifest_pilot.json}"
out_base="${PILOT_OUT:-training/bc_task_vlm/eval_runs/pilot}"
model="${GEMINI_MODEL:-gemini-3-flash-preview}"
# Manifests record the node-local /tmp root they were built from; evals read
# images from the exported subset instead.
dataset_root="${EVAL_DATASET_ROOT:-training/bc_task_vlm/eval_data_subset}"

run_one() {
  local tag="$1"; shift
  local out_dir="${out_base}/gemini_${tag}"
  if [[ -f "${out_dir}/structured_eval_metrics.json" ]]; then
    echo "=== ${tag}: already complete, skipping"
    return
  fi
  echo "=== pilot ablation: ${tag}"
  "${python_bin}" -m training.bc_task_vlm.eval_standalone \
    --backend gemini \
    --model "${model}" \
    --manifest "${manifest}" \
    --dataset-root "${dataset_root}" \
    --output-dir "${out_dir}" \
    --concurrency "${CONCURRENCY:-8}" \
    --max-cost-usd "${MAX_COST_USD:-5}" \
    --resume \
    "$@"
}

run_one "forced_t0_think0"        --forced-json    --temperature 0.0 --thinking-budget 0
run_one "freeform_t0_think0"      --no-forced-json --temperature 0.0 --thinking-budget 0
run_one "forced_t0_thinkdefault"  --forced-json    --temperature 0.0 --thinking-budget -1 --max-output-tokens 4096
run_one "forced_t07_think0"       --forced-json    --temperature 0.7 --thinking-budget 0
run_one "forced_t0_think0_fewshot" --forced-json   --temperature 0.0 --thinking-budget 0 --few-shot 1

echo ""
echo "=== pilot summary ==="
"${python_bin}" - <<'EOF'
import json
from pathlib import Path

base = Path("training/bc_task_vlm/eval_runs/pilot")
rows = []
for metrics_path in sorted(base.glob("gemini_*/structured_eval_metrics.json")):
    metrics = json.loads(metrics_path.read_text())
    rows.append(
        (
            metrics_path.parent.name,
            metrics.get("structured_eval_num_samples", 0),
            metrics.get("structured_eval_tool_call_parse_rate", 0),
            metrics.get("structured_eval_tool_name_accuracy", 0),
            metrics.get("structured_eval_exact_tool_call_accuracy", 0),
        )
    )
print(f"{'config':38s} {'n':>5s} {'parse':>6s} {'tool':>6s} {'call':>6s}")
for name, n, parse, tool, call in rows:
    print(f"{name:38s} {n:5.0f} {parse:6.3f} {tool:6.3f} {call:6.3f}")
csv_path = base / "pilot_ablation.csv"
with csv_path.open("w") as handle:
    handle.write("config,n,parse_rate,tool_name_accuracy,exact_tool_call_accuracy\n")
    for row in rows:
        handle.write(",".join(str(v) for v in row) + "\n")
print(f"wrote {csv_path}")
EOF
