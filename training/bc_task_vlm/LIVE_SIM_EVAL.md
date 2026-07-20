# Live-sim evaluation runbook

This runbook covers operational use of the closed-loop evaluator implemented
in `training.bc_task_vlm.live_sim_eval`. Read
[`plans/live_sim_eval_plan.md`](plans/live_sim_eval_plan.md) for its motivation,
architecture, and experiment design.

## Prerequisites

- A CUDA-capable environment with PyTorch, Transformers, PEFT, MuJoCo,
  RoboSuite/RoboCasa, PyOpenGL, Pillow, and ImageIO.
- EGL-capable NVIDIA drivers. MP4 recording additionally requires an ImageIO
  FFmpeg backend.
- `training/bc_task_vlm/eval_data_subset` and the exp52 manifests.
- Hugging Face access to `Qwen/Qwen3-VL-8B-Instruct` and the adapter being
  evaluated. The v2-scratch adapter is
  `DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch`.
- Enough GPU memory for the 8B model plus image inputs. An RTX 5090 (32 GB) or
  one H200 is suitable.

Do not commit the model cache, evaluation subset, generated frames, MP4s,
evaluation run directories, or SLURM logs.

## Rendering environment

Set EGL before importing MuJoCo or RoboSuite:

```bash
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
```

Warnings that no private RoboSuite macro file exists are harmless for this
evaluation. OSMesa errors involving `OpenGL.GL` indicate that the host lacks a
working software-rendering setup; use EGL on a GPU host.

## Recommended execution sequence

Always run these gates in order:

1. One scripted oracle trajectory.
2. Two model-controlled pilot trajectories.
3. The full `heldout_tasks` split (75 trajectories, 5 environment builds).
4. The full `heldout_trajectories` split (75 trajectories, 46 environment
   builds).

The oracle must report native success, FSM success, and judge agreement of 1.0
with no harness errors. The model pilot must complete without GPU OOM,
`harness_error`, native-check errors, or systematic parsing failures before a
full run is started.

## 1. Local oracle gate

```bash
python -m training.bc_task_vlm.live_sim_eval \
  --backend oracle \
  --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_tasks.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --output-dir /tmp/live_sim_oracle_local \
  --tasks beverage_organization \
  --max-trajectories 1 \
  --gl-backend egl \
  --no-record-firsts
```

The validated local reference run completed in about 10 seconds total:
approximately 5.4 seconds to build the environment and 0.8 seconds to replay
19 expert actions. Timing varies by machine and cache state.

## 2. Local v2-scratch pilot

Keep two-pass views enabled so the measurement represents the intended full
configuration:

```bash
python -m training.bc_task_vlm.live_sim_eval \
  --backend hf \
  --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch \
  --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_tasks.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --output-dir training/bc_task_vlm/eval_runs/live_sim_v2scratch_local_pilot \
  --tasks beverage_organization \
  --max-trajectories 2 \
  --gl-backend egl \
  --no-record-firsts
```

For a quick harness-only diagnostic, add `--no-two-pass-views`. Do not use that
faster configuration to estimate or report the headline run.

## 3. Full local evaluation

Run the cheaper held-out-task split first:

```bash
python -m training.bc_task_vlm.live_sim_eval \
  --backend hf \
  --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch \
  --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_tasks.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --output-dir training/bc_task_vlm/eval_runs/live_sim_v2scratch__heldout_tasks \
  --gl-backend egl \
  --resume
```

Then run held-out trajectories:

```bash
python -m training.bc_task_vlm.live_sim_eval \
  --backend hf \
  --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch \
  --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_trajectories.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --output-dir training/bc_task_vlm/eval_runs/live_sim_v2scratch__heldout_trajectories \
  --gl-backend egl \
  --resume
```

`--resume` reads `live_sim_trajectories.jsonl` and skips completed
`(task_name, trajectory_id)` pairs. Re-run the same command and output directory
after interruption. Do not combine results from different models or prompt
configurations in one output directory.

## Cluster execution

The SLURM launcher defaults to EGL and accepts configuration through exported
environment variables:

```bash
sbatch \
  --export=ALL,BACKEND=hf,SPLIT=heldout_tasks,\
RUN_NAME=v2scratch__heldout_tasks,\
MODEL=Qwen/Qwen3-VL-8B-Instruct,\
ADAPTER=DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch \
  training/scripts/live_sim_eval.sh
```

For smoke tests, request a short walltime so the job can backfill:

```bash
sbatch --time=01:00:00 \
  --export=ALL,BACKEND=hf,SPLIT=heldout_tasks,\
RUN_NAME=v2scratch_pilot,MODEL=Qwen/Qwen3-VL-8B-Instruct,\
ADAPTER=DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch,\
EXTRA_ARGS='--tasks beverage_organization --max-trajectories 2 --no-record-firsts' \
  training/scripts/live_sim_eval.sh
```

Use an `afterok` dependency when a model pilot must wait for an oracle gate:

```bash
sbatch --dependency=afterok:<ORACLE_JOB_ID> ...
```

## Outputs

Each output directory contains:

- `live_sim_trajectories.jsonl`: proposals, legality, execution outcome,
  success flags, termination, timing, agent turns, and scene configuration for
  each trajectory.
- `live_sim_metrics.json`: aggregate native/FSM success, judge agreement,
  partial goals, rejections, efficiency, turn patterns, and timing.
- `recordings/<task>/{success,failure}/`: the first successful and first failed
  rollout per task, with stills and per-camera MP4s (enabled by default).
- `frames/`: all-trajectory frames only when `--save-frames` is requested.

Use `--no-record-firsts` for timing pilots so video encoding does not affect
the estimate. Full qualitative runs should leave recording enabled.

## Runtime estimation

The relevant fields in `live_sim_metrics.json` are:

- `policy_load_s`
- `total_session_build_s`
- `mean_model_proposal_elapsed_s`
- `mean_trajectory_elapsed_s`
- `total_elapsed_s`

The two exp52 splits contain 150 trajectories, 51 distinct task environment
builds, 2,250 expert actions, and a maximum 2x budget of 4,500 model turns.
A conservative estimate from a representative pilot is:

```text
model load
+ (mean environment-build time × 51)
+ (mean proposal time × expected turns)
+ simulator and recording overhead
```

Use 4,500 turns as the worst case. Actual runs are shorter when models declare
completion, succeed before budget, or terminate after three consecutive
rejections. Early estimates for one H200 were roughly 4–6.5 hours for both
splits; an RTX 5090 should be budgeted approximately 8–12 hours until its pilot
provides measured proposal latency.

## Troubleshooting

- **CUDA OOM:** close other GPU processes; ensure only one evaluator is using
  the device. Reducing render resolution changes the evaluated visual contract
  and should only be used for diagnosis.
- **Repeated parsing errors:** confirm the adapter is v2 agent-prediction SFT
  and uses `tool_call` format. Inspect each step's `reason` in the JSONL.
- **Native checker references a missing `obj_N`:** update to the runner version
  that synchronizes variable task cardinality after loading each trajectory.
- **Native success is `null`:** inspect `native_error`; do not treat the run as
  a valid result even if the FSM succeeds.
- **Three immediate rejections:** inspect the first proposals for a missing or
  invalid `agent`, failure to communicate, invalid symbolic IDs, or simulator
  execution errors.
- **Interrupted process:** rerun the identical command with `--resume`.
- **Slow shared-filesystem startup:** use a local model/data cache or node-local
  staging; distinguish `policy_load_s` and `total_session_build_s` from actual
  proposal latency.

## Verification tests

Renderer-independent checks:

```bash
python -m pytest -q tests/test_bc_task_vlm_live_sim_eval.py
python -m py_compile \
  training/bc_task_vlm/live_sim_eval.py \
  training/bc_task_vlm/divergence_analysis.py
bash -n training/scripts/live_sim_eval.sh
```

These tests do not replace the EGL oracle gate.
