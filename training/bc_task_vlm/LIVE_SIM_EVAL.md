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

## Component timing benchmark

Use the fixed eight-trajectory manifest to separate model generation, image
rendering, simulator tool execution, FSM work, trajectory loading, and native
checks. Keep outputs under `eval_runs`; timing events and summaries are
evaluation outputs and must not be committed.

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m training.bc_task_vlm.live_sim_timing_benchmark -- \
  --backend hf \
  --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch \
  --manifest training/bc_task_vlm/eval_manifests/exp52/live_sim_timing_benchmark_8.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --output-dir training/bc_task_vlm/eval_runs/live_sim_timing_benchmark_8 \
  --gl-backend egl \
  --no-record-firsts

python -m training.bc_task_vlm.live_sim_timing_summary \
  training/bc_task_vlm/eval_runs/live_sim_timing_benchmark_8 \
  --output training/bc_task_vlm/eval_runs/live_sim_timing_benchmark_8/timing_summary.json
```

The July 2026 RTX 5090 baseline completed eight trajectories in 1,669.4
seconds. Across 248 policy decisions and 470 inference passes, the full HF
generation pipeline averaged 1.306 seconds per pass and high-level simulator
tool execution averaged 29 milliseconds. Rendering consumed 62.3% of episode
time: map bundles averaged 9.112 seconds because the placement map was emitted
at 6000 by 4800 pixels, while non-map bundles averaged 7.7 milliseconds. These
measurements establish a baseline; they do not change evaluation behavior.

Placement maps have two independent, opt-in throughput controls:

```text
--map-renderer legacy|raster
--map-dpi N
```

`legacy` at 300 DPI remains the compatibility default. The raster renderer
draws the occupancy classes as one image artist rather than approximately
30,000 individual Matplotlib rectangles; fixtures, robots, objects, labels,
axes, colors, and world geometry are unchanged, but the fine 5 cm cell-edge
grid is omitted. Lowering DPI alone does not materially reduce map creation
time because the per-cell Matplotlib artists dominate the legacy path.

The fixed July 2026 HF A/B used the same eight trajectories, EGL, two-pass
views, one evaluator, and no recordings. Cache invalidation was retained for
physical tools but skipped for non-mutating `communicate`, `wait`, and
`get_image` calls.

| Configuration | Total time | Map-render total | Native/FSM success |
|---|---:|---:|---:|
| Original legacy baseline | 1,669.4 s | 1,020.5 s | 2/8, 2/8 |
| Cache fix, legacy 300 DPI | 959.7 s | 305.9 s | 2/8, 2/8 |
| Cache fix, raster 60 DPI | 642.5 s | 15.4 s | 2/8, 2/8 |

The raster run reduced total wall time by 33.1% relative to the cache-fixed
legacy run and 61.5% relative to the original baseline. All three runs had the
same terminations and partial-goal results, with no harness/native errors.
Exact proposal streams are not a sufficient pixel-regression test here: a
legacy-versus-legacy repeat diverged on 3/8 trajectories despite identical map
pixels, while legacy-versus-raster diverged on 4/8; once a greedy generation
changes at one token, later closed-loop states can differ. Use task outcomes,
error rates, visual inspection, and repeated samples when validating a new map
setting.

For throughput experiments, use `--map-renderer raster --map-dpi 60`. Keep the
legacy default for compatibility comparisons, and do not mix map settings
within a reported evaluation split.

## V3 active-observation causal compatibility

For the existing v3 adapter, use the causal single-cache mode:

```text
--train-get-image --get-image-observation-mode causal_cache
```

For a newly trained matching adapter, build training and standalone-evaluation
examples with:

```text
--predict-acting-agent --train-get-image --causal-single-cache
```

The causal mode deliberately does not reproduce the legacy target-conditioned
image selection. Each proposal's visual input is determined entirely by
already-executed events:

1. The episode starts with no active visual context, so the first proposal is
   image-free.
2. A successful `get_image(agent, views)` renders immediately, updates only
   that agent's logical cache, and makes that cache the sole visual context for
   the next proposal.
3. Even when both logical caches exist, a proposal receives at most the active
   agent's cache. It never receives both agents' images. The prompt explicitly
   names that prefix-derived cache owner.
4. A proposal for a non-image tool must name the active cache owner. A physical
   action without an active agent-owned observation is rejected. Image-free
   communication is allowed when no cache is active.
5. Communication can use only the caller's own active cache. Successful
   communication does not invalidate that cache. A successful physical tool
   clears the active visual context because the scene or robot
   pose changed. The separate logical caches remain recorded but are not
   attached again until refreshed by a later request.
6. Global views remain logically agent-specific. Identical pixels may be
   render-memoized later, but requesting global views for one agent does not
   populate the other agent's cache.

This is a near-term **fake-partial-observation compatibility baseline**, not a
fully distributed policy. The model still receives the centralized joint
symbolic history and predicts the acting agent with every tool call. The
single-cache visual input keeps it closer to the current v1/v2 off-sim input
shape while removing current-target lookahead from live inference.

The current v3 checkpoint was trained with a non-causal image-selection
contract, so no live harness can be perfectly in-distribution for it.
`causal_cache` is the interpretable protocol for new diagnostic runs; it should
not be reported as a faithful partial-observability evaluation.

The legacy default `next_turn` mode still attaches a requested observation only
to the immediately following proposal. Use separate output directories for the
two modes and always report the selected mode.

## vLLM throughput mode

Run simulator clients in the documented RoboCasa environment and keep the
vLLM server in a separate environment. On the local RTX 5090, the tested server
command is:

```bash
VLLM_USE_FLASHINFER_SAMPLER=0 \
conda run -n robocasa-vllm vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.70 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image":3}' \
  --mm-processor-kwargs '{"min_pixels":262144,"max_pixels":262144}' \
  --mm-processor-cache-gb 2 \
  --enable-lora \
  --max-lora-rank 16 \
  --lora-modules \
    robocasa-v2=DorianAtSchool/qwen3vl-8b-robocasa-agentsft-scratch \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

`VLLM_USE_FLASHINFER_SAMPLER=0` is a local compatibility workaround for the
installed CUDA-header/toolkit mismatch. It disables only the optional
FlashInfer sampler path; these evaluations use greedy decoding
(`temperature=0`). Remove it when FlashInfer compiles cleanly. A server
allocation of 0.88 caused an OOM when EGL rendering shared the GPU; 0.80 passed
one- and two-client runs, and 0.70 passed four clients with headroom.

Point one live-sim client at the served LoRA name:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m training.bc_task_vlm.live_sim_eval \
  --backend vllm \
  --vllm-model robocasa-v2 \
  --manifest MANIFEST.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --output-dir OUTPUT_DIR \
  --gl-backend egl \
  --map-renderer raster \
  --map-dpi 60 \
  --resume
```

For several clients, use `live_sim_parallel_eval` rather than launching them
against one output directory manually:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
python -m training.bc_task_vlm.live_sim_parallel_eval \
  --workers 4 -- \
  --backend vllm \
  --vllm-model robocasa-v2 \
  --manifest MANIFEST.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --output-dir PARALLEL_OUTPUT_ROOT \
  --gl-backend egl \
  --map-renderer raster \
  --map-dpi 60 \
  --resume
```

The coordinator assigns a complete task to exactly one worker, preserving
environment reuse. Shard manifests stay under
`PARALLEL_OUTPUT_ROOT/manifests`; worker JSONLs, metrics, and logs stay under
`PARALLEL_OUTPUT_ROOT/workers`, and combined results are derived under
`PARALLEL_OUTPUT_ROOT/aggregate`. Never point independent evaluator processes
at the same JSONL or metrics file. On resume, use the identical command. The
coordinator rejects a changed manifest, worker count, or evaluator arguments,
and it never edits worker JSONLs while aggregating them.

The July 2026 fixed-eight benchmark used the v2 adapter, raster maps at 60 DPI,
EGL, two-pass views, and no recordings. The one-worker evaluator time is used
as its wall-clock baseline; parallel rows use the slowest worker's measured
wall clock.

| Clients | Wall time | Speedup | Mean server round trip | Generation throughput |
|---:|---:|---:|---:|---:|
| 1 | 323.65 s | 1.00x | 0.602 s | 1.46 passes/s |
| 2 | 202.81 s | 1.60x | 0.718 s | 2.32 passes/s |
| 4 | 149.36 s | 2.17x | 0.880 s | 3.15 passes/s |

All configurations produced 2/8 native successes, 2/8 FSM successes, six
`budget_exhausted` and two `goal_satisfied` terminations, full native/FSM
agreement, and no harness, native, simulator, generation, or CUDA errors.
Closed-loop proposal streams can still diverge between HF and vLLM, or between
repeated greedy runs, so do not silently combine backends within one reported
split. Four clients are the current local throughput choice; repeat the useful
configurations when the v3 `get_image` adapter is available.

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
