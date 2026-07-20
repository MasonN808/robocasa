# Live-sim (closed-loop) evaluation of tool-calling VLMs

Operational commands, prerequisites, resume behavior, timing guidance, and
troubleshooting are documented in [`../LIVE_SIM_EVAL.md`](../LIVE_SIM_EVAL.md).

## Implementation status (2026-07-20)

Stages 0 and 1 are implemented in `divergence_analysis.py` and
`live_sim_eval.py`. The runner supports oracle, degenerate, local HF, and
Gemini native-function backends; transactional FSM/simulator execution;
self-scheduled agents; native and symbolic judges; partial goals; rejection,
efficiency, and turn-pattern metrics; resume; all-trajectory debug frames; and
bounded first-success/first-failure MP4 recordings. `training/scripts/live_sim_eval.sh`
is the EGL/H200 launcher. Focused renderer-independent regression tests live in
`tests/test_bc_task_vlm_live_sim_eval.py`.

Stage 2 and the results portion of Stage 3 remain execution work: run the
oracle gate on an EGL-capable node, run the pilot/headline matrix, inspect the
recordings, confirm determinism, and publish measured results below. The staged
exp52 subset contains no per-trajectory scene-combo field; its known generation
combo (`layout=11`, `style=34`, `seed=42`) is therefore explicit in the runner
defaults and emitted into every trajectory record.

## Context

The teacher-forced eval assumes expert trajectories are the unique ground truth: the moment a model deviates, the step is marked wrong and the model is re-anchored to the expert history. But the divergence analysis (computed from existing predictions) shows **first divergence is almost always at step 0–1, and 91–100% of first divergences are `communicate` steps** — models pick a *different but potentially valid plan* at the opening task-allocation message (SFT position-0 accuracy .44; after surviving the opening, per-position accuracy is ~.90 and flat). Teacher forcing therefore likely **understates** true task competence. The fix: let models control the agents live in sim and judge success by task outcome, not expert matching.

Answers to the three scoping questions (verified in code/data):
1. **Current eval predicts tool+args only.** The acting agent and step index are given in the prompt (`Current acting agent: {agent_id}`, prompting.py:105) with an explicit rule "The acting agent is fixed by the prompt" (prompting.py:79/93).
2. **The SFT model conditions on the agent but never generates it** — the trained response formats (`{"tool","args"}` / `<tool_call>{name,arguments}</tool_call>`) have no agent slot. Asking it to choose the agent would be out-of-distribution.
3. **The data is NOT round-robin**: 34–37% of expert transitions keep the same agent; same-agent runs go up to 9 steps. Any imposed schedule is a distribution shift, so turn policy is implemented as a pluggable flag and its effect is measured (see Stage 2).

Success can use **robocasa's native task success directly**: all 52 verified tasks map to real env classes with `_check_success` (spec `source_python_module` → e.g. `robocasa/environments/kitchen/composite/setting_the_table/beverage_organization.py:79`), and the live executor already instantiates the real task class (`run_one` passes `composite_task` to robosuite.make). The symbolic FSM checker is kept as a second, spec-level judge plus per-step legality gate.

## Architecture (all major pieces exist)

Two engines run in **lockstep** per executed step:
- **MuJoCo executor** (`robocasa/utils/sim_tool_executor.py`) — executes the call physically (teleport-based), renders images on demand (`get_image` tool :4426), judges native success (`env._check_success()`). Its `run_tool_plan` loop (sim_tool_executor_execution.py:21) is synchronous — the live loop replaces the pre-baked list with predict→execute per iteration. Env built once per (task, scene combo), rewound per trajectory via `restore_baseline_state()` (:528).
- **Symbolic FSM mirror** (`data_generation/task_level/tasks/shared/fsm.py`, `specs/runtime.py`) — each executed call is applied to a `TaskRuntimeState` via `_apply_generic_effects` (fsm.py:723) + task effects; provides **precondition/legality checks** (runtime.py:159, incl. the communicate-first gate fsm.py:174-201) and the **goal checker** `is_goal_state_satisfied` (runtime.py:321).

Model-generated args are resolved to sim ids via the adapter's existing per-symbol resolution (`TrajectoryAdapter._resolve_step_args` trajectory_adapter.py:946, alias cache seeded by `grounding_map` + `_apply_sim_ground_truth` :103; type-based fallback for unseen symbols).

Prompts are built from the **model's own partial history** — the builders are pure functions (`build_user_prompt`, `format_history_steps` prompting.py:46/60; `build_messages`), decoupled from stored trajectories. Fresh top/room/map images are rendered each turn via the executor.

Model backends reuse eval_standalone.py: HF = `VisionGenerationCollator` + `model.generate` + decode (eval_standalone.py:531-683); Gemini = `build_request`/`decoded_text_from` closures (:762-833). Parsing/validation reuses `parse_first_qwen_tool_call`, `_parse_plain_json_tool_call`, `tool_call_to_single_step_payload`.

## Stage 0 — Divergence analysis (formalize; results already computed)

Create `training/bc_task_vlm/divergence_analysis.py`: from `structured_eval_predictions.jsonl` + `comm_judge.jsonl` of any run, emit per-position judged accuracy, first-divergence step distribution, and diverging-step tool type. Note `step_index` is the global trajectory index (includes skips) — rank steps per (task, trajectory) as metrics.py does. Output `divergence_analysis.csv` + a small figure (consult dataviz skill). Headline finding to document: divergence is early (median step 0–1) and communicative (91–100%), motivating live sim.

## Stage 1 — Live-sim runner

Create `training/bc_task_vlm/live_sim_eval.py` (`python -m training.bc_task_vlm.live_sim_eval`):

- **Args**: `--backend {hf,gemini}`, model/adapter/prompt flags mirroring eval_standalone (`--sft-format`, `--no-forced-json`, `--task-spec-detail`, `--few-shot`), `--manifest` (reuse exp52 manifests' `trajectory_ids_by_task`), `--dataset-root` (for original_trajectory.json + scene combo from each traj dir's metadata), `--step-budget-factor 2.0`, `--max-consecutive-rejections 3`, `--output-dir`, `--resume`, `--save-frames`, `--max-trajectories`.
- **Env setup per task**: build `SimToolExecutor` with the same (layout, style, seed) combo the trajectory data was rendered with (read from the staged traj dir's metadata); reuse the executor-cache pattern from `scripts/sweep_trajectories.py:241` (extract `_get_or_create_cached_executor` into an importable helper or import from the script). Per trajectory: `restore_baseline_state()`, build `TrajectoryAdapter` (alias cache from `grounding_map`), init `TaskRuntimeState` from the spec as `FiniteStateTaskValidator.validate` does.
- **Loop per turn**: scheduler picks agent → render 3 views via `get_image` → build prompt from running history (`build_user_prompt`; history lines for failed attempts get a short `FAILED: <reason>` suffix — small extension to `format_history_steps`, live-sim only) → generate → parse+validate → FSM legality check (preconditions + comm gate) → if legal: resolve args, `executor.execute(...)`, mirror into `TaskRuntimeState`, append to history; if illegal/unparseable: **no-op + error feedback** in history, count rejection → check `env._check_success()` and `is_goal_state_satisfied` → stop on success, step budget (2× expert length), or 3 consecutive rejections.
- **Turn taking: the model schedules itself — no imposed policy.** Every turn uses
  the v2 agent-prediction contract (`predict_agent=True` prompt + the `agent`
  argument in every tool schema, `task_complete` available): the model decides
  which agent acts, so there is no external scheduler, no `wait` workaround, and
  no scheduling confound to control for. The formerly planned `round_robin` /
  `expert` policies are DROPPED — with the model choosing the agent they answer
  no question. This contract is not v2-only: the un-tuned base emits the agent
  argument 99.9% of the time zero-shot (teacher-forced v2 base eval), and Gemini
  gets the same `agent` parameter via its function declarations. Consequence:
  the v1 SFT adapter (trained with the agent given in the prompt) is out of
  distribution under this contract and is excluded from the live-sim roster;
  its closed-loop story is carried by v2-continue, which starts from its
  weights. Degenerate self-scheduling (one agent hogging every turn, starving
  the communicate gate) is a legitimate observable failure, visible in the
  per-trajectory records and rollout recordings, not something to engineer away.
- **Outputs**: per-trajectory jsonl (executed steps, legality per step, both success flags, partial goal fraction, steps used vs expert length, termination reason) + aggregated `live_sim_metrics.json`; optional saved frames for debugging.
- **Rollout recording** (`--record-firsts`, **default ON**): for each task, record the
  **first successful** and the **first failed** trajectory as a watchable rollout —
  per-step stills plus an MP4 assembled via the executor's existing video pipeline
  (`TrajectoryAdapter.execute` already supports `skip_videos=False`/`fps`; action
  steps push frames in `run_tool_plan`). At most 2 recordings per task, so disk and
  inode cost stay bounded (≤ 2 × n_tasks per run) while every task still gets one
  qualitative example of each outcome — enough to diagnose *why* failures happen
  (wrong object vs deadlocked turn-taking vs premature task_complete) without
  recording all 75 trajectories. `--no-record-firsts` disables; `--save-frames`
  remains the separate record-everything debug knob.

## Stage 2 — Runs

- **Oracle replay check first** (verification, not a model): feed the expert trajectory's own steps through the live loop as a scripted policy → must yield ~100% `_check_success` and FSM-goal agreement. Validates executor/FSM/success wiring before spending model compute.
- **Pilot**: ~10 trajectories (mix of both splits), best available v2 adapter. Inspect histories + rollout recordings.
- **Headline**: v2-scratch + v2-continue (native tool calls, self-scheduled) and the
  out-of-the-box controls (base 8B; Gemini-3.5 Flash via native function calling with
  the agent parameter) × both 75-trajectory splits — all under the same v2 contract.
- **Compute**: HF runs on 1×H200 (EGL rendering + 8B inference share the GPU); Gemini runs can use CPU partition with osmesa (no local model). heldout_tasks is cheap (5 tasks → 5 env builds); heldout_trajectories spans 46 tasks → env creation dominates; group trajectories by task, short `--time` for backfill.

## Stage 3 — Metrics + reporting

Primary: **live success rate** (`_check_success`) per model/split. Secondary: FSM goal rate (+ agreement between judges), partial-goal fraction, steps-to-success ratio vs expert, rejection/illegal-call rate, agent-turn distribution (same-agent run lengths vs the expert's 34-37%). The money comparison: live success vs teacher-forced `judged_traj_all` (.173/.253 SFT native; 0.000 all out-of-the-box) — does closed-loop confirm the gap or reveal teacher-forcing understated the models? Add a live-sim section to EXPERIMENT.md, results table columns, and an artifact view.

## Files

Create: `training/bc_task_vlm/live_sim_eval.py`, `training/bc_task_vlm/divergence_analysis.py`, `training/scripts/live_sim_eval.sh` (sbatch).
Modify (small): `training/bc_task_vlm/prompting.py` (optional FAILED suffix in `format_history_steps`), possibly extract executor-cache helper from `scripts/sweep_trajectories.py`.
Reuse (no changes): sim_tool_executor + execution mixin, trajectory_adapter resolution, fsm/runtime validators + `is_goal_state_satisfied`, task env `_check_success`, prompting builders, eval_standalone backends, tool_calling parsers, exp52 manifests.

## Verification

1. Oracle replay (expert steps through live loop) → ~100% success on a handful of trajectories per split; FSM and `_check_success` agree.
2. Degenerate policy (always `wait`/garbage) → 0% success, terminates by budget/rejections, no crashes.
3. Pilot SFT run: manually review 2–3 saved trajectories (frames + history) for sane grounding and honest failure classification.
4. Confirm determinism: same seed + same model temperature-0 → identical trajectory.

## Risks / notes

- MuJoCo executor enforces no physical preconditions (teleports liberally): legality lives in the FSM mirror; `_check_success` still judges final physical state.
- Self-scheduling removes the imposed-schedule distribution shift entirely; the agent-turn-distribution metric checks whether models schedule plausibly vs the expert's 34-37% same-agent pattern.
- Error-feedback history lines are unseen in training (clean-history OOD) — kept short; rejection-rate metric will show if the SFT model trips on them.
- 27B cluster SFT remains parked (NCCL example-cache); unrelated to this work.
