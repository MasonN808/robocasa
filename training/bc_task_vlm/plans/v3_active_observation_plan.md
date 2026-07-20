# Plan: SFT v3 — active observation (train `get_image`)

## Context

v1 predicts tool+args; v2 adds the acting-agent choice + `task_complete` (result:
agent head is free — .99/.84 agent accuracy, action-exact on par with v1). v3
adds the last piece of a self-sufficient closed-loop policy: **the model requests
its own observations** by emitting `get_image`, instead of the harness rendering
for it.

Motivation is measured (see `agent_prediction_sft_plan.md` §Future work): the
expert's view choice is a **deterministic function of the NEXT action** (scout
set before `navigate_to_fixture`, wrist set before manipulation, overhead at the
opening) but only ~50–88% predictable from history — so no harness-side rule can
fully mirror the training distribution in live-sim. Training `get_image` closes
that gap: model requests views → harness renders → model acts, exactly as the
expert data interleaves.

**Scale decision:** v3 is a **v1/v2-scale run (30 traj/task)**, not scaled data.
Rationale to revisit later, not now: scratch vs continue tied per-step at 30
traj/task, hinting we may be data-limited — so a 100+ traj/task control is worth
running *eventually*, but v3 first establishes the capability at the established
scale. Skip to a v3-scale (100/task) run only if v3 numbers look good.

**Start fresh, do not continue from a v2 adapter.** Evidence: v2-continue (warm
from v1) was a per-step tie with v2-scratch and *worse* on novel-task trajectory
completion (.107 vs .227). Warm-starting from a checkpoint trained on a different
contract is at best neutral, at worst a generalization tax — and v3 changes the
contract again. → fresh LoRA from `Qwen/Qwen3-VL-8B-Instruct`, v2 flags on.

## What v3 newly trains

1. **`get_image` as a first-class emitted step.** Re-add it to the tool schemas
   (currently stripped) and keep its steps as supervised targets. The agent
   binding is already solved by v2: every tool schema carries a required `agent`
   argument, so a request is
   `{"name":"get_image","arguments":{"agent":"agent_0","views":["wrist","agentview_center"]}}`
   and the `agent` arg routes the wrist/agentview cameras (global views
   top/room/map ignore it). No new agent mechanism.
2. **`get_image` steps re-enter the history text.** Today `format_history_steps`
   never sees them (they're dropped in the builder). v3 keeps them in history so
   the model conditions on what it just observed → requested-views become
   in-distribution for the following action.

Deferred / noted, not in v3:
- **Data scale (100 traj/task)** — a control run, revisit after v3 lands.
- **Per-step reasoning supervision** — reclassified as its own **v1.5** track
  (independent of the agent/observation line); see `reasoning_sft_v1_5_plan.md`.

## Code changes (behind `--train-get-image`, off = v2 behavior)

- `dataset.py` `_build_centralized_examples_for_trajectory`: stop the
  `if raw_step["tool"] == "get_image": continue` skip when the flag is on —
  build a target example for each get_image step (target tool = `get_image`,
  args = its `views`, agent = step agent) AND append it to `history_steps`.
  Normalize the agent on *global-only* view requests (top/room/map) to a fixed
  placeholder so the model doesn't learn the arbitrary requester as signal.
  Cache fingerprint gains `train_get_image` (only when set → v2 caches stay
  valid).
- `schema_utils.py` / `tool_calling.py`: add a `get_image` tool spec
  (`views` = STRING_ARRAY, enum of the known view names; `agent` via the v2
  include-agent path) to the augmented tool set; the OBSERVATION_TOOL_NAMES
  constant already exists (`data_generation/.../shared/constants.py:17`).
- `prompting.py`: drop the "do not output get_image" line from the system
  prompt when `--train-get-image`; `format_history_steps` already renders
  arbitrary tools, so get_image history lines need no special-casing.
- `evaluation.py`/`metrics.py`: new metrics — `observation_recall` (did the
  model request an observation where the expert did) and view-set exact-match
  on those steps; keep action metrics observation-excluded so they stay
  comparable to v1/v2.
- `eval_standalone.py` + `live_sim_eval.py`: `--train-get-image` pass-through.
  In live-sim, when the model emits `get_image`, the harness renders the
  requested views for that step's agent (executor `get_image` already takes
  `agent_id`) instead of auto-rendering — the observation becomes model-driven.
- `main.py`: `--train-get-image` threads through example build + fingerprint.

## Training + eval

- **v3 run:** fresh LoRA from Instruct, same 30-traj/task subset + v2 flags +
  `--train-get-image`, effective batch 8, 3 epochs (compute-matched to v2).
  Cluster wrapper mirrors `train_v2_continue_8b_cluster.sh`.
- **Eval A (teacher-forced):** same manifests, `--train-get-image`; report the
  new observation metrics plus the v2 metrics (action-exact must stay ≈ .98/.70
  — adding observation shouldn't cost tool competence).
- **Eval B (live-sim):** the payoff — run with model-driven observation and
  compare `_check_success` against the harness-rendered v2 models. This is where
  closing the view-distribution gap should show up.

## Verification

1. Regression: `--train-get-image` off ⇒ examples byte-identical to v2.
2. Dataset spot-check: get_image steps are now targets + in history; global-view
   requests have the normalized placeholder agent; wrist requests keep the real
   agent.
3. Schema/parse roundtrip for a get_image emission.
4. Train smoke (MAX_STEPS=20) + eval smoke (--max-samples 8).
