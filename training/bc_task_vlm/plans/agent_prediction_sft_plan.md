# Plan: SFT v2 — the model also chooses the acting agent

## Context

Today's models predict only `tool + args`; the acting agent is given in the prompt
(`Current acting agent: {agent_id}`, prompting.py:105) and the trained response
formats have no agent slot. That makes the model unusable as a *scheduler* in
closed-loop settings (see `live_sim_eval_plan.md`: turn order had to be imposed
externally, which is a distribution shift — expert data is 34–37% same-agent
transitions with runs up to 9, i.e. genuinely not round-robin). SFT v2 trains the
model to emit the acting agent itself, evaluated both teacher-forced and live-sim.

**Ordering note — the "instruct model" side-question is already resolved.** The
existing SFT adapter was trained on `Qwen/Qwen3-VL-8B-Instruct`
(`adapter_config.json` on `DorianAtSchool/qwen3vl-8b-robocasa-sft`:
`base_model_name_or_path = Qwen/Qwen3-VL-8B-Instruct`), and every "Qwen3-VL-8B
base" row in our results is the *un-fine-tuned Instruct* model. A non-instruct
base does not exist on the hub (`Qwen/Qwen3-VL-8B` and `-Base`: 404); the only
other startable checkpoint is `Qwen/Qwen3-VL-8B-Thinking`. So there is no
instruct-vs-base training to run first — v2 can start immediately. An optional
Thinking-variant probe is listed at the end (low priority; our 27B evals showed
thinking-mode outputs parse poorly out of the box, though SFT could fix that).

## What v2 newly trains the model to output

1. **Acting agent** (the core ask): every step's supervision includes which agent
   acts. Combined with tool+args this makes the model a full single-step planner.
2. **Task-completion signal**: a synthetic terminal step (`task_complete`, no
   args) appended after each trajectory's last action. Without it, a live-sim
   episode can only end by oracle goal-checking or budget; with it, "does the
   model know it's done" becomes measurable, and live-sim termination stops
   depending on the judge. (~1 extra supervised step per trajectory, ~7% more
   examples.)
**Deferred to a later, separate run — failed-step robustness** (augmentation,
not a new output): injecting synthetic `FAILED: <reason>` attempts into training
histories would make the live-sim error-feedback loop in-distribution. Deferred
deliberately so v2 changes exactly one thing (what the model outputs) and the
agent-prediction effect isn't confounded with a data-distribution change. Note
on *why* steps fail despite teleport-based tools: failures are symbolic, not
physical — hallucinated/unknown object ids (resolution `ValueError`), FSM
precondition violations (placing an unheld object, acting before the
communicate gate, disallowed tool), parse failures, and only rarely a sim
placement refusal (`pick_up_object` → `success=False` when the robot can't be
posed near the fixture).

Considered and rejected for now: predicting the step index (always provided),
requesting observations (`get_image` — deferred to v3, see the final section:
measured evidence shows the expert's view choice is a deterministic function of
the NEXT action, which only the model knows, so harness-side mirroring is
fundamentally lossy), and emitting both agents' actions jointly (breaks the
single-step API and the eval).

## Response format (native tool calling retained)

Native `<tool_call>` format won every interface cell, so v2 stays on it. The
agent rides inside the arguments: **every tool schema gains a required `agent`
parameter** (enum `agent_0|agent_1`), rendered naturally by Qwen's chat template
and emitted as e.g.

```
<tool_call>
{"name": "pick_up_object", "arguments": {"agent": "agent_0", "object_id": "drink_0", "source_id": "counter"}}
</tool_call>
```

The parser strips `agent` out of the arguments into the payload's `agent` field.
This keeps one `<tool_call>` block, no new syntax, and works identically for
`task_complete`. The prompt drops the `Current acting agent:` line and the
"acting agent is fixed" rule becomes "first decide which agent acts next."

## Code changes (all behind a `--predict-acting-agent` flag; v1 behavior when off)

- `training/bc_task_vlm/prompting.py` — `build_user_prompt(predict_agent=...)`:
  omit the agent line, swap the rule text; `format_history_steps` learns the
  optional `FAILED:` suffix (shared with live-sim).
- `training/bc_task_vlm/dataset.py` — example builder injects `agent` into the
  target arguments; synthesizes the terminal `task_complete` example per
  trajectory. (`--history-failure-rate` augmentation: deferred, see above.)
- `training/bc_task_vlm/tool_calling.py` — `build_tool_schemas(include_agent=...)`
  adds the `agent` parameter (+ a `task_complete` schema);
  `tool_call_to_single_step_payload` pops `agent` from arguments and uses it as
  the payload agent (today it comes from the caller).
- `training/bc_task_vlm/schema_utils.py` — validator accepts the agent arg /
  `task_complete` tool.
- `training/bc_task_vlm/evaluation.py` + `metrics.py` — new metrics:
  `agent_accuracy`, strict step accuracy (agent+tool+args), and the existing
  tool/args metrics both strict and agent-agnostic (an alternative-but-valid
  agent choice is exactly the ambiguity the judged metrics exist for — report
  both).
- `training/bc_task_vlm/eval_standalone.py` — pass-through flag; forced-JSON
  schema variant gains the agent field for control runs.
- `training/bc_task_vlm/main.py` — `--init-adapter-path`: initialize LoRA from a
  published adapter and continue training (the existing
  `--resume-from-checkpoint` only resumes a local Trainer checkpoint; small,
  needed for v2-continue below).

## Training runs (local RTX 5090 — scripts already exist)

Entry points: `training/scripts/local/` — `stage_subset.sh` (HF-direct staging),
`probe_throughput.sh`, `train_qwen3vl_8b.sh`, shared config `_launch_args.sh`
(env-var overrides: `MODEL_PATH`, `OUTPUT_DIR`, `NUM_EPOCHS`, batch/resolution…);
see its `README.md`. Both runs use the same staged subset + new data flags:

1. **v2-scratch** — fresh LoRA from `Qwen/Qwen3-VL-8B-Instruct` with
   `--predict-acting-agent` (no failure augmentation — deferred).
   `OUTPUT_DIR=runs/qwen3vl-8b-agentsft-scratch`.
2. **v2-continue** — same data, LoRA initialized from
   `DorianAtSchool/qwen3vl-8b-robocasa-sft` via the new `--init-adapter-path`.
   `OUTPUT_DIR=runs/qwen3vl-8b-agentsft-continue`.
   Hypothesis: tool competence transfers, only the agent head is new; compare
   compute-matched against v2-scratch.

Push both adapters to HF (as `qwen3vl-8b-robocasa-agentsft-{scratch,continue}`)
so cluster evals can pull them.

## Eval A — teacher-forced, agent predicted (same GT trajectories)

Same exp52 manifests and pipeline; prompt built with `predict_agent=True`.
Score: `agent_accuracy` vs the expert's agent; strict exact (agent+tool+args);
agent-agnostic tool/args metrics; judged comm as today. Compare four models:
v1 SFT (agent given) as the ceiling reference, v2-scratch, v2-continue, and the
un-tuned Instruct base with the v2 prompt (out-of-the-box control). Caveat to
document: like tools, the "correct" agent is sometimes ambiguous (either agent
could validly act) — strict agent accuracy is a lower bound, which is another
reason Eval B exists.

## Eval B — live-sim with `model_choice`

Run `live_sim_eval_plan.md` Stage 2 with the new `model_choice` turn policy
(now in-distribution): the model's emitted agent drives turn order; a
`task_complete` emission ends the episode (success still judged by
`_check_success` + FSM goals). Compare against v1-SFT under `round_robin` /
`expert` policies to answer: does letting the model schedule itself beat an
imposed schedule?

## Optional side probe — Thinking variant

If a base-model comparison is still wanted after the Instruct clarification:
LoRA on `Qwen/Qwen3-VL-8B-Thinking` (v1 data, no agent prediction, one-epoch
probe on the 5090 via `MODEL_PATH=Qwen/Qwen3-VL-8B-Thinking`), eval nativefmt on
both splits. Run only if v2 queues leave the GPU idle — expected payoff is low.

## Verification

1. Unit: schema→render→parse roundtrip with the agent arg and `task_complete`;
   payload agent comes from the model output, not the caller.
2. Dataset spot-check: built example's prompt has no agent line; target
   arguments contain `agent`; one `task_complete` example per trajectory;
   (FAILED-line rendering is verified when the deferred augmentation lands.)
3. Regression guard: with `--predict-acting-agent` off, built examples are
   byte-identical to v1 (protects all existing results).
4. Train smoke (`MAX_STEPS=20`) + eval smoke (`--max-samples 8`) before the full
   local runs, per the local README flow.

## Execution order

1. Code changes + verification 1–3 (cluster, no GPU needed).
2. v2-scratch and v2-continue on the 5090 (sequential; probe first).
3. Eval A on cluster (or locally), including the v1 reference rows.
4. Live-sim Eval B once `live_sim_eval_plan.md` Stage 1 lands.

## Future work — split into separate tracks

- **v3 (active observation, train `get_image`)** — full plan in
  `v3_active_observation_plan.md`. v1/v2-scale (30 traj/task), fresh LoRA
  from Instruct (NOT continue-from-v2: scratch beat continue on novel-task
  trajectory completion, .227 vs .107).
- **v1.5 (per-step reasoning)** — orthogonal track, `reasoning_sft_v1_5_plan.md`.
- **Data scale (100 traj/task)** — noted, deferred; revisit after v3.

### Original v3 measurement note: train `get_image` — active observation

Currently `get_image` steps are never supervised and never appear in history;
the harness delivers observations. Measured across 162 trajectories
(~3,300 get_image calls), the expert's view choice follows a **deterministic
forward rule — views match the UPCOMING action**:

| next action | views requested |
|---|---|
| episode opening (steps 0-1, both agents) | `top_view, room_view, map` (100%) |
| `navigate_to_fixture` | `agentview_center/left/right` scout set (100%) |
| any manipulation / `give_space` | `wrist, agentview_center` (100%) |
| `communicate` | whatever is current (opening overhead vs mid-work wrist, ~50/50) |

But the rule is only ~50-88% predictable from *history* (after a navigate or
communicate it is a coin flip), because it anticipates a decision the model has
not yet emitted. Consequence: **no harness-side rendering rule can fully mirror
the training distribution in live-sim** — but the model itself can, if v3
trains `get_image` as a first-class action (supervision is essentially free:
the view set is determined by the model's own intended next action). That
makes live-sim trajectories in-distribution end-to-end: model requests views →
harness renders them → model acts on them, exactly as the expert data
interleaves. Until v3, live-sim uses the best history-conditioned
approximation (opening → overhead set; after a manipulation → wrist set,
99-100% correct; after navigate/communicate → accept the ~50% mismatch or use
a bounded two-pass re-render keyed on the predicted action).
