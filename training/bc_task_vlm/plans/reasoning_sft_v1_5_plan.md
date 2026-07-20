# Plan: SFT v1.5 — per-step reasoning supervision

## Context

A separate, **orthogonal** track from the agent/observation line (v2 → v3):
supervise a short rationale before each tool call. The trajectory data already
carries a per-step `reasoning` string (e.g. "I am picking up the first drink
from the counter."), currently **discarded** by the example builder. v1.5 tests
whether emitting that rationale first helps — especially at the step-0 plan
choice, where the divergence analysis showed models fail most.

Named v1.5 (not v3) because it's independent of agent prediction: it can layer
onto v1 (agent given) OR v2/v3 (agent chosen). Runs on its own, results compose.

## What v1.5 trains

Assistant turn becomes `<think>{reasoning}</think>` + the `<tool_call>` block.
The eval scorer already strips `<think>…</think>` before parsing
(`evaluation.py::split_reasoning`), so tool metrics are unaffected — this only
adds a supervised reasoning prefix.

## Risks (why it's a probe, not a default)

- Longer generations; parse fragility (un-fine-tuned 27B thinking parsed at .34,
  though *supervised* short rationales are a different regime).
- Reasoning-on-top *hurt* out-of-the-box Gemini (traded communication for
  action). SFT may or may not reproduce that.

## Code changes (behind `--train-reasoning`)

- `dataset.py`: stop dropping `raw_step["reasoning"]`; wrap it as the assistant
  `<think>` prefix in the tool_call target (plain path: prepend to target text).
- `prompting.py`/`build_messages`: emit the think-prefixed assistant message.
- No scorer change — `split_reasoning` already handles it.
- Cache fingerprint gains `train_reasoning`.

## Gate

Run only after v2 numbers are in hand (they are). Decision rule from the
checklist: if v2 **agent accuracy stays low after training**, planning-level
reasoning supervision is the natural lever — v2 agent accuracy is .99/.84, so
this is **lower priority**; keep as a probe unless held-out-task competence
(.67 action) is the thing we want to push, where a rationale might help most.

## Eval

Teacher-forced on both splits; compare action-exact + judged against the
matching non-reasoning run (v1 or v2). A win = same-or-better tool metrics with
the reasoning prefix; a regression = reasoning is off-distribution for SFT here.
