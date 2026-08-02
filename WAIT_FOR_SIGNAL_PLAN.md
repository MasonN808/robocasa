# Implementation plan: `wait_for_signal`, self-contained specs, step-index A/B

Ordering matters: the tool must exist before specs are materialised, and specs
must be materialised before trajectories are rewritten, or each artefact gets
migrated twice.

---

## Phase 0 — independent quick fixes (no spec or data changes)

**0.1 Order-preserving expert replay.**
`live_sim_eval.py:~1522` splits the expert trajectory into per-agent queues and
lets the scheduler interleave them. That reordering is what makes expert replay
fail: in `arrange_bread_bowl/traj_000009`, `agent_1` picked up the bowl and
carried it to `dining_counter` before `agent_0` could place bread into it, so a
correct plan was rejected for "missing navigation".

Non-model policies must execute `trajectory["steps"]` in recorded order.

*Acceptance:* oracle-gate FSM success rises well above today's 69.8%. Whatever
still fails after this is a real defect rather than a scheduling artefact.

**0.2 Re-baseline the oracle gate.** Re-run all 52 tasks x 10 after 0.1. The
current Level-0 list is not trustworthy until then.

---

## Phase 1 — `wait_for_signal`

### Semantics (deliberately dumb harness)

    wait_for_signal(from: agent_id, about: str)

- The harness **never inspects `about`**. Any message delivered to a waiting
  agent wakes it. `about` is *declared intent*, recorded for scoring only.
- Deciding whether the delivered message was actually the awaited one is the
  **model's** job: if it was not, the model calls `wait_for_signal` again.
- This is the point of the design. A harness that guaranteed relevance would
  leave nothing to learn and nothing to measure; making it the model's decision
  creates an observable comprehension signal we otherwise cannot get from logs.

### Harness changes

- `AgentRuntime.waiting_for: dict | None`; the scheduler skips waiting agents.
- Any delivered message clears `waiting_for` and re-enters the agent.
- **Livelock guard:** cap consecutive waits per agent; the first wait is free,
  subsequent ones consume step budget. Bounded and visible, never a hang.
- **Deadlock breaker:** all agents waiting -> timeout, force-resume, and log a
  coordination failure. Without this, budget exhaustion becomes a hang, which is
  strictly worse operationally.

### Scoring signals it unlocks (post-hoc, no judge in the loop)

- resumed on a message that *was* about `about` -> correct comprehension
- resumed on an unrelated message -> false wake
- kept waiting through a relevant message -> missed release
- waited for a signal the partner never sent -> partner-side failure

---

## Phase 2 — materialise task specs (once, after Phase 1)

Today the tool set is computed in one place and read from another:

    spec file declares      : communicate, give_space, navigate_to_fixture,
                              pick_up_object, place_next_to
    initial_state implies   : hinged parts on {fridge, cabinet}
    GENERATOR validator sees: ... + open_hinged_part   (derived allowlists)
    EVALUATOR registry sees : ...                      (raw declared spec)

`task_registry.py` does `deepcopy(task_spec.allowed_tool_specs)` and never calls
`build_task_definition_from_spec`, so the evaluator misses every runtime
injection. This single defect produces three symptoms already observed:
"Tool open_hinged_part is not allowed here", `KeyError: 'hinged'`, and the
over-narrow `give_space` allowlist.

**Do:** resolve once and write the full tool set back into each spec file —
including `wait_for_signal`, injected `open_hinged_part` with derived
allowlists, auto-appended shared tools, and canonical fixture names. Reduce
`build_task_definition_from_spec` to a pure loader. Point `task_registry` at the
same field.

*Acceptance:* CI asserts `resolved == derive(spec)` for every task, so the two
paths cannot drift again.

---

## Phase 3 — rewrite trajectories (once, after Phase 2)

Offline pass, no simulator time:

1. **Insert waits** at cross-agent dependencies — wherever agent B acts on an
   object agent A last touched (23 of 47 tasks have these).
2. **Insert the announcement**: the waiter states what it is waiting for, so the
   dependency is explicit in the dialogue and the sender has an obligation.
3. **Insert the release** from the partner at the point the dependency clears.
4. **Keep distractor messages.** Training must contain informative-but-unrelated
   messages (progress updates) between a wait and its release. Without them
   every message is the awaited one and the model learns "resume on any
   message" — the degenerate policy.

Inserting waits may make the existing *sequential* demonstrations valid under
the *concurrent* scheduler with no regeneration: the waits encode the ordering
the plans always implicitly assumed.

### Hard gate

**The oracle gate must return 100% FSM success on order-preserving replay.**
Any expert demo that still fails is a wait-insertion bug, not a hard task. This
is the "expert demos have no mistakes" requirement, made mechanical.

Additional tests:
- unit: wait blocks; any message wakes; livelock cap fires; mutual wait breaks
- property: every inserted wait has a matching release from the named partner
- negative: a trajectory with a wait but no release must fail the gate

---

## Phase 4 — retrain, with the step-index A/B

Two runs, identical except `--partial-step-index-mode`:

| arm | setting |
|---|---|
| A | `local` (current default) |
| B | `none` |

Hypothesis: the step index is a hidden progress signal that lets the model avoid
learning trajectory understanding. If so, arm B should show better
history-grounded behaviour even at similar raw success.

Report beyond `fsm_goal_rate`, since that alone will not separate them:
precondition-violation rate, say-do fidelity, perseveration, and the
speak-when-stuck contingency ratio.

---

## Explicitly deferred: DAgger

Recovery from **state-tracking** errors cannot come from expert demonstrations.
The planner has ground-truth symbolic state and never mis-tracks what it holds,
so it will never demonstrate recovering from that class of error — which is the
single largest failure category (43.8% of trained partial-obs rejections).

That needs student rollouts with expert correction, and it is a separate
project. It should not be smuggled into regeneration.

---

## Already landed (context)

- `--rejection-mode report-failed` is now the default (adaptation after a
  rejection: 8.0% -> 100% trained, 9.4% -> 94.1% untrained; 28-32% of the step
  budget was going to re-emitting known-failed calls)
- prompt renders `tool_arg_any_of`
- `give_space` allowlist widened to reachable fixtures
- eval-only `cab` <-> `cabinet` alias
- communicate-first gate disclosed in the prompt

Note Phase 4 partly repays a debt: `report-failed` shows models a `FAILED:`
history entry they never saw in training. `format_history_steps` already renders
it and the docstring notes training data never sets it, so if regenerated
trajectories ever carry `error` fields, the plumbing is already there.
