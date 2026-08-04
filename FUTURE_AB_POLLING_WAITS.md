# Future A/B: polling waits (re-emitted `wait_for_signal`)

**Not implemented. Do not start this until the current approach works end to
end** — it changes both the data and the eval harness, and it is only worth
testing against a working baseline.

## The idea

Today a blocked agent produces **nothing**. It is simply absent from the tick
rows, and at eval time `AgentRuntime` does not invoke it at all until a
`communicate` arrives whose `releases` names what it is waiting for. Waiting is
implemented by the **harness**, not by the policy.

Instead: fill those empty cells so the blocked agent **re-emits
`wait_for_signal(from=..., about=...)` on every tick it is blocked**, until the
release lands, and then continues as before.

Use the existing tool, not a new `"waiting…"` sentinel string: the tool
vocabulary stays closed, no schema change is needed, the validator has
something typed to check, and "I am still waiting on X" is exactly what the
call already means.

## Why it is worth testing

**1. The harness gets dumb.** This does NOT remove the harness — something must
still invoke the agent each tick to get the next `wait_for_signal`. What
changes is what the harness must *understand*: today it parses `releases` and
decides semantically when to wake an agent; under polling it invokes everyone
every tick and the **policy** decides whether to keep waiting. That is a
stronger claim — right now the coordination arguably belongs to the harness
rather than to the model.

**2. It dissolves the lost-wakeup limitation.** This is the bigger point and it
was not the original motivation. A polling wait is **level-triggered**, not
edge-triggered: the agent re-evaluates every tick instead of depending on a
one-shot signal landing at the right instant. That is exactly the
`while (!predicate) wait()` pattern — option (3) in
`CONCURRENCY_LIMITATIONS.md` — so **lost wakeups become structurally
impossible**, and with them the variable-duration failure that forces us to
validate under `LOCK_STEP` only. Two separate problems, one fix.

## Feasibility: high, and no regeneration needed

Derive it as a **post-process injection, exactly like `get_image`** — the
concurrent FSM already computes which instants each agent is blocked
(`replay()` tracks `blocked_on` and `waiting_since`), so filling the empty
cells is deterministic over data we already have. Apply the same discipline as
the image injector: preserve the tick schedule, then re-validate.

## Risks to measure

1. **Dataset imbalance.** Every blocked tick becomes a training example whose
   answer is trivially "keep waiting". Long waits could make these a large
   fraction of the corpus and dominate the loss. Wants downsampling or loss
   weighting, and the A/B should report the waiting-example fraction.
2. **Livelock — a genuinely new failure mode.** An over-trained "keep waiting"
   habit means the agent may never resume. This is *impossible* today because
   the harness controls resumption. The earlier tick A/B already showed a bias
   toward getting stuck ("asks and never releases"), so treat this as likely,
   not hypothetical.
3. **Observation cost.** Decide whether a polling agent receives images each
   tick. Cheapest and most sensible is no observation while blocked — it
   conditions on its own history plus delivered messages, which is what it
   needs to notice the release.
4. **Eval harness change.** `live_sim_eval` does not currently invoke blocked
   agents. This touches the partial-obs path, which is shared with the parallel
   evaluator — scope it deliberately.

## Related

`CONCURRENCY_LIMITATIONS.md` (lost wakeups, option 3),
`data_generation/task_level/generation/image/processor.py` (the injection
pattern to copy).
