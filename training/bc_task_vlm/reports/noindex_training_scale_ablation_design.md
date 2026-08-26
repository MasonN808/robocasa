# No-index training-scale ablation

## Question

How much of the new SFT model's performance comes from improved trajectory and
evaluation semantics, and how much comes from increasing training data from 27
to 90 trajectories per trained task?

## Arms

| Arm | Train/validation trajectories per trained task | Held-out trajectories | Held-out tasks |
|---|---:|---:|---:|
| Previous no-index SFT | 27/3 | 3 × 47 = 141 | 15 × 6 = 90 |
| Current fixed-size SFT | 27/3 | 3 × 47 = 141 | 15 × 6 = 90 |
| Current full-size SFT | 90/10 | 10 × 47 = 470 | 10 × 6 = 60 |
| Current out-of-box models | no SFT | 10 × 47 = 470 | 10 × 6 = 60 |

The current 30-trajectory corpus is a seeded subset of the current validated
100-trajectory corpus. Both current SFT runs start independently from the same
Qwen3-VL-8B base checkpoint and use the same optimizer, epoch count, prompt
format, observation policy, no-index mode, and live-evaluation implementation.

## What each comparison estimates

- Previous 27/3 versus current 27/3: the combined effect of all current data,
  prompt, validator, rendering, and evaluation corrections at fixed data count.
- Current 27/3 versus current 90/10: the effect of additional trajectories
  under the current pipeline. This also includes the additional optimizer
  updates naturally caused by training for the same number of epochs on more
  examples.
- Current 90/10 SFT versus current out-of-box models: the best-current SFT
  comparison under identical current 90/10 evaluation cohorts.

These comparisons do not identify the causal effect of any one protocol change.

## Material changes since the previous 27/3 model

1. A sampled coordinator identity is visible in generation, SFT, and evaluation
   context. Opening coordination is causal: the coordinator proposes first and
   the partner confirms on the next coordination phase, instead of both agents
   independently proposing simultaneous plans.
2. Tick rows are the canonical concurrent plan. The same shared scheduler is
   used by the concurrent FSM and live evaluation for blocking, releases, and
   next-tick wake-up semantics.
3. A release wakes a matching waiter only on a later tick. Same-tick wait and
   release races are rejected.
4. Missing invocations for unblocked agents and invocations from blocked agents
   are rejected through the concurrent FSM's canonical tick validation.
5. Agents may report only their own portion as complete. An agent that finishes
   early reports `portion_complete` and waits instead of emitting filler or a
   premature global-completion claim. An agent assigned no physical work waits
   directly rather than falsely reporting completion.
6. `wait_for_signal` and `communicate.releases` use exact sender/resource
   matching. `releases` is a single optional string, and ordinary communication
   does not wake a waiter.
7. Tool argument schemas are stricter. Irrelevant placement fields are removed,
   object/support/site roles are grounded more precisely, and movable placement
   targets are resolved according to the specific tool.
8. The FSM enforces global single-object ownership: a second agent cannot pick
   up an object already held by its partner.
9. `get_image` remains a supervised model call but is scheduler-transparent:
   observations do not advance physical ticks, waits, releases, or completion
   tails. Training and evaluation history represent observation and wait calls
   consistently.
10. Forced work partitions were removed. Plans remain model-chosen; structured
    random generation and randomized initial states provide diversity without
    requiring inefficient ownership splits.
11. Retry seeds vary across attempts, retry feedback includes the failed
    trajectory and error history, and focused repair feedback is grounded in
    concrete task IDs. These affect which expert trajectories are accepted,
    not the model's evaluation-time retry behavior.
12. Live evaluation has no silent retry. A rejected call is recorded through
    `report_failed`; metrics also retain whether an episode would fail if it
    terminated on its first rejection.
13. Training and live evaluation both use no local or global step indices,
    consume-once agent observations, an 8,192-token context limit, and explicit
    `context_limit_exceeded` accounting.
14. Every selected trajectory is revalidated by the current concurrent FSM and
    must pass real simulator rendering before preprocessing.

## Fixed historical baseline

The previous corrected no-index artifact records:

- held-out trajectories: 75/141 = 53.2%
- held-out tasks: 31/90 = 34.4%

Those historical results remain immutable and are cited from the previous
artifact rather than recomputed with the current evaluator.
