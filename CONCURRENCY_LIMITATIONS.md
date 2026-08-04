# Concurrency model: what we assume, and what breaks without it

**Status as of 2026-08-04.** Tick-format generation validates under `LOCK_STEP`
only (`ConcurrentTaskValidator(inner, models=(LOCK_STEP,))`, commit `4b1364c`).
That is a deliberate scope decision, not an oversight. This file records the
assumption it rests on and what has to change before variable-duration
execution — real VLAs — is sound. **Cite this as a stated limitation.**

## The assumption

Every tool call costs the same, and both agents advance one call per tick. A
tick is a barrier: the tick does not advance until both agents have finished
whatever they are doing.

Under that assumption the tick grid the model writes *is* the execution
schedule, and the coordination protocol is correct by construction.

## Why the assumption is load-bearing

A `releases` field is a **condition-variable signal, not a mutex**. It wakes an
agent that is *already blocked*; if it fires when nobody is waiting, it is
discarded and nothing remembers it (`AgentRuntime.deliver` gates on
`_releases_awaited`). Correctness therefore depends on an *ordering* — the
waiter must block before the holder releases — and that ordering is not
preserved when tool durations differ.

Worked example, real durations (`navigate_to_fixture` 4.0, `communicate` 0.25,
default 2.0), same plan under both regimes:

| | agent_0 blocks | agent_1 releases | verdict |
|---|---|---|---|
| `LOCK_STEP` | t = 4.00 | t = 5.00 | correct |
| `EXECUTOR`  | t = 12.25 | t = 3.00 | **deadlock** |

agent_0 makes three navigations before it can ask; agent_1 sends four cheap
messages before it releases. The release fires nine seconds into an empty room.
Measured on job 266947: **7 of 9 deadlocks were executor-only**, every one of
them clean under lock step. This is the classic **lost wakeup**.

## Both failure directions

Variable timing breaks the plan two ways, and only the first is about releases:

1. **Lost rendezvous.** Contention that existed in tick space evaporates in real
   time, the wait becomes unnecessary, and the discarded release turns it into a
   permanent block.
2. **New collisions.** Two agents on *different* ticks can overlap in real time
   at the same exclusive fixture. No change to release semantics can fix this
   one — it needs barriers or runtime arbitration.

`wait_for_signal` protects the *dependency*. It cannot protect the
*rendezvous*, and it does nothing about (2).

## What must change for variable-duration execution

Three options, not mutually exclusive.

**(1) Barrier-synchronise each tick.** Let VLA execution time vary under the
hood but do not advance the tick until both agents finish. This recreates
lock step exactly and fixes *both* failure directions.
*Cost:* every tick runs at the speed of the slowest agent, and a long navigation
stalls a partner doing unrelated work. Real multi-robot systems avoid global
barriers for exactly this reason — this buys correctness by giving up most of
the concurrency benefit. *Advantage:* requires no change to what the model
learns.

**(2) Make releases persist (a mutex), with re-blocking.** Store "R is free" as
state rather than an event. **Unsound on its own**: if an agent later
re-acquires R, a waiter wakes on stale information and walks into an occupied
fixture. Needs re-acquisition to re-block, which in turn obliges the model to
re-release when it is done — new semantics the model must be taught, and a new
error class when it forgets.

**(3) Make the wait conditional — `while (!predicate) wait()`.** Have
`wait_for_signal` check occupancy at the moment of blocking and return
immediately if the resource is already free. This is the standard lost-wakeup
fix, and it is **strictly cheaper than (2)**: no persistent state, no
re-release obligation, no re-blocking logic, because the predicate is
re-evaluated rather than remembered. The predicate already exists — the FSM
tracks occupancy (`_occupancy()`), which is what the contention invariants are
built on.
*Cost:* a wait can then only mean "wait until this thing is free", never "wait
until you have finished brewing". Since `about` already names a THING and not
an event, that is the semantics we have anyway.

Fixing (1) alone is sufficient. Fixing (3) alone addresses lost wakeups but
leaves the new-collision case, so it wants runtime arbitration alongside it.

## For the write-up

State plainly: results are under a lock-step execution model where each tool
call occupies one tick and agents synchronise at tick boundaries. Coordination
correctness is verified under that model. Extending to asynchronous
variable-duration execution requires either tick-level barriers or a conditional
wait, because the handover protocol is a condition-variable signal and is
subject to lost wakeups.
