
# Design: centralized Task-VLM vs distributed partial observability

Status: partial-observability **v1 and v3 off-sim are implemented** (dataset
builder, eval harness, manifest builder, tests) and a partial-v1 adapter is in
training. The distributed runtime (per-agent scheduler, concurrency, live-sim)
remains design-only. Sections marked "Superseded" or "Revised July 2026" were
changed by corpus measurement — see Corpus measurements behind the July 2026
revisions.
Date: 2026-07-22, revised 2026-07-25

## Purpose

The current Task-VLM is centralized: one prompt contains the joint symbolic
history, the model chooses which agent acts next, and the dataset builder uses
the known target agent to select that agent's images. This is convenient for
teacher-forced SFT but is not causally reproducible in live inference.

The target architecture is instead a shared-weight distributed policy:

- one model and one model server;
- two logical agent sessions with separate private state;
- agent-owned `get_image` results, including global-camera results;
- observations consumed by the next decision, so every attached image is fresh
  by construction (revised; see Memory lifetime and freshness);
- concurrent agent decisions and, eventually, overlapping physical work;
- no learned `yield_control` or centralized next-agent oracle;
- no global step number as an input dependency.

The same weights serve both agents. "Distributed" refers to information,
history, cache, and execution semantics, not to separate checkpoints or model
servers.

## Decision summary

1. Each logical agent owns its visual cache, private history, and inbox.
2. A `get_image` result is revealed only to the requesting agent.
3. Identical rendering may be memoized internally without granting access to
   an agent that did not request it.
4. **(Revised July 2026)** A `get_image` result feeds that agent's next target
   tool call and is then discarded (`--partial-observation-mode consume-once`).
   Every attached image is therefore fresh by construction, and no freshness
   metadata is emitted — a freshness flag would leak the other agent's hidden
   activity for the 20% of staleness it alone causes. See Memory lifetime and
   freshness.
5. Physical actions require a fresh, agent-owned observation with appropriate
   views. Under `consume-once` this holds automatically, and the corpus already
   satisfies it: 0 of 9,153 physical-action targets carried a stale image.
6. Both agents may request decisions independently whenever idle. Their
   requests are multiplexed or dynamically batched by one model server.
7. The episode runtime is a deterministic event scheduler, not an intelligent
   central policy and not a handoff protocol.
8. Model inputs should omit the global step number. Logs retain per-agent turn
   indices and shared world-event indices for traceability.
9. A causal centralized Task-VLM remains a valid baseline: it sees the complete
   joint prefix and both agents' prefix-derived caches, then predicts which
   agent acts. It must not use the current target to construct those inputs.
10. Rejected or non-invoked proposals never update agent-visible state. Only
    accepted/executed events call the state-update function; a model call that
    is generated but not accepted must not be appended to any agent's private
    history, cache, or inbox, even for the agent that proposed it.
11. The runtime scheduler is per-agent, not a shared central arbiter. Each
    logical agent runs an independent loop that blocks only on its own
    outstanding tool call, exactly like a standard single-agent tool-use loop.
    The only shared state is the environment itself (resource locks, inboxes,
    world version). Resource contention surfaces to the losing agent as an
    ordinary tool-call error, not as a joint decision the model must learn to
    coordinate.
12. Sequencing dependencies between agents (agent 1's action requires agent
    0's action to have actually completed) are enforced as FSM legality
    preconditions on the dependent tool call itself, never inferred from
    message content. This keeps the check consistent with partial
    observability: it reveals only whether the caller's own action is
    currently valid, never the other agent's private state or reasoning.
13. The model never predicts the acting agent under partial observability: the
    caller **is** the actor. Agent prediction is a centralized-scheduling
    construct; in the distributed design the WHO decision belongs to the
    runtime's per-agent loops (decision #11), not the policy. `partial_history`
    with `predict_agent` is rejected at build time.
14. Model inputs carry no joint step index. `--partial-step-index-mode none`
    (default for partial runs) omits indices entirely; `local` renumbers per
    agent. The joint index leaks the other agent's activity through the gaps
    between this agent's turns. Measured cost of removing it: none — all three
    modes scored within noise zero-shot (exact tool-call .323 global / .325
    local / .328 none, 95% CI half-width ~.028).

## Why the current centralized contract is insufficient

The current v3 example builder has four properties that conflict with a
distributed partial-observability interpretation:

1. Every `get_image` target is built with zero images, even when that agent
   already has a valid observation from an earlier request.
2. Every non-`get_image` target receives images selected using the known target
   agent. Live inference must choose its input before it knows the output agent.
3. Global-only requests are normalized to a placeholder agent, erasing which
   logical agent acquired the observation.
   private tool calls and actions, whether or not they were communicated.

This creates a circular live-inference dependency:

```text
choose images using the predicted tool and agent
    -> predict the tool and agent using those images
```

Step-parity rules and image-free routing passes do not solve this generally.
The demonstrations contain consecutive image requests and consecutive actions,
so neither input pattern is implied by the global step number.

## Causal centralized Task-VLM baseline

The current training approach is causally wrong even if the intended policy
remains centralized. The problem is not created by partial observability.

For each supervised example, the current builder knows the target event—the
next agent, tool, and arguments—and uses at least the target agent to decide
which images are attached. From the input's point of view, that target is
future information. At live inference time, the centralized model must receive
its input before it has predicted the next agent and tool, so it cannot
reproduce the teacher-forced image selection rule.

This is label leakage even though the builder need not inspect several later
events. Looking at the current target event is already one-step lookahead. The
fix is to construct every input entirely from the executed trajectory prefix,
using persistent per-agent caches:

```text
for event in chronological_event_stream:
    input = snapshot(
        full_joint_prefix_history,
        agent_0_cache_from_prefix,
        agent_1_cache_from_prefix,
        public_world_metadata,
    )
    target = event.tool_call_with_acting_agent
    emit(input, target)
    execute event
    update history, caches, and freshness
```

For this centralized baseline:

- the model receives the completed prefix history of both agents, never the
  unexecuted future suffix;
- it may receive both agents' existing image caches, clearly labelled by owner,
  views, capture version, and freshness;
- each cache contains only observations produced by earlier `get_image` events;
- a `get_image` target receives the pre-request caches rather than being made
  image-free automatically;
- after `get_image` executes, only the requesting agent's cache is updated;
- a physical-action target receives exactly the caches that existed before the
  action, regardless of which agent the target says will act;
- the model predicts the acting agent as part of each tool call;
- cache attachment, staleness handling, and invalidation are deterministic
  functions of prefix state, never of the current or future targets.

Passing both caches may increase the image-token cost. That is a performance
tradeoff to benchmark, not a reason to select one cache using the answer.
Target-independent compression, deterministic view limits, or explicit cache
metadata are valid optimizations if they are identical during training and
live inference.

The main architectural distinctions are:

| Property | Causal centralized Task-VLM | Distributed partial observability |
| --- | --- | --- |
| Model weights/server | One shared model and server | One shared model and server |
| Logical session | One joint policy session | One private session per agent |
| History visible to a call | Full completed prefix for both agents | Caller history, inbox, and defined public events |
| Images visible to a call | Both prefix-derived agent caches | Only the caller's prefix-derived cache |
| Acting agent | Predicted with each tool call | Fixed by the caller identity |
| Who chooses the next agent | Central policy output | Event scheduler invokes every eligible agent |
| Communication | Visible in joint history and potentially redundant for information access | Required bridge for non-public information |
| Concurrent decisions | Requires a joint-action output or an explicit central scheduling extension | Independent eligible-agent requests can run concurrently |
| Privacy boundary | No internal agent privacy | Histories, caches, and invocation events are agent-scoped |

Both variants share the same causal invariant:

```text
model_input = function(executed_prefix_state)
model_input != function(current_target_or_future_suffix)
```

The centralized version is therefore a useful controlled baseline for testing
the cache correction separately from the additional effects of privacy,
fixed-caller identity, and concurrent scheduling.

## Near-term existing-v3 live-sim compatibility

The causal centralized training baseline above is a future training contract.
Passing both prefix-derived caches would be valid for a fully observable central
policy, but it is not the right compatibility intervention for the already
trained v3 adapter. That checkpoint saw one selected visual bundle per example,
not both agents' visual inputs.

For new diagnostic live-sim runs of the existing v3 checkpoint, use a
causal **single-cache** or **fake-partial-observation** protocol:

- retain one centralized model session and the full joint symbolic history;
- continue predicting the acting agent with every tool call;
- maintain separate logical image caches for agent 0 and agent 1;
- attach at most one agent's cache to any proposal;
- select that cache only from prior executed events, never from the tool call
  currently being predicted;
- identify the prefix-derived cache owner explicitly in the prompt;
- begin image-free;
- after `get_image(agent, views)`, update only that agent's cache and make it
  the active visual context;
- allow communication to use only the caller's own active cache, and retain
  that context because communication does not mutate the physical world;
- clear the active context after a successful physical action;
- require a physical action to name the active cache owner;
- keep global-camera access logically agent-specific even when identical
  rendering can be memoized.

This protocol may show an existing cache while the model requests another
image. The legacy SFT builder forced every `get_image` target to be image-free,
but the explicit `--causal-single-cache` training mode now follows this causal
contract. Existing v3 checkpoints retain the old mismatch and should be
treated as diagnostics; matching checkpoints must be retrained.

This is not genuine partial observability: the central model still sees both
agents' completed symbolic history and centrally chooses which agent acts. It
only prevents a single proposal from jointly processing both agents' visual
inputs.

The protocol is also the closest causal continuation of the existing off-sim
evaluation family:

- v1 uses the joint expert prefix, a fixed acting agent, and one selected
  visual bundle;
- v2 uses the joint expert prefix and one selected visual bundle, while the
  model predicts the acting agent;
- the near-term v3 live protocol uses the joint on-policy prefix, at most one
  prefix-selected visual bundle, and predicts both `get_image` and acting-agent
  tool calls.

They are therefore architecturally comparable as centralized Task-VLMs, while
their metrics remain different: off-sim v1/v2 measure exact or judged
next-step imitation under expert prefixes, whereas live-sim v3 measures task
outcome under the model's own prefix. That distinction is the intended
scientific comparison, not a reason to make the live input target-dependent.

## Evidence from the local trajectory subset

The July 2026 audit covered all 162 trajectories in
`training/bc_task_vlm/eval_data_subset`:

- 5,724 total steps;
- 3,314 `get_image` steps;
- 2,410 non-image targets;
- exactly one opening global request per agent per trajectory;
- 162 agent-0 and 162 agent-1 global requests;
- 1,596 agent-0 and 1,394 agent-1 local-camera requests.

All 486 paired opening global files (three views times 162 trajectories) are
byte-identical between the two agents. This supports render memoization, but
not access sharing.

For the 2,410 non-image targets:

- every target had a cache owned by its acting agent;
- 2,093 had no intervening physical world mutation;
- 317 used a cache made stale by intervening mutations;
- all 317 stale-cache targets were `communicate`;
- every physical action had a fresh cache;
- every physical action was the first action-level use of that cache;
- 127 targets reused a cache after an earlier decision by the same agent, and
  all 127 were communications.

For the 3,314 image-request targets:

- 324 were an agent's first request, exactly two per trajectory;
- 2,990 occurred when that agent already had a private cache;
- 900 of those prior caches were still fresh;
- among the 900 fresh-cache requests, 592 requested a different view family
  and 308 repeated the same view family.

The 900 fresh-cache requests are direct evidence that `get_image` targets must
sometimes be trained with existing images. The 308 fresh same-view repetitions
should be reviewed or deduplicated rather than automatically taught as useful
behavior.

## Logical agent state

The runtime maintains separate state for each logical agent:

```text
AgentState:
    agent_id
    private_history
    inbox
    latest_observation:
        image_paths
        views
        requested_by
        captured_world_version
        captured_sim_time
    local_turn_index
```

The environment separately maintains:

```text
WorldState:
    world_version
    sim_time
    event_index
    active_tools
    active_resource_uses
    render_memo
```

A model invocation for agent 0 may attach only agent 0's latest observation.
It may contain messages sent by agent 1, but never agent 1's images or private
tool history.

## Agent-specific global observations

Global cameras are visually agent-independent but informationally
agent-specific.

```text
agent_0 calls get_image(global_views)
    -> render or memo lookup
    -> store a logical result only in agent_0 state

agent_1 later calls get_image(global_views) at the same world version
    -> reuse the rendered bytes
    -> store a distinct logical result in agent_1 state
```

The render memo key should include at least:

```text
(world_version, view_set, render_size, map_renderer, map_dpi)
```

Memo reuse is an implementation optimization. It must still produce a
request/result event for the second agent and must not pre-populate its cache.

The existing global-request agent normalization should be removed. Requester
identity is part of the partial-observability state even when the pixels are
identical.

## Memory lifetime and freshness

**Superseded (July 2026).** This section previously recommended a persistent
per-agent cache plus explicit `Fresh: false` metadata, and rejected one-step
image consumption as "artificial amnesia". Measurement on the full training
corpus reversed both conclusions. The implemented default is
`--partial-observation-mode consume-once`: a `get_image` result feeds that
agent's **next** target tool call and is then discarded. `cache` is retained
as a flag value for ablation.

Three measurements drove the change:

| finding | measurement |
| --- | --- |
| The cache does no work where vision matters | **100%** of 9,153 physical-action targets are immediately preceded by that agent's own `get_image` — zero exceptions |
| Most cached pixels are wrong | **70%** of 18,306 `get_image` targets with a prior cache were shown a *stale* image; 34% of communicate targets likewise |
| Physical actions never act on stale pixels | **0** of 9,153 physical targets had a stale cache |

Under `consume-once` every attached image is fresh **by construction**, so the
freshness question does not arise: there is no stale pixel to label or drop.

### Why a freshness label would leak

The earlier `Fresh: false` proposal is not privacy-neutral, and this is the
decisive argument against it. Splitting the causes of staleness:

| cause | share of `get_image` targets |
| --- | --- |
| the agent's **own** prior action (inferable from its private history) | 50% |
| genuinely fresh | 30% |
| **the other agent's action only — invisible to this agent** | 20% |

For that last 20%, telling agent 1 "your observation is stale" reveals, in one
bit, that agent 0 acted. That is the same covert channel as the joint step
index (see Step numbers and clocks), and it is forbidden by the same
reasoning. A self-caused-only freshness label would be privacy-clean but
redundant, since the agent's own history already implies it.

The correct response to the unknowable 20% is **defensive re-observation** —
which is also what a real robot should do when it cannot see what its teammate
has been doing. Training data already teaches this: the expert always looks
before acting. Watch `redundant request rate` in evaluation to quantify the
cost of that uncertainty.

### Residual role of the legality gate

Off-sim training never exercises a stale-action case (0 of 9,153), so the
legality gate is not needed to build the dataset. It remains worthwhile in
live-sim, where a model's own choices could reach a state the demonstrations
never contain.

At minimum, the following tools mutate pose or scene state and invalidate
relevant fresh observations:

- navigation and `give_space`;
- pickup and placement;
- opening or closing fixture parts;
- button presses;
- any future tool that moves a robot, fixture, or object.

`communicate`, `wait`, and observation lookup do not mutate the physical world.

## Shared weights, separate agent invocations

Training one model does not require a centralized prompt. Both agents use the
same weights, tokenizer, adapter, and tool schemas, but each inference request
has a fixed caller identity:

```text
Current logical agent: agent_1
Private observation: agent_1 cache only
Private history: agent_1 calls and results
Inbox: messages delivered to agent_1
```

The action call should not predict a different acting agent. The caller is the
actor. The `agent` tool argument can be removed, or retained only as a validated
redundant field during migration.

Messages are the explicit information bridge:

- the sender sees the message it emitted;
- the recipient receives the text in its inbox;
- images are never transferred implicitly;
- another agent's action is not automatically private-history knowledge.

Publicly observable events must be defined explicitly rather than inherited
from the current global history.

## Agent invocation protocol (brainstorming)

The model does not decide when its own weights execute. An event-driven runtime
decides when to invoke each logical agent; the model decides what that agent
does once invoked.

Each agent should have an explicit runtime state:

```text
READY
    -> INFERENCE_PENDING
    -> TOOL_RUNNING | WAITING_FOR_IMAGE | STANDBY
    -> READY after an agent-visible wake event
```

An agent is eligible for inference only when:

1. it is idle and has no outstanding inference or tool call; and
2. its observable state version has advanced since its previous decision.

Candidate agent-visible wake events are:

- episode start;
- that agent's physical tool completing, failing, or being rejected;
- that agent's requested image becoming available;
- a message arriving in that agent's inbox;
- a resource that agent explicitly waited for becoming available;
- an explicitly public environment event;
- an agent-owned timer expiring, if timers are part of the contract.

Multiple wake events should be coalesced into one state update before inference.
The runtime must not continuously poll the model while nothing observable has
changed.

Invocation timing is itself information. A private action by agent 0 must not
wake agent 1 merely because the centralized scheduler knows it happened. Agent
1 may be awakened only if the event is visible under the documented observation
contract, changes a resource it explicitly waits for, or produces a delivered
message. Otherwise, the timing of the model call would leak hidden state.

At most one inference request may be outstanding per agent. Each logical
request should have an idempotency identity such as:

```text
(episode_id, agent_id, local_turn_index, observable_state_version)
```

Retrying that identity after a transport failure must not create a second
logical decision or execute the returned tool twice. A response produced for
an older observable-state version must be rejected or explicitly revalidated.

## Concurrency with one model server

`yield_control` is not part of the target design. Both agents remain eligible
to work whenever they are idle and unblocked.

One vLLM server can serve both logical agents:

```text
one base model + one adapter
    <- agent_0 inference requests
    <- agent_1 inference requests
```

The requests have independent messages, images, caches, and request IDs. vLLM
may dynamically batch them on one GPU. Prefix-cache reuse is acceptable, but
conversation or KV state must never be shared semantically between agents.

### Per-agent loop, not a shared central scheduler

The scheduler is per agent, modeled directly on an ordinary single-agent
tool-use loop rather than a joint arbiter. Each logical agent runs its own
independent loop:

```text
while episode not terminal:
    if agent has an outstanding tool call:
        block until that specific call returns
    else:
        propose(agent_state[agent])
        submit the tool call to the shared environment
    on result: append to this agent's own private_history only
```

"Waiting for a tool" is therefore not a learned decision, the same way a
single-agent coding loop does not decide to wait for its own shell command —
the harness simply does not re-invoke the agent until its own call resolves.
Two agents running this loop independently need no joint decision batch,
snapshot-and-resolve step, or shared eligibility state machine; the only
shared component is the environment underneath both loops (resource locks,
inboxes, world version, FSM legality). Concurrency emerges from running two
independent loops against one shared environment, not from an explicit
central event loop that decides both agents' turns at once.

Resource contention is therefore an ordinary tool-call error, not a joint
decision: if agent_0's call would conflict with a lock agent_1 currently
holds, the environment simply returns a failed result to agent_0's own loop,
which reacts to it the same way it would react to any other tool error
(retry, communicate, try something else). No new "standby" primitive or
central conflict-resolution batch is required for this case (see Resource
conflicts and Sequencing dependencies below).

### Concurrency implementation levels

True simultaneous robot motion should be introduced in stages.

#### Level A: concurrent decisions, lockstep execution

- Both idle agents propose from the same pre-action world snapshot.
- Requests may be submitted concurrently to one model server.
- Non-conflicting high-level tools are accepted as one joint decision batch.
- The current executor may run accepted calls in deterministic order, but
  neither proposal sees the other's result before generation.
- Only tool pairs demonstrated to be order-independent may use this shortcut.

This is the minimum useful distributed evaluation and is compatible with one
simulator process.

#### Level B: event-time overlapping high-level tools

- Tools have start time, duration, completion time, and resource sets.
- Navigation or manipulation by one agent can overlap with disjoint work by
  the other.
- Observations and messages are timestamped.
- Cache freshness changes when relevant events begin or complete.

#### Level C: simultaneous low-level control

- Both robots step through the physics engine together.
- Collision avoidance, shared workspace constraints, and interruptible actions
  are evaluated continuously.

Level C is the realistic endpoint but requires changes below the current
high-level trajectory executor. Level A should be implemented and validated
before claiming physical concurrency.

## Resource conflicts

Each proposed high-level tool declares resources such as:

```text
robot
object
fixture
workspace zone
receptacle or destination
```

Because the scheduler is per-agent (see above), resource conflicts do not
require a joint decision batch. The environment holds the lock table; each
agent's independent loop calls into it and gets an ordinary success or
failure result for its own attempted call. Conflicting proposals must not gain
an arbitrary hidden advantage from Python call order, but this is an
environment-level atomicity property (like a mutex acquire), not a property
that requires collecting both agents' proposals before deciding either one.

Initial deterministic policy (**revised**: rejecting *both* proposals livelocks
under deterministic decoding — see the implementation findings section. A
visible, constant priority rule now decides which proceeds):

- the losing call fails with a structured, ordinary tool-call error (e.g.
  resource held by the other agent);
- the losing agent's own loop reacts to that error like any other tool
  failure — retry, communicate, or attempt something else;
- record the conflict as a coordination event for metrics, not a harness
  error, and not as a joint "reject both" decision the model must learn.

A seeded priority or reservation protocol may be added later, but must be
visible to the agents (via the ordinary error result) and constant across
compared runs.

## Sequencing dependencies and FSM legality gating

A distinct problem from resource contention: agent 1's action can depend on
agent 0's action having actually *completed*, independent of any resource
lock (e.g. agent 1 places an item that requires agent 0 to have already
retrieved it). Agents may narrate this dependency ("I'll wait for your
confirmation") in `communicate`, but the runtime must not parse message
content to decide whether the dependent action is allowed — that would
require semantic understanding of the message and would blur the privacy
boundary.

Instead, the dependent tool call carries an ordinary FSM legality
precondition, the same mechanism already used for observation freshness:

- the FSM/task graph declares that agent 1's action requires agent 0's
  action's real world-state effect, not agent 0's message;
- if agent 1's own loop attempts the action before that effect exists, the
  tool call fails with an ordinary precondition-not-met error, exactly like a
  resource conflict;
- agent 1's loop reacts to the failure the same way it reacts to any other
  tool error, and may retry after (real or eventual) confirmation.

This does not compromise partial observability: the rejection reveals only
that the caller's own action is currently invalid, never agent 0's private
state, images, or reasoning — the same class of feedback as a resource-lock
failure. It also means no `standby`/`wait` action is required to solve this
case: an early attempt simply fails and the agent's ordinary tool-use loop
retries, at the cost of a wasted call rather than a correctness problem (see
zero-progress tracking below). Key the legality check to the real task
dependency in the FSM goal graph, not to the verbal commitment — the message
is narrative, the precondition is the enforcement mechanism.

As of the July 2026 trajectory audit there is no evidence of an
*information*-blocked case in the existing demonstrations (an agent lacking
information it has no resource or sequencing dependency for) — only
resource- and sequencing-blocked cases, both handled above without a learned
wait primitive. Revisit if live-sim collection surfaces a genuine
information-blocked case.

## Standby, progress, and deadlock (brainstorming)

Resource contention and sequencing dependencies are handled above via
ordinary tool-call failures on the per-agent loop, without a new action. A
durable `standby` primitive remains open only for a case not yet evidenced in
the data: an agent that has no resource or sequencing block on any action,
but genuinely has nothing useful to attempt. Do not build this speculatively;
revisit only if live-sim smoke testing surfaces it. If added, it is not
`yield_control`:

- `yield_control` implies exclusive turn-taking or transfer of control;
- `standby` changes only the caller's state and does not prevent the other
  agent from acting;
- the caller is not invoked again until one of its declared, visible wake
  conditions occurs.

Candidate wake conditions include:

```text
message_received
resource_available
own_timer_expired
own_tool_result
public_event
```

The exact schema remains an open decision. A predicate such as
`resource_available` must name the resource narrowly enough that unrelated
world changes do not wake the agent and leak information.

The runtime should also track whether a decision caused progress:

- physical world mutation;
- a newly acquired observation;
- a newly delivered message;
- acquisition or release of a required resource;
- advancement of a task goal;
- a meaningful change in the agent's declared wait condition.

Repeated identical observation, communication, rejected-action, or standby
cycles with no progress must not run indefinitely. Candidate safeguards are a
per-agent consecutive-zero-progress limit, duplicate-call counters, and a
global deadlock detector. These safeguards should terminate or classify the
episode as a policy loop, budget exhaustion, or deadlock, not as a harness
error.

If both agents are in standby, no tool is running, no scheduled visible event
can wake either one, and the task is not complete, the episode is deadlocked.
Evaluation should record the agents' final wait conditions and observable
state versions so the failure is diagnosable.

## Step numbers and clocks

The global step number should not be a model input. It is neither private agent
state nor a reliable observation/action phase marker.

Keep three values in logs:

- `agent_turn_index`: incremented independently for each model invocation;
- `world_event_index`: incremented when a scheduled environment event commits;
- `sim_time`: the simulated timestamp used for overlapping tools.

Training examples may retain these as metadata. If a step number is included in
the prompt for an ablation, it must be the local agent turn index, never the
joint demonstration index. No cache rule may depend on parity or an absolute
step number.

## Causal training construction

Training examples are built from the state before the target event:

```text
for event in chronological_event_stream:
    agent = event.owner
    input = snapshot(agent_state[agent], public_state)
    target = event.tool_call
    emit(input, target)
    execute event and update only causally affected state
```

Input construction may not inspect the current target tool or another agent's
private state.

Each sample should also correspond to a valid invocation boundary. In other
words, the example input is the agent-private state after an agent-visible wake
event, not an arbitrary slice of a global step sequence. Candidate training
metadata includes:

```text
agent_id
local_turn_index
observable_state_version
wake_reason
world_event_index
sim_time
```

`wake_reason`, global event indices, and simulation time should initially be
metadata rather than prompt features unless an experiment shows that exposing
them is both necessary and causally valid. If `standby` is adopted, training
must include genuine standby decisions and their wake conditions; otherwise
the model cannot be expected to learn quiescence from action-only
demonstrations.

For a `get_image` target:

- attach the requester's current private cache if one exists;
- include freshness and view metadata;
- execute the request only after the target example is formed;
- store the returned result only in the requester's state.

For a physical-action target:

- require a fresh compatible cache already present in the prefix state;
- reject or repair demonstrations that violate this condition;
- mark affected caches stale after the action changes the world.

For communication:

- deliver only the message text to the recipient;
- preserve sender and recipient private-state separation;
- do not reveal the sender's images.

## Converting existing sequential demonstrations

Existing data can bootstrap the distributed policy, but it does not by itself
demonstrate true simultaneous execution.

Conversion should:

1. Preserve the original requester on every global and local image call.
2. Build separate private histories and caches for each agent.
3. Remove target-agent cache selection.
4. Filter joint history into own actions, received/sent messages, and explicitly
   public events.
5. Build inputs from prefix state before applying the target event.
6. Track freshness using mutations from either agent.
7. Audit or remove redundant fresh same-view requests.
8. Store local agent turn indices as metadata and omit global step indices from
   prompts.

To train concurrency rather than only interleaving, add a partial-order layer:

- per-agent program order;
- observation-before-action dependencies;
- message send/receive dependencies;
- object, fixture, and workspace resource dependencies;
- task-specific goal dependencies.

Independent events may be assigned overlapping timestamps or the same decision
batch. Ambiguous dependencies should remain sequential until validated. New
data generated with an event-aware concurrent expert will eventually be needed
for Level B and Level C behavior.

## Evaluation with one server

A distributed live-sim episode uses:

- one task environment;
- two logical agent clients;
- one model server and one adapter;
- two private histories and observation caches;
- one deterministic event scheduler.

At a decision boundary, requests for both ready agents are sent concurrently.
With a non-batching backend, they may be generated sequentially for performance
reasons only if both inputs were frozen from the same snapshot and execution
waits for both outputs. The semantic result must not depend on request order.

Evaluation must record:

- model request start/end and batch membership;
- agent-local turn index;
- proposal snapshot world version;
- tool start/end time;
- resources held;
- cache owner, capture version, and freshness;
- messages and delivery time;
- conflicts, retries, and legality failures.

Native and FSM success are checked after committed events, not between the two
members of a joint proposal batch.

## Metrics

Task outcome remains primary:

- native success;
- FSM goal completion and partial-goal fraction;
- native/FSM agreement;
- harness, parsing, simulator, and native-check errors.

Distributed-efficiency metrics should include:

- simulated makespan and critical-path efficiency;
- per-agent active, inference, blocked, and idle time;
- fraction of simulated time with both agents productively active;
- model-server batching and throughput;
- action and observation counts per agent;
- redundant request rate;
- cache freshness violations;
- attempts to act using another agent's cache;
- communication count and delivery latency;
- resource-conflict and collision rate;
- physical-action efficiency per successful run.

Raw step count alone is not a sufficient efficiency metric once work overlaps.

## Required tests

### Dataset contract

## Comparing success and communication effectiveness

A central research goal is to compare centralized and partially observable
task success under matched conditions. Hold constant the base model, adapter
capacity, training trajectories, split, task initial states, tool interface,
rendering, and evaluation budget. Vary only the information and scheduling
contract where possible.

The comparison can reveal more than whether privacy hurts success. It can help
measure whether explicit agent communication is actually doing useful work:

- In the centralized policy, one model sees both agents' completed histories.
  Communication may help structure the plan, but it is not the only route by
  which the central model learns what both agents did.
- In the partially observable policy, the other agent's private history and
  images are hidden. Delivered messages are the explicit information bridge,
  so communication should have greater causal value.

Candidate matched conditions are:

| Condition | Centralized policy | Partial policy |
| --- | --- | --- |
| Normal | Joint history and normal communication | Private histories and delivered messages |
| Message-content mask | Preserve calls/timing but hide message text | Preserve calls/timing but hide delivered text |
| No-delivery | Record sends but do not deliver information | Record sends but do not deliver information |
| Oracle-message diagnostic | Replace selected messages with expert content | Replace selected messages with expert content |
| Delay/noise diagnostic | Delay or corrupt messages deterministically | Delay or corrupt messages deterministically |

The main communication-effect estimate within each architecture is:

```text
communication_value = success(normal) - success(message_ablated)
```

The difference between those effects is an interaction estimate:

```text
partial_observability_communication_gain =
    communication_value(partial) - communication_value(centralized)
```

Useful accompanying metrics include:

- native and FSM task success;
- partial-goal completion;
- successful-run efficiency and makespan;
- messages, message tokens, and repeated-message rate;
- failures following missing, late, or contradictory information;
- resource conflicts and duplicated work;
- how often an agent acts consistently with information available only through
  a delivered message.

These ablations are distribution shifts if applied only at evaluation time.
Treat them first as diagnostics, and train matched masked/no-delivery variants
before making strong causal claims. Message quality also should not be reduced
to exact text match; use task outcome, semantic message judging, and downstream
behavior together.

- Prefix invariance: changing the current or future target cannot change the
  current input images or private history.
- Agent isolation: agent 0 never receives agent 1 images or private calls.
- Request ownership: every image result enters only the requester cache.
- Memo/access separation: a render memo hit still requires a logical request.
- Consecutive request and consecutive action examples remain causal.
- No cache behavior depends on step parity.

### Runtime contract

- Simultaneous requests use the same pre-decision world snapshot.
- An agent is invoked only after its observable state advances.
- Hidden events do not wake an agent or leak information through call timing.
- At most one inference and one high-level tool are outstanding per agent.
- Retried inference identities cannot execute the same decision twice.
- Request order does not change accepted non-conflicting outcomes.
- Conflicting calls receive deterministic structured feedback.
- A stale or foreign cache cannot authorize a physical action.
- Communication does not transfer images.
- Standby agents wake only for declared, agent-visible conditions.
- Zero-progress loops and terminal deadlocks are classified without polling.
- One vLLM server can serve both independent agent sessions without state
  leakage.

### Smoke tasks

- global-view requests by both agents with one physical render;
- consecutive per-agent view changes;
- concurrent navigation to disjoint fixtures;
- disjoint simultaneous manipulation;
- shared-fixture conflict and recovery;
- communication based on stale private memory;
- a navigation-heavy task with measurable overlap;
- native/FSM success agreement after joint event batches.

## Staged roadmap: partial observability starts off-sim at v1

Do not begin the partial-observability implementation with active observation,
live simulation, or concurrency. First isolate the information-boundary change
in the simplest existing training/evaluation setting.

1. **Near-term existing-v3 diagnostic:** use the causal single-cache live-sim
   mode described above. Do not call it a true partial-observability result and
   do not use it as the training specification for the future partial policy.
2. **Partial-observability v1 dataset — implemented:** a `partial_history`
   flag on `_build_centralized_examples_for_trajectory`/
   `build_centralized_examples` (`training/bc_task_vlm/dataset.py`) replaces
   the flat joint `history_steps` with a per-agent `private_history` dict:
   each step's example sees only the acting agent's own prior actions plus
   `communicate` events delivered to it (recipient appended at execution
   time, not before). Images were already agent-owned via
   `latest_observations_by_agent`, so no change was needed there. Guarded to
   require `predict_agent=False` and `train_get_image=False`. Threaded as
   `--partial-history` through `main.py`, `eval_standalone.py`, and
   `preprocess.py`, alongside the existing `--causal-single-cache` flag.
   Tests: `tests/test_bc_task_vlm_partial_observability.py` (agent isolation,
   message delivery timing, prefix invariance, the flag-combination guard,
   and cache-fingerprint invalidation).
3. **Partial-observability v1 off-sim training — supported, not yet run:**
   `main.py --partial-history` trains a fresh adapter at the same scale as
   centralized v1; no dedicated partial-history adapter has been trained yet
   (see the diagnostic sweep below for why).
4. **Matched off-sim evaluation — harness implemented, diagnostic sweep
   pending a dedicated adapter:** `eval_standalone.py --partial-history`
   reuses the existing teacher-forced `EvalSample`/`finalize_metrics` scoring
   loop unchanged, just fed per-agent-private examples. Before training a
   dedicated adapter,
   `training/scripts/run_partial_observability_backend_sweep.sh` runs the
   same partial-history-formatted off-sim samples through five backends with
   no dedicated partial-history training (Gemini 3 Flash, a second Gemini
   model, base Qwen3-VL-8B-Instruct, and the existing centralized v1/v1.5 SFT
   adapters evaluated out-of-distribution on this prompt shape) to gauge
   whether the privacy restriction is costly before committing training
   compute. Results from the two SFT checkpoints are diagnostics, not
   partial-observability results, per the same reasoning as the v3
   checkpoint interpretation below.
5. **Off-sim communication diagnostics:** run the message-content and delivery
   ablations where meaningful. Inspect whether private agents use messages to
   predict actions that the centralized policy can infer directly from joint
   history.
6. **Causal active-observation dataset (partial v3) — implemented:** `get_image`
   becomes a supervised target with caller identity still fixed
   (`partial_history=True, train_get_image=True, predict_agent=False`; the
   guards were relaxed to permit exactly this combination). Changes required:
   the global-request agent relabel is disabled under partial history so the
   true requester is preserved (verified 21/21 agent split where the old code
   filed all 42 under agent_0); `get_image` enters the tool specs via
   `augment_tool_specs_with_get_image` without dragging in `task_complete` or
   the agent argument; and a new `SYSTEM_PROMPT_ACTIVE_OBSERVATION_FIXED_AGENT`
   replaces both the v1 prompt (which forbade `get_image`) and the centralized
   v3 prompt (which asked the model to choose an agent). No `task_complete`:
   off-sim is teacher-forced, so nothing needs to decide termination, and an
   agent with private-only history often cannot know the joint task is done.
   Eval manifests must be rebuilt with `--train-get-image`; the pre-existing
   exp52 manifests contain **zero** `get_image` targets and `eval_standalone`
   silently skips manifest rows with no matching example, so reusing them
   would quietly score only the v1-shaped subset.
7. **Single-agent-at-a-time live smoke:** run the partial policy with a fixed
   caller scheduler before adding concurrent inference or overlapping tools.
8. **Concurrent live evaluation:** add event-driven invocation, one shared
   model server, separate agent sessions, conflict handling, and Level A joint
   decision batches.
9. **Physical overlap:** add event durations and Level B/Level C concurrency
   only after the private-state and communication contracts are stable.

At every stage, use fresh adapters rather than continuing the existing v3
checkpoint, which learned target-conditioned image presence. This ordering
creates interpretable intermediate comparisons and keeps a failure in
partial-observability v1 from being confused with `get_image`, scheduler, or
simulator failures.

## Evaluation coverage: privacy x sim mode x control mode

Coverage for centralized-vs-partial and off-sim-vs-live-sim is not
complete without a third axis: whether the model's output actually
determines what happens next (**execution-driving**) or whether WHO and WHEN
are fixed from the ground-truth trajectory and the model is only scored
(**passive-scoring**). This axis is what separates today's off-sim v1/v2
(passive-scoring) from a genuine test of the per-agent scheduler
(execution-driving).

| | passive-scoring | execution-driving |
| --- | --- | --- |
| **Centralized, off-sim** | existing v1/v2: fixed or predicted acting agent, scored against the expert prefix | marginal value: one session has no real concurrency to schedule, so this degenerates close to v2 |
| **Centralized, live-sim** | not meaningful (see below) | existing causal single-cache live-sim: one session, sequential, real physics |
| **Partial, off-sim** | v1 off-sim (stage 2-5 above): cheap, isolates privacy alone | next build target after v1: symbolic (no-physics) per-agent scheduler replay — cheapest place to validate WHO/WHEN coordination logic before paying for live-sim |
| **Partial, live-sim** | not meaningful (see below) | the stage 8/9 target: real per-agent loops, real environment, real physics |

**Live-sim is only meaningful paired with execution-driving.** Once physics
is real, the world after an agent's action is whatever its actual action
produced, not what the expert recorded; forcing an external ground-truth
WHO/WHEN schedule onto a world that has already diverged from the recording
makes the schedule incoherent the moment the model's action differs even
slightly from the expert's. Off-sim passive-scoring has no such problem
because the whole trajectory is teacher-forced against a fixed recording.
Drop the two live-sim/passive-scoring cells from the matrix; they are not a
useful ablation.

Recommended build order given the above: partial off-sim passive-scoring (v1,
already staged) -> partial off-sim execution-driving (new: validates the
per-agent-loop/FSM-legality scheduler cheaply, without live-sim cost) ->
partial live-sim execution-driving (stage 8/9, the real target). The
centralized execution-driving/off-sim cell is a low-priority footnote, not a
build target — it does not isolate a variable v2 does not already isolate.

## Corpus measurements behind the July 2026 revisions

All figures from the full local training corpus
(`/home/dorian/robocasa_local_train_subset_21`, 47 tasks), not the 162-trajectory
subset used earlier in this document.

**Observation lifetime.** Of targets that had a prior observation from the same
agent, the share where that observation was taken *immediately* before (i.e. no
other own-decision intervened):

| target type | n | immediately preceded by own `get_image` |
| --- | --- | --- |
| physical | 9,153 | **100.0%** |
| communicate | 5,730 | 85.8% |
| `get_image` | 18,306 | 25.2% |

**Staleness** (world mutated by either agent since the observation was taken):

| target type | stale |
| --- | --- |
| physical | **0.0%** |
| communicate | 34.3% |
| `get_image` | 70.0% |

**Who caused the staleness**, for `get_image` targets — the basis for rejecting
a freshness label: 50% own action (inferable), 30% fresh, **20% the other agent
only (invisible to this agent)**.

**View vocabulary.** Only three distinct view bundles exist across 20,280
`get_image` calls, and no bundle ever appears in two orderings:
`[wrist, agentview_center]` (62%), `[agentview_center, left, right]` (28%),
`[top_view, room_view, map]` (10%). List-order sensitivity in
`canonicalize_for_comparison` is therefore harmless for a trained model, though
`observation_view_set_exact_match` compares sets and is the safer metric.

## Partial-observability diagnostic results (zero-shot, no partial training)

Gemini-3 Flash, `heldout_tasks`, 1,097 samples (v1) / 2,649 (v3). No checkpoint
in this table was trained on a partial-observability prompt, so these are
floors and harness validation, not partial-observability results.

| condition | exact tool-call | tool-name | action exact |
| --- | --- | --- | --- |
| v1 partial, step index `global` | .323 | .622 | .505 |
| v1 partial, step index `local` | .325 | .639 | .509 |
| v1 partial, step index `none` | .328 | .620 | .514 |
| v3 partial (`consume-once`, `none`) | .146 | .422 | .492 |

Two findings worth carrying forward:

- **Removing the step-index leak costs nothing.** All three v1 modes fall
  within a 95% CI half-width of ~.028 of each other.
- **v3's aggregate is *deflated*, not inflated, zero-shot.** The prior
  expectation was that an easy 3-way view choice would inflate it. Instead
  `observation_recall` is .273 and `observation_view_set_exact_match` is .039:
  the observation protocol is a learned convention, not something inferable
  from the prompt. Crucially the physical columns barely move (action exact
  .505 -> .492, action tool-name .840 -> .832), so adding `get_image` targets
  does not degrade action prediction. Report per-tool breakdowns for v3;
  aggregates mix a convention-dependent subtask with the rest.

## What partial v3 changes relative to partial v1

"v3 adds a tool" understates it. Measured on three tasks of the training
corpus, holding `partial_history` and `consume-once` fixed:

| property | v1 | v3 |
| --- | --- | --- |
| examples | 802 | **1,964** (+145%) |
| image-free examples | 45 (6%) | **911 (46%)** |
| mean history length | 4.5 (max 13) | **9.2** (max 26) |
| own `get_image` lines visible in histories | **0** | 8,790 |
| tools offered | 5 | 6 |

Physical and communicate targets are **identical** between the two (same
144/126/84/80/42 physical and 326 communicate); v3 adds 1,162 `get_image`
targets on top. Consequences:

1. **Aggregate v1/v3 numbers are not comparable** — 59% of v3's denominator is
   a target type v1 does not contain. Compare the physical/communicate subsets,
   which are target-identical, to ask whether active observation costs anything.
2. **`get_image` steps enter the private history in v3 but not v1.** In v1 they
   hit the `continue` before the history append, so an agent never sees its own
   looks; in v3 they are supervised and therefore also accumulate.
3. **Image-free prompts go from an edge case to the normal state** (6% -> 46%).
   Under consume-once the absent image is the cue to look.
4. **The system prompt inverts.** The v1 prompt says "do not output get_image";
   v3 requires it. Hence `SYSTEM_PROMPT_ACTIVE_OBSERVATION_FIXED_AGENT` rather
   than a tool-list edit.
5. **Perception control moves from harness to model.** A v1 policy cannot drive
   a live loop without an external rule for when to render; a v3 policy is
   self-sufficient. This is the property that matters for live-sim.

## Observation-mode A/B (measured, zero-shot)

Identical manifest, model, and flags; only observation lifetime differs
(Gemini-3 Flash, partial v3, `heldout_tasks`, n=2,649):

| metric | `cache` | `consume-once` |
| --- | --- | --- |
| observation recall | .191 | **.273** |
| view-set exact match | .024 | **.039** |
| tool-name accuracy | .373 | **.422** |
| action exact (physical) | .495 | .492 |
| comm judged | .189 | .174 |

Observation recall improves 43% relative (n=1,552 `get_image` targets, CI
half-width ~.022). Mechanism: an **absent** image is itself the cue to look,
while a persistent cache always shows something (70% of it stale) and destroys
that cue. Action and communication columns are unchanged, so this is a targeted
gain rather than a trade. The comm-judged difference is within noise
(n=396, CI half-width ~.038), which is the first evidence that consume-once's
image-free communicates (14% of them) cost nothing measurable.

## Off-sim v3 vs live-sim: what transfers

`live_sim_eval.py` already implements an FSM mirror with incremental legality
checking, `validate_task_preconditions`, and a goal checker, with model
proposals passing the legality gate (oracle replay bypasses it). It has **no**
partial-observability support, no per-agent loops, and no resource locks.

| dimension | transfers to live-sim? |
| --- | --- |
| observation semantics | **yes** — consume-once maps exactly onto a live loop: request -> receive -> use -> discard |
| prompt shape / tool schema | yes — one builder feeds both |
| WHO / WHEN scheduling | **no** — off-sim inherits the expert ordering; live-sim needs the per-agent scheduler |
| legality rejection | **no** — see below |

**The rejection gap is a real train/serve mismatch.** Every step in the corpus
executed successfully, so off-sim never shows the model a refused call. Live-sim
*will* refuse calls, and `format_history_steps` renders those as a trailing
`FAILED: <reason>` line whose own docstring notes "training data never sets it".
So in live-sim the model meets a history line-shape it has never seen, exactly
when it most needs to recover. This affects centralized v3 equally — partial
observability did not introduce it — but it should be closed before any partial
live-sim run, either by synthesizing rejection examples or at minimum verifying
graceful degradation when a `FAILED:` line appears.

## Dataset roots and manifest construction

The two local roots are curated for different splits, and neither serves both:

| root | train tasks | held-out tasks | usable for |
| --- | --- | --- | --- |
| `eval_data_subset` | 46 tasks, **1-3 trajectories each** | 5 tasks, **15 each** | `heldout_tasks` (75 trajectories, matches exp52) |
| `robocasa_local_train_subset_21` | 47 tasks, ~21 each | **none** | `heldout_trajectories`; also the training root |

A `heldout_trajectories` manifest built from `eval_data_subset` yields only ~34
trajectories, because that split draws from the *train* tasks' 10% validation
holdout and the subset holds 1-3 trajectories per train task. It is not
comparable to exp52's 75-trajectory split. For a checkpoint trained on
`robocasa_local_train_subset_21`, build that split from the training root with
the same `--validation-split-seed 42` and `--validation-trajectory-fraction 0.1`
so the manifest selects exactly the trajectories training held out.

Note also that `eval_standalone` **silently skips** manifest rows with no
matching example (`examples_by_id.get(...)` then `continue`), so a manifest and
an eval whose flags disagree fail quietly rather than loudly.

## Evaluation cost

Off-sim eval on the `hf` backend runs ~11 minutes per 1,097-sample split
(Qwen3-VL-8B + LoRA, RTX 5090, `--batch-size 4`, 512px images). `eval_standalone`
supports only `hf` and `gemini`; `live_sim_eval` additionally supports a `vllm`
backend against a localhost OpenAI-compatible server. Porting that backend, or
simply raising `--batch-size`, is the available speedup once run counts grow.

## Result: partial v1 vs centralized v1 (trained, matched root)

Both adapters: Qwen3-VL-8B + LoRA r=16, `robocasa_local_train_subset_21`, same
47 train tasks, same box. Centralized 14,840 examples (no holdout); partial
12,702 (+2,134 validation). Native tool calls, greedy. The matched baseline
lands within noise of the previously-used cluster adapter on every metric, so
the training-data confound is ruled out.

| metric | centralized | partial | delta |
| --- | --- | --- | --- |
| **held-out trajectories** | | | |
| action exact (physical) | .986 | .978 | -.008 |
| comm judged | .726 | **.726** | **.000** |
| judged overall | .883 | .878 | -.005 |
| full trajectory (judged) | .160 | **.200** | **+.040** |
| **never-seen tasks** | | | |
| action exact (physical) | .675 | **.675** | **.000** |
| comm judged | .669 | .561 | **-.108** |
| judged overall | .673 | .634 | -.039 |
| full trajectory (judged) | .227 | .067 | -.160 |

**Privacy is free in-distribution; its cost is a generalization cost, and it
lives entirely in communication.** Physical action prediction is *exactly*
equal in both regimes. On familiar tasks the partial policy matches
centralized on communication to three decimals and completes more
trajectories.

### The qualifier that matters

Of the 78 communicate steps where centralized was judged correct and partial
was not, **39 (50%) had zero hidden information** — the partial agent's
history was identical to the centralized one at that step. By failure mode:

| | count |
| --- | --- |
| wrong content, same information | 37 (47%) |
| wrong content, information hidden | 24 (31%) |
| no communicate call, information hidden | 15 (19%) |
| no communicate call, same information | 2 (3%) |

So per-step information loss explains **at most half** the gap. The rest is
better described as a training-distribution effect than a per-step
information effect: the partial model trained on prompts where partner
context was *never* present, so it never learned to exploit it even at the
minority of steps where the two histories coincide (typically early, before
the partner has acted).

A confirmed instance of the information mechanism does exist
(`cluster_items_for_clearing/traj_000013/step_17`): agent_1 says it is going
to the table to **place** item2 when the expert says **pick up**, having not
seen agent_0's complete `navigate -> pick_up -> place_next_to -> give_space`
cycle. But it is not the dominant driver.

**The message-ablation experiment is what would separate these two
explanations causally**; the evidence above is correlational plus one
hand-inspected case.

### Metric caveats

- `judged_trajectory_all_steps_rate` carries ~1.4 points of LLM-judge noise:
  byte-identical predictions scored .067 and .053 on separate judge runs.
- The judge's `action_exact_call_accuracy` and structured eval's use different
  denominators under v3 (all non-communicate steps vs physical only), so the
  same 388 matches read as .172 or .492 depending on the source.

## Live-sim partial observability: implementation findings (July 2026)

The distributed contract is implemented in `live_sim_eval.py` as a separate
`run_trajectory_partial`, gated behind `--partial-history`. `run_trajectory` and
`_model_propose_step` are byte-identical to their pre-change versions, so the
centralized path carries no regression risk. Verified with the oracle backend
against the real simulator: **5/5 native success, 5/5 FSM goal, 0 harness
errors**, 27% of cycles genuinely simultaneous.

Three findings changed the design as written above.

### 1. Tool durations cannot be measured — the floors ARE the model

The section "Durations — measured, with escape hatches" overstated what is
available. Measurement yields **nothing**:

- `ToolResult` carries only `tool_name`, `success`, `details` — no step count.
- MuJoCo `sim.data.time` advances **~0** across a tool call. Of 51 executed
  steps, only 8 registered any delta at all, and every one rounded to 0.00 s.

The cause is not that settle logic is missing — `_settle_scene()` does loop
`sim.step()`. It is that (a) most tool paths call `sim.forward()`, which
recomputes kinematics **without advancing time**, and (b) where settling does
run it is `_SETTLE_STEPS = 2` or `_PLACEMENT_SETTLE_STEPS = 6` steps, which at
control_freq 20 is a few hundredths of a second — orders of magnitude below the
seconds-scale motion a real robot would take.

So the per-tool floors (`communicate`/`get_image` 0.25, manipulation 2.0,
navigation 4.0) are not a fallback; they are the only duration model available,
and the concurrency structure of every live-sim run is a consequence of those
chosen constants. State this whenever reporting makespan or idle time.

**TODO (not now, but worth checking):** whether the executor can be made to
free-run the physics for a short interval after each teleport. If it can,
durations become measured rather than assumed, and makespan becomes a physical
quantity instead of a bookkeeping one. `_settle_scene` is the natural hook.

### 2. "Reject both conflicting proposals" livelocks

The Resource conflicts section proposed rejecting both conflicting proposals as
the initial deterministic policy. It does not survive contact with a
deterministic policy: both agents propose `navigate_to_fixture` at the same
fixture, both are rejected, both sleep until the next world event, both wake and
propose **the identical call** again. Observed live: 6 conflicts, zero progress,
`makespan 0.0`, episode failed.

Symmetric rejection plus greedy decoding has no tie-breaker. Resolved with
`--conflict-priority {agent-order,seeded}`: one agent proceeds, the loser gets an
ordinary tool-call error and replans. This preserves the property that matters —
the harness never queues on the agent's behalf — while removing the livelock.
The same trajectory then reached `goal_satisfied` with 2 conflicts and a
makespan of 14.25 vs ~8.75 for uncontended ones, i.e. contention now costs time
instead of deadlocking.

### 3. Silent retry must not consume state

Under `--rejection-mode silent-retry` a rejected call leaves the agent untouched.
For the oracle backend that means a rejected expert step must be **pushed back
onto that agent's queue**; popping it consumed expert actions through rejections
and drained the episode into `policy_exhausted`. The general principle applies
to any policy with consumable state.

Non-model policies also need **per-agent expert queues**: an oracle replays a
joint sequence, but a per-agent scheduler asks a *specific* agent what it wants
to do, so "the next expert step" is only well defined per agent.

### Environment note

Live sim requires the `robocasa-live-sim` conda env (numpy 2.2.5, numba 0.61.2,
mujoco 3.3.1). The base env has numpy 2.4, which numba rejects, so the simulator
cannot import there at all — this is unrelated to partial observability and
affects centralized live-sim identically. `robocasa-vllm` carries vLLM 0.25.1 for
the ported off-sim vLLM backend.

## Open decisions

- ~~Whether stale pixels remain attached with an explicit freshness marker~~
  **RESOLVED (July 2026):** neither. `consume-once` makes every attached image
  fresh by construction; a freshness marker was rejected because it leaks the
  other agent's hidden activity (see Memory lifetime and freshness).
- Which events are public without communication (resolved for v1: only
  delivered `communicate` messages; see Decision summary #12 and Sequencing
  dependencies for how non-communicated sequencing is instead enforced by FSM
  legality rather than treated as public information).
- The resource granularity required to determine safe concurrency, and which
  preconditions the FSM must declare per tool for sequencing gating (see
  Sequencing dependencies and FSM legality gating).
- How to infer partial-order concurrency safely from existing sequential
  demonstrations.
- Whether local agent turn indices should appear in the prompt at all; the
  default recommendation is metadata only.
- Whether a persistent cache ever beats `consume-once` for message quality.
  The only cases that differ are communicate targets following another decision
  by the same agent (14% of communicates): `cache` shows them a possibly-stale
  image, `consume-once` shows none. The comm-judged delta between two otherwise
  identical runs is the experiment.
- Whether `standby` is ever needed at all: resolved that resource- and
  sequencing-blocked waiting do not need it (ordinary tool-call failure on the
  per-agent loop suffices); remains open only for a genuinely
  information-blocked case, which is unevidenced in the current data.
- Which events advance an agent's observable state version and which remain
  completely hidden.
- The zero-progress threshold and whether repeated calls (including calls that
  fail an FSM legality or resource precondition) are rejected, penalized, or
  only used as an evaluation termination criterion.

## Interpretation of the existing v3 checkpoint

The current v3 adapter remains useful evidence that the model learned
`get_image` and capable physical tools. Its fixed-eight alternating-protocol
result is diagnostic and is superseded for new smoke tests by the causal
single-cache protocol above. Neither protocol is a faithful evaluation of the
future distributed contract because the checkpoint itself was trained with
target-conditioned image presence. Full-split results for this checkpoint must
be labelled centralized fake-partial-observation diagnostics, not
partial-observability results. True partial live evaluation should wait until
partial v1 succeeds off-sim and active-observation training uses the causal,
agent-private state model defined here.
