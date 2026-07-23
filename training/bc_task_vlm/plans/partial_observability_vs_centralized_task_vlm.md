# Design: centralized Task-VLM vs distributed partial observability

Status: brainstorming design options; proposed follow-up to v3
active-observation SFT. The contracts below are not final until they are
implemented, smoke-tested, and compared experimentally.
Date: 2026-07-22

## Purpose

The current Task-VLM is centralized: one prompt contains the joint symbolic
history, the model chooses which agent acts next, and the dataset builder uses
the known target agent to select that agent's images. This is convenient for
teacher-forced SFT but is not causally reproducible in live inference.

The target architecture is instead a shared-weight distributed policy:

- one model and one model server;
- two logical agent sessions with separate private state;
- agent-owned `get_image` results, including global-camera results;
- explicit observation freshness independent of memory lifetime;
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
4. An agent's latest visual memory persists until replaced. Freshness is tracked
   separately and changes when the world changes.
5. Physical actions require a fresh, agent-owned observation with appropriate
   views. Communication may use stale private memory.
6. Both agents may request decisions independently whenever idle. Their
   requests are multiplexed or dynamically batched by one model server.
7. The episode runtime is a deterministic event scheduler, not an intelligent
   central policy and not a handoff protocol.
8. Model inputs should omit the global step number. Logs retain per-agent turn
   indices and shared world-event indices for traceability.
9. A causal centralized Task-VLM remains a valid baseline: it sees the complete
   joint prefix and both agents' prefix-derived caches, then predicts which
   agent acts. It must not use the current target to construct those inputs.

## Why the current centralized contract is insufficient

The current v3 example builder has four properties that conflict with a
distributed partial-observability interpretation:

1. Every `get_image` target is built with zero images, even when that agent
   already has a valid observation from an earlier request.
2. Every non-`get_image` target receives images selected using the known target
   agent. Live inference must choose its input before it knows the output agent.
3. Global-only requests are normalized to a placeholder agent, erasing which
   logical agent acquired the observation.
4. Every example sees the joint symbolic history, including the other agent's
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
    resource_locks
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

One-step image consumption is not recommended. It creates artificial amnesia
and conflicts with the observed reuse of caches for communication and later
image requests.

Instead, distinguish memory from action validity:

- The latest agent-owned observation remains private memory until replaced.
- It is fresh while its captured world version matches the relevant current
  world version.
- A physical state change marks affected observations stale.
- The model may still receive stale private memory if it is explicitly labelled
  stale.
- The legality gate must reject a physical action that lacks a sufficiently
  fresh, agent-owned observation.
- Communication may proceed with stale visual memory.

The input must state:

```text
Observation owner: agent_0
Views: wrist, agentview_center
Captured world version: 18
Current world version: 20
Fresh: false
```

Attaching stale images without freshness metadata is unsafe because the model
may treat them as the current scene. Dropping all stale pixels is a simpler
initial option, but it weakens visual memory. Whichever policy is chosen must
be deterministic and independent of the current target tool.

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

The episode runtime uses an event loop:

```text
while task is not terminal:
    identify idle, unblocked agents
    snapshot the world state visible at this decision boundary
    submit all ready agent requests concurrently
    validate proposals against the same snapshot
    resolve resource conflicts deterministically
    schedule compatible tools
    advance simulation events
    deliver tool results and messages to their owners
    update world version, freshness, and success checks
```

This runtime coordinates clocks and resources; it does not choose semantic
actions for either agent.

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

Proposals are validated against the same pre-batch state. Disjoint proposals
may proceed together. Conflicting proposals must not gain an arbitrary hidden
advantage from Python call order.

Initial deterministic policy:

- reject both conflicting proposals with the same structured conflict result;
- let both agents replan from the updated inbox and history;
- record the conflict as a coordination error, not a harness error.

A seeded priority or reservation protocol may be added later, but must be
visible to the agents and constant across compared runs.

## Standby, progress, and deadlock (brainstorming)

An agent sometimes has no useful action while the other agent continues
working. A durable `standby` or `wait` action is therefore a candidate part of
the protocol. It is not `yield_control`:

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
2. **Partial-observability v1 dataset:** construct one example for the expert
   acting agent with caller identity fixed, that agent's own cached image,
   private action history, sent messages, received messages, and explicitly
   public events. Do not predict the acting agent and do not train `get_image`.
3. **Partial-observability v1 off-sim training:** train a fresh adapter at the
   same scale as centralized v1. This isolates history/image privacy from active
   observation, self-scheduling, live execution, and concurrency.
4. **Matched off-sim evaluation:** evaluate centralized v1 and partial v1 on
   the same expert targets and splits. Report per-step accuracy, judged
   communication, trajectory-all-correct rate, and correct-prefix fraction.
5. **Off-sim communication diagnostics:** run the message-content and delivery
   ablations where meaningful. Inspect whether private agents use messages to
   predict actions that the centralized policy can infer directly from joint
   history.
6. **Causal active-observation dataset:** only after partial v1 works, add
   `get_image` as a target using prefix-derived private caches. Train partial
   v3 off-sim and verify request ownership, view choice, and cache freshness.
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

## Open decisions

- Whether stale pixels remain attached with an explicit freshness marker or are
  replaced by metadata-only memory in the first implementation.
- Which events are public without communication.
- The resource granularity required to determine safe concurrency.
- Whether equal-time conflicts reject both proposals or use a visible seeded
  reservation policy.
- How to infer partial-order concurrency safely from existing sequential
  demonstrations.
- Whether local agent turn indices should appear in the prompt at all; the
  default recommendation is metadata only.
- Whether to add `standby`, which wake predicates it supports, and whether
  agent-owned timers are causally appropriate.
- Which events advance an agent's observable state version and which remain
  completely hidden.
- The zero-progress threshold and whether repeated calls are rejected,
  penalized, or only used as an evaluation termination criterion.

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
