# Communication Evaluation

## Objective

Evaluate agent communication at three distinct levels:

1. **Message quality:** Is the content grounded, relevant, specific, timely, and consistent?
2. **Receiver utility:** Does the receiver incorporate the information and coordinate more effectively?
3. **Causal value:** Does communication improve task outcomes relative to controlled interventions?

Task success alone is insufficient. A policy can succeed while sending irrelevant messages, or fail despite communicating correctly because manipulation fails.

## Required instrumentation

For every simulator step, record:

- Episode ID, task, seed, and global step.
- Acting agent and receiver, when applicable.
- Complete simulator state for evaluator-only use.
- Each agent's observation and message history.
- Objects and attributes visible to each agent.
- Tool calls, arguments, results, and action failures.
- Message sender, receiver, text, send step, and delivery step.
- Task predicates before and after each action.
- Episode success, termination reason, and total steps.
- Model probabilities or sampling metadata, when available.

The omniscient simulator state must remain evaluator-only; policies continue to receive their normal partial observations.

Create one communication-event record per message containing the sender knowledge state, receiver knowledge state, parsed message content, subsequent receiver actions, and episode outcome.

## Message representation

Parse messages into structured factual claims, requests, warnings, and plans. For example:

```json
{
  "intent": "state_report",
  "subject": "red_cup",
  "relation": "inside",
  "object": "cabinet",
  "certainty": "asserted"
}
```

Requests should additionally compile into a simulator-checkable completion predicate:

```json
{
  "intent": "request",
  "action": "place",
  "object": "red_cup",
  "destination": "counter",
  "completion_predicate": "on(red_cup, counter)"
}
```

Entity resolution must track the same object across messages, observations, and simulator states.

## Metrics

### Groundedness and factual accuracy

**Question:** Is each factual claim supported by information available to the sender?

For every parsed proposition, check whether the sender had observed supporting evidence, whether that evidence was current at send time, and whether contrary evidence was available.

\[
\text{groundedness} =
\frac{\text{supported factual claims}}
{\text{all factual claims}}
\]

Report separately:

- Supported claims.
- Unsupported claims.
- Claims unverifiable from the sender's history.
- Claims contradicted by sender-visible evidence.

Also compute factual accuracy against the omniscient simulator state. Groundedness and accuracy are different: a lucky guess can be accurate but ungrounded.

**Required factors:** sender observation history, observation timestamps, object identity resolution, proposition parser, and simulator-state predicate checker.

### Relevance

**Question:** Can the information affect the receiver's current task decision?

A proposition is relevant when it concerns an unsatisfied goal predicate, an object required by one, an active subtask or plan, a resource conflict, a recent failure, or information that eliminates a plausible receiver action.

Score relevance using distance to the remaining task graph:

- Directly affects the receiver's next required decision: high.
- Affects a later unresolved subtask: medium.
- Describes an already satisfied or unrelated predicate: low or zero.

**Required factors:** current goal predicates, satisfied predicates, task dependency graph, object-to-goal mapping, agent subtask assignments, and parsed message entities/intents.

### Specificity and actionability

**Question:** Does the message provide enough information for the receiver to identify the intended update or action?

Check intent-dependent slots:

- Request: action, object, and destination when required.
- State report: object and relevant state or location.
- Warning: conflict or hazard and affected action/object.
- Plan: responsible agent and intended action/subtask.

\[
\text{specificity} =
\frac{\text{required slots supplied}}
{\text{required slots for the message intent}}
\]

Resolve descriptions relative to context. “The cup” is specific when only one cup is plausible and ambiguous when several are.

**Required factors:** intent classifier, entity/relation extraction, intent-specific required slots, receiver context, and ambiguity checks over candidate objects.

### Timeliness

**Question:** Does the information arrive before the receiver commits to a costly or conflicting action?

Track:

- `information_birth_step`: when the sender first learned the information.
- `delivery_step`: when the receiver received it.
- `decision_step`: the first receiver action affected by it.
- `deadline_step`: the first irreversible or costly conflicting action.

\[
\text{delivery latency} = \text{delivery step} - \text{information birth step}
\]

\[
\text{uptake latency} = \text{decision step} - \text{delivery step}
\]

A message is timely when it arrives before its deadline. Report timeliness separately from uptake because a timely message can still be ignored.

**Required factors:** observation and delivery timestamps, receiver action trace, message-consistent/conflicting action definitions, task transitions, and irreversible-action labels.

### Novelty, redundancy, and information gain

**Question:** Does the message tell the receiver something it did not already know?

Classify each supported proposition as directly observed already, previously communicated, inferable from shared task state, new, or a useful confirmation of uncertain state.

\[
\text{novelty} =
\frac{\text{new supported propositions}}
{\text{supported propositions}}
\]

Where a receiver belief model is available, measure task-relevant information gain:

\[
IG(m) = H(Z \mid h_r) - H(Z \mid h_r, m)
\]

Here, \(Z\) is a relevant hidden variable and \(h_r\) is receiver history. A practical proxy is the number of viable task-relevant hypotheses or candidate actions eliminated by the message.

**Required factors:** receiver observation/message history, structured belief state, duplicate-proposition matching, task planner or candidate-action generator, and confidence estimates when available.

### Consistency and staleness

**Question:** Does the message contradict reliable prior information or current state?

Compare each proposition with sender history, receiver observations, established shared facts, and evaluator-only simulator state.

\[
\text{contradiction rate} =
\frac{\text{contradictory propositions}}
{\text{all factual propositions}}
\]

Classify self-contradictions, cross-agent contradictions, stale claims, incorrect current-state claims, and plan conflicts separately. Apply temporal validity rules so a formerly correct statement that has become obsolete is labeled stale rather than inherently false.

**Required factors:** canonical propositions, entity tracking, temporal validity rules, and the source/timestamp of each belief.

### Message uptake

**Question:** Does receiver behavior show incorporation of the message?

For each actionable message, generate message-consistent and message-conflicting actions and inspect the receiver's next \(k\) turns.

\[
\text{uptake@k} =
\frac{\text{actionable messages followed by a consistent action within }k\text{ turns}}
{\text{actionable relevant messages}}
\]

Use receiver turns rather than global steps. Avoid assigning uptake labels to messages that require no observable response.

**Required factors:** intent/request extraction, action-semantic parser, object matching, consistent/conflicting action sets, and receiver-local turn indices.

### Request completion

**Question:** Are communicated requests fulfilled?

Compile each valid request into a simulator predicate and track acknowledgment, first attempt, completion, cancellation, supersession, and episode termination.

\[
\text{request completion rate} =
\frac{\text{completed valid requests}}
{\text{valid actionable requests}}
\]

Also report time to acknowledgment, first attempt, and completion; failure reason; and whether the sender ultimately completed the request itself.

**Required factors:** request parser, predicate evaluator, deadlines, action-agent attribution, and cancellation/supersession handling.

### Duplicate work

**Question:** Does communication prevent agents from unnecessarily pursuing the same exclusive subtask?

Map actions to objects, subtasks, and goal predicates. Mark overlapping work as duplicate only when one contribution was sufficient. Collaborative subtasks must not be penalized.

\[
\text{duplicate-work rate} =
\frac{\text{redundant subtask-action steps}}
{\text{total productive-action steps}}
\]

Examples include both agents searching for the same already-located object, both attempting the same placement, or repeating an already completed action.

**Required factors:** action-to-subtask mapping, exclusive/collaborative subtask definitions, predicate ownership over time, object interaction history, and matched communication ablations.

### Conflict reduction

**Question:** Does communication prevent incompatible actions?

Detect agents targeting the same exclusive object, blocking a required region, undoing satisfied predicates, moving objects relied upon by the other agent, following mutually incompatible plans, or repeatedly failing due to contention.

Report conflicts per episode and recovery latency from conflict onset until productive non-conflicting behavior resumes.

**Required factors:** object ownership/contact state, spatial blocking/collision events, predicate-regression detection, action failures, and planned or inferred subtasks.

### Coordination latency

**Question:** How quickly does shared information become coordinated progress?

Measure separate intervals:

- Discovery to message.
- Message to acknowledgment or plan change.
- Message to first relevant action.
- Message to predicate completion.
- Discovery to predicate completion.

**Required factors:** discovery-event detection, message references, receiver action attribution, and predicate-completion timestamps.

### Communication efficiency

**Question:** How much useful coordination is achieved for its communication cost?

Track messages, tokens, communication tool calls, simulator steps spent communicating, and waiting delays. Candidate metrics include useful messages per episode, success per 100 message tokens, and communication precision:

\[
\text{communication precision} =
\frac{\text{grounded, relevant messages with uptake}}
{\text{all messages}}
\]

Do not optimize message count alone: silence is inexpensive but may destroy coordination.

**Required factors:** token counts, tool-call timing, step cost, task outcomes, and a preregistered definition of a useful message.

## Causal evaluation

The strongest evaluation reruns matched task-seed pairs from the same initial simulator state under controlled message interventions:

- **Normal:** deliver the original policy messages.
- **Suppressed:** prevent message delivery.
- **Delayed:** deliver messages after a fixed number of receiver turns.
- **Scrambled:** deliver a plausible but episode-irrelevant message.
- **Oracle:** optionally substitute concise, correct task-relevant content.

Measure paired changes in success, completion steps, task predicates achieved, conflicts, duplicate work, and recovery behavior:

\[
\Delta_{success} =
\frac{1}{N}\sum_i
\left(success_i^{normal} - success_i^{ablated}\right)
\]

Use paired confidence intervals or a paired bootstrap over task-seed pairs. Deleting a message from an existing trajectory is not a valid causal replay because subsequent policy behavior diverges; reset and rerun the simulator under each intervention.

Control task, initial state, simulator seed, checkpoint, decoding parameters, and sampling configuration. Repeat stochastic conditions enough times to distinguish intervention effects from policy sampling variance.

## Recommended first implementation

Start with metrics that are interpretable and can be checked reliably:

1. Grounded factual-claim rate.
2. Actionable request rate.
3. Request completion rate.
4. Uptake within three receiver turns.
5. Message-to-action latency.
6. Duplicate-subtask rate.
7. Contradiction and staleness rate.
8. Paired success and completion-step changes under message suppression.

This requires four primary evaluator components:

1. Structured simulator event logging.
2. A message parser producing claims, requests, entities, and predicates.
3. A deterministic predicate checker over simulator state.
4. Matched normal-versus-suppressed live-sim execution.

This initial suite measures both whether messages appear good and whether they actually improve coordination.
