# RoboTalk: project and implementation reference for paper writing

**Inspection date:** September 14, 2026.  
**Repository HEAD:** `af38f5fbd048a57f3506480ee915c3ef8a0431f3`.  
**Scope:** the working tree, canonical RoboTalk data, stored training configurations, and the existing 43/10 evaluation artifact. The working tree contains uncommitted changes, so HEAD alone does not reproduce this snapshot. This document does not modify experiments, regenerate data, or certify all historical runs against today's code.

This is a technical reference for writing the paper, not a draft manuscript. It separates implemented mechanisms, measured results, intended experimental contrasts, and unresolved discrepancies. Repository links below are relative to this document. Files in external cluster directories are identified by absolute path. Counts labeled “scanned” were recomputed from the stored source trajectories during this inspection; evaluation numbers were read from the existing artifact rather than rerun.

## 1. Project in one paragraph

RoboTalk studies high-level coordination between two embodied agents pursuing a shared household task in RoboCasa. Each agent is invoked independently with its identity, a natural-language goal, symbolic initial state, private execution history, received messages, and requested visual observations. Agents choose symbolic tools for navigation, manipulation, observation, and communication. A concurrent finite-state validator governs legal actions, resource access, waiting, and goal satisfaction; a simulation executor grounds those tools into a two-robot MuJoCo scene. The project includes a validated demonstration-generation pipeline, a 7,950-trajectory dataset, supervised fine-tuning of Qwen3-VL-8B variants, fixed-cohort live simulation evaluations, communication-guidance ablations, and an iterative expert-correction pipeline inspired by DAgger.

### 1.1 What the project evaluates

- Whether models can complete tasks with decentralized decision making and agent-specific information.
- How communication guidance and an explicit coordinator–follower opening protocol affect success.
- How demonstration scale affects performance on task types represented and absent during training.
- How Instruct versus Thinking base models and training with rationale traces affect performance.
- Whether correcting learner-generated trajectories improves a previously trained policy.

The central object of study is **coordination**, rather than separately measured competence at each constituent skill. Resource handovers, consistent work division, state tracking, and temporal dependencies are concrete mechanisms through which coordination can succeed or fail.

### 1.2 Important boundaries for the paper

This is high-level planning and tool use, not end-to-end motor-control learning. Navigation and manipulation are executed through scripted simulation tools, including robot/object repositioning. Concurrent decisions are logically synchronized, while physical simulation operations are executed through an ordered tool executor. Neither tick count nor logical makespan is automatically a real-robot execution time.

Partial observability does **not** mean that the agents have no initial state description or can see only an egocentric camera. They receive symbolic initial state, and `get_image` permits global views. The restriction is primarily separate evolving histories and observations: agents do not automatically receive their partner's private calls, tool results, waits, or rationale traces.

Useful paper wording: “two VLM-based agents with shared model weights and separate contexts,” “symbolic manipulation tools grounded in live simulation,” and “error-free FSM success.” Avoid claiming independent learned policies, continuous simultaneous manipulation, or native physical success when reporting an FSM metric.

## 2. Implementation map

| Area | Main implementation | Responsibility |
|---|---|---|
| Task definitions | [`tasks/specs/verified`](../data_generation/task_level/tasks/specs/verified), [`tasks/specs/runtime.py`](../data_generation/task_level/tasks/specs/runtime.py) | Goals, symbolic entities, initial states, preconditions, effects, and terminal conditions |
| Shared tool interface | [`subatomic_tool_specs.py`](../data_generation/task_level/subatomic_tool_specs.py), [`subatomic_tool_calls.py`](../data_generation/task_level/subatomic_tool_calls.py) | Tool names, argument types, descriptions, optional fields, model-facing schemas |
| State/configuration sampling | [`instances.py`](../data_generation/task_level/tasks/shared/instances.py), [`structured_random.py`](../data_generation/task_level/sampling/structured_random.py) | Initial locations, access states, reproducible generation variation |
| Demonstration prompt | [`tasks/shared/prompting.py`](../data_generation/task_level/tasks/shared/prompting.py) | Full tick-trajectory generation instructions and task-specific facts |
| Symbolic transitions | [`fsm.py`](../data_generation/task_level/tasks/shared/fsm.py) | Generic navigation/manipulation preconditions and state effects |
| Concurrent semantics | [`concurrent_fsm.py`](../data_generation/task_level/tasks/shared/concurrent_fsm.py), [`scheduling.py`](../data_generation/task_level/tasks/shared/scheduling.py) | Atomic ticks, contention, frozen preconditions, eligibility, waits, release delivery |
| Workspace relationships | [`workspace_semantics.py`](../data_generation/task_level/tasks/shared/workspace_semantics.py) | Cabinet aliases, parent access, exclusive-child location preservation |
| Demonstration retry cascade | [`production_cascade.py`](../data_generation/task_level/generation/raw/production_cascade.py), [`cascade_canary.py`](../data_generation/task_level/generation/raw/cascade_canary.py) | Per-task generation, bounded escalation, cumulative errors, attempt metadata |
| Observation insertion | [`generation/image/processor.py`](../data_generation/task_level/generation/image/processor.py) | Add observation calls and preserve action-tick semantics |
| Grounding/simulation | [`grounding_specs.py`](../data_generation/task_level/grounding_specs.py), [`sim_tool_executor.py`](../robocasa/utils/sim_tool_executor.py), [`trajectory_runner.py`](../robocasa/utils/trajectory_runner.py) | Resolve symbolic IDs, initialize robots/objects, execute tools and render views |
| Scene selection | [`scene_sampling.py`](../data_generation/task_level/scene_sampling.py), [`build_scene_compatibility_cache.py`](../training/bc_task_vlm/build_scene_compatibility_cache.py) | Reproducible selection from audited scene/configuration combinations |
| Shared policy prompt | [`training/prompting.py`](../training/bc_task_vlm/prompting.py) | SFT and default evaluation context construction |
| Training data | [`dataset.py`](../training/bc_task_vlm/dataset.py), [`preprocess.py`](../training/bc_task_vlm/preprocess.py), [`preprocessed_data.py`](../training/bc_task_vlm/preprocessed_data.py) | Agent-turn examples, images, chat templates, tokenization, labels |
| Training | [`main.py`](../training/bc_task_vlm/main.py), [`training/scripts`](../training/scripts) | LoRA, distributed training, checkpoints, experiment metadata |
| Evaluation | [`live_sim_eval.py`](../training/bc_task_vlm/live_sim_eval.py), [`live_sim_parallel_eval.py`](../training/bc_task_vlm/live_sim_parallel_eval.py) | Independent agent generation, simulation execution, rejection feedback, records |
| Evaluation populations | [`fixed_cohort_selection.py`](../training/bc_task_vlm/fixed_cohort_selection.py), [`freeze_configuration_cohort.py`](../training/bc_task_vlm/freeze_configuration_cohort.py) | Duplicate-free configurations and repeatable episode selection |
| Metrics and figures | [`summarize_fixed_live_sim.py`](../training/bc_task_vlm/summarize_fixed_live_sim.py), [`build_43_10_eval_artifact.py`](../training/bc_task_vlm/build_43_10_eval_artifact.py) | Success, errors, participation, confidence intervals, comparison artifacts |
| Corrected demonstrations | [`dagger`](../training/bc_task_vlm/dagger), [`dagger/data_generation`](../training/bc_task_vlm/dagger/data_generation) | Diagnose, correct, resume, materialize, mix with original data, train |
| Dataset publication | [`robotalk_hf/export_robotalk.py`](../training/bc_task_vlm/robotalk_hf/export_robotalk.py), [`robotalk_hf/static_space`](../training/bc_task_vlm/robotalk_hf/static_space) | Importable tables, media archives, interactive trajectory explorer |

The generic top-level training README still describes centralized prediction and much older defaults. Do not use it as the current experimental specification. Similarly, comments mentioning tool-dependent evaluation durations can be older than launchers that explicitly select uniform durations.

### 2.1 Task-specification preparation upstream of trajectory generation

The [`task-level pipeline`](../data_generation/task_level/pipeline/README.md) supports static candidate filtering, LLM transferability scoring, task-specification generation, static/FSM validation, optional LLM review, trajectory generation, simulator sweeps, and failure aggregation. Simulator-aware normalization can resolve fixture parts, controls, and support sites while building a specification; this is distinct from initializing a simulator for every later symbolic state sample.

The current phase-2 implementation retains a flat example replay as a legacy diagnostic and optionally gates specifications through a newly generated concurrent canary. A passed legacy example alone is not a concurrency certificate. The 53 verified JSON specifications are the task inventory used here; other registered or experimental task specifications are not automatically members of RoboTalk. Sibling low-level policy backends and newer RL directories are separate experiments, not evidence for the SFT/communication results in this document.

## 3. Task representation and symbolic interface

### 3.1 A task specification

A verified JSON task specification contains a composite-task identifier and natural-language `task_goal`, symbolic initial state, tool constraints, task-specific preconditions/effects, and goal conditions. Symbolic state distinguishes agents, movable objects, fixtures, and task-machine state. An agent has a symbolic location and at most one held object. An object can be located on a fixture or within/on another object. Fixtures can expose parts and controls.

Illustrative state fragment using the actual vocabulary:

```json
{
  "agents": {
    "agent_0": {"location": "fridge", "held_object": null},
    "agent_1": {"location": "counter", "held_object": null}
  },
  "objects": {
    "bowl": {"object_type": "bowl", "location": "counter"},
    "meat2": {"object_type": "steak", "location": "pan"}
  },
  "fixtures": {
    "counter": {"fixture_type": "counter"},
    "fridge": {
      "fixture_type": "fridge",
      "parts": {"door": {"part_type": "hinged_part", "state": "closed"}}
    }
  }
}
```

This is a fragment, not a complete runnable task. The model copies dictionary keys: `door` is the `part_id`; `hinged_part` describes its type. Likewise a `control_id` refers to a key under a fixture's `controls`, not an invented name inferred from a photograph. Object containment is separate from physical support: a steak may be in a pan that is on a stove.

### 3.2 Global policy tools versus task-specific expert constraints

`build_model_tool_specs` constructs a global, task-agnostic interface. Its function descriptions explain tool meaning and required/optional arguments. It does not publish each task's allowed-value lists as hints about which particular IDs solve that task. For partial-history policies, the scheduler fixes the acting agent; ordinary function arguments do not ask the model to select an agent.

Demonstration generation additionally receives task-specific constraints, because it must produce an entire legal expert sequence. These include preconditions/effects and task-specific facts. That is deliberately a richer expert prompt than the policy prompt. The policy still receives the natural-language task goal, symbolic initial state, general mechanics, and state-derived concurrency facts.

| Tool family | Examples | Meaning |
|---|---|---|
| Navigation | `navigate_to_fixture(fixture_id)` | Move to a symbolic fixture/workspace |
| Acquisition | `pick_up_object(object_id, source_id)` | Pick one object from its current source, with an empty hand |
| Fixture-surface placement | `place_on_surface(object_id, support_id)` | Put an object on a fixture surface |
| Movable-support placement | `place_on_object(object_id, support_object_id)` | Put an object on a plate, tray, or other movable support |
| Containment | `place_in_receptacle(object_id, receptacle_id)` | Put an object inside a receptacle |
| Relative placement | `place_next_to(object_id, reference_object_id=...)` or `reference_fixture_id=...` | Place beside a reference on the resolved supporting surface |
| Appliance-relative placement | `place_under(object_id, reference_fixture_id)` | E.g. mug under a coffee dispenser |
| Parts/controls | `open_hinged_part`, `close_hinged_part`, sliding-part tools, `press_button`, `press_lever`, `set_rotary_control` | Change an exposed part/control with applicable preconditions |
| Communication | `communicate(to, message, releases?, coordination_phase?)` | Send text; optionally carry a release or protocol phase |
| Waiting | `wait_for_signal(from, about)` | Block until a later matching release from that sender |
| Physical handover | `give_space(fixture_id)` | Vacate a workspace; does not itself transmit a release |
| Observation | `get_image(views)` | Request valid named views for the caller |

`source_site_id` and `target_site_id` are optional sub-sites, not alternate names for fixture IDs. The shared descriptions say to omit them when no named sub-site matters. Multi-slot future tasks need explicitly represented sites and unambiguous grounding; do not assume that every new appliance can safely use the present coarse abstraction.

### 3.3 Placement distinctions that matter

`place_on_object(strawberry, cake_plate)` places on the plate. `place_next_to(strawberry, cake)` identifies the cake as a spatial reference and uses the resolved surrounding support; it must not be described as necessarily placing on `cake_plate`. This ambiguity motivated changes to the GarnishCake instruction and shared placement guidance.

For fixture references, the current interface permits navigation either to the reference fixture or its declared adjacent placement location before `place_next_to`. Navigation and placement must resolve the same supporting fixture; arbitrary selection of a generic “counter” was a previous grounding failure. Source: the `place_next_to` interface extension and reference-fixture probe scripts.

`give_space` remains permitted on shared spaces when applicable; guidance discourages using it as filler. It has no `releases` argument. Only `communicate` transmits a release. The current model-facing `releases` argument is a single symbolic ID; tolerant legacy normalization in lower-level scheduling is not a reason to advertise arbitrary string/array interchangeability.

## 4. Concurrent execution, communication, and observations

### 4.1 What a tick means

A canonical tick specifies one non-observation call for each runnable agent. A blocked agent has no executable call. Generation's explicit `{"state":"blocked"}` marker records this condition; it cannot create blocking. The normalized raw data stores the canonical executable grid in `tick_rows` and a serialization in `steps`.

The live evaluator obtains both agents' proposals from their pre-cycle private contexts. It checks pairwise contention and each proposal's ordinary preconditions against copies of the same beginning-of-cycle symbolic state, then executes accepted calls and commits effects. Therefore an agent cannot rely on its partner opening a door earlier in the same tick. Physical tool execution is ordered, even though the decision and validation model is concurrent.

`ConcurrentScheduler` shares agent eligibility, wait state, readiness, and release semantics between offline and live paths. `ConcurrentTaskValidator` shares contention and transition validation. The offline replay and live simulator are not literally one program loop: one consumes recorded calls; the other queries models and calls the simulator. `FsmMirror` is the live bridge to the shared concurrent validator.

### 4.2 Opening protocol

One coordinator is sampled per episode and shown to each agent. Both roles use the same model weights.

| Opening tick | Coordinator | Follower |
|---|---|---|
| 0 | `communicate(..., coordination_phase="propose")`: concrete division using symbolic IDs | `communicate(..., coordination_phase="await_plan")` |
| 1 | `communicate(..., coordination_phase="await_confirmation")` | `communicate(..., coordination_phase="confirm")` |
| 2 onward | Physical work may begin | Physical work may begin |

The follower's first message cannot depend on the coordinator's same-tick proposal. Messages become available after the concurrent tick. Optional observation calls do not advance the two communication phases. The protocol constrains message phases and ordering; it does not prove that the natural-language plan is optimal, truthful, or semantically consistent with every later action.

### 4.3 Waiting and releasing

`wait_for_signal(from="agent_1", about="fridge")` is private. The partner sees the preceding communication request, not the wait tool call. A natural-language request must name the release keyword, so the partner can use it. The generation guidance calls for the request immediately before the wait.

Blocking requires an actual wait call. A promise to wait or a `portion_complete` message alone does not suspend an agent. A matching release is a later `communicate` call from the specified sender with the matching `releases` value. An arbitrary message does not wake the waiter.

The waiter remains blocked during the release tick and resumes on the following tick. Physical vacancy and message release are distinct. `give_space(fridge)` frees the workspace after the call's tick; an unblocked agent may enter next tick without a release. An already blocked agent still requires its matching message.

Illustrative timeline; assume agent 1 occupies the fridge and agent 0 is elsewhere:

| Tick | Agent 0 | Agent 1 |
|---|---|---|
| N | Communicate “When done, release fridge.” | Finish a valid fridge action |
| N+1 | Wait for agent 1, about fridge | Give space from fridge |
| N+2 | Blocked | Communicate with `releases="fridge"` |
| N+3 | Navigate to fridge | Another valid call, or a previously justified wait |

This is an explanatory fragment, not a proposed complete demonstration. In particular, every unblocked agent still needs a legal call, and a successful final tick needs no artificial tail merely to add waits.

### 4.4 Task completion

The production evaluator uses FSM success as its stopping criterion. Agents do not decide global completion by saying “done.” An agent that finishes its own portion reports that fact, names a release keyword, and waits on its next turn if the episode continues. It does not become unblocked because the other agent satisfies the goal: the episode ends instead.

The current live loop terminates mutual waits as deadlock. It does not automatically clear waits to manufacture recovery. Historical evaluations predate this correction, so a run's code version matters when interpreting error-free success or deadlock recovery.

### 4.5 Resource semantics

The shared classifier treats a manipulated object as exclusive. A stationary source, plate, tray, or other supporting object can be read/shared while agents manipulate different objects. Moving that support while a partner uses it is a conflict. Thus two different skewers can be picked from `skewer_plate`, but one agent cannot move `skewer_plate` while the other picks from it.

Counters, islands, and tables are broad shared workspaces under the symbolic model. Cabinet and parent counter locations are aliases for a shared workspace. Agents may access different cabinet contents concurrently, but a cabinet door transition conflicts with another agent's access to that cabinet in the same tick.

The symbolic exclusive fixture types currently include drawers, fridge, microwave, oven, dishwasher, stove, sink, toaster/toaster oven, coffee machine, blender, stand mixer, and electric kettle. **Symbolic exclusivity and physical front-facing placement are not identical classifications.** Cabinets were removed from the exclusive category; some appliances remain exclusive even where their physical candidate-placement path does not use a generic `require_front=True` flag.

An agent at an exclusive child appliance can interact with its parent counter and supported items while retaining its child location and physical pose. It continues to occupy the child workspace. Cabinet navigation is kept as a backward-compatible alias to the parent workspace; newly sampled initial locations canonicalize cabinets to their parent counters.

### 4.6 Observation policy and information available to a model

Valid views are `top_view`, `room_view`, `map`, `wrist`, `agentview_center`, `agentview_left`, and `agentview_right`. Camera names are exact identifiers. The map has a faster raster rendering path selected by fixed-cohort launchers.

Observation calls are agent-local and transparent to physical/coordination tick ordering. They can cause another model invocation with images without advancing a handover. `observation_transparent_tick_rows` recovers the action grid after observation insertion.

Production SFT uses consume-once observations: pending images enter the next applicable invocation and are consumed. It does not append an unlimited collection of past image tensors. Textual history persists according to the private-history rules. The explorer's “latest available image” display is a visualization convenience and should not be confused with images actually supplied to every later policy invocation.

Observation insertion defines initial global views, navigation-related left/center/right views, and action-related wrist/center views. It also provides final observations. These observation requests are added programmatically; they are not all independently chosen by the demonstration LLM. Rationale text for inserted image calls is templated by the observation processor.

## 5. Demonstration generation and configuration diversity

### 5.1 Configuration sampling

Initial symbolic states are built from task specifications without initializing MuJoCo for every LLM request. Sampling enumerates valid joint agent locations and eligible open/closed access-part states. It excludes simultaneous occupancy of the same exclusive starting fixture and canonicalizes cabinet locations to parent workspaces.

Access-state eligibility is derived from storage-fixture type, an exposed part already represented as open/closed, and compatible opening-tool targets/parts. It is not arbitrary randomization of appliance activation. Turning on a coffee machine is not interchangeable with opening a cabinet because activation can have task prerequisites.

The sampler cycles over the Cartesian product of location and access assignments. For N requested trajectories and K configurations, requested exposure is floor(N/K) or ceil(N/K). This is a scheduling guarantee, not a proof that generation succeeds equally often for every configuration. Exhausted generation can leave holes; it must remain visible rather than silently replacing difficult configurations with easy ones.

The source dataset contains **384 distinct physical configuration signatures across 53 tasks**. Adding two coordinator identities gives **768 cohort configurations** in the frozen configuration manifest. These counts do not include all layouts/styles/seeds or all possible continuous object poses. Several tasks have only one physical symbolic configuration; HotDogSetup has 60. The task catalogue below lists observed counts.

### 5.2 Structured-random generation

The current production cascade sets `sampling="structured_random"`, `partition_policy="none"`, `tick_format=True`, and `prompt_style="simplified_v3"`. The structured variables are:

- `communication_message_length`: low, medium, high;
- `communication_message_complexity`: low, medium, high;
- `tool_call_diversity`: low, medium, high.

They influence stylistic/behavioral variation, with guidance against unnecessary actions. They are not sampled object-ownership partitions. Older explicit work partitions and balanced-local partition experiments exist in the repository but are not the canonical production setting.

The structured seed hashes task identity and the complete run/attempt variation key. Retries can therefore change generation variation reproducibly while preserving the requested physical configuration. This is separate from a decoder's random seed and from the simulator scene seed.

### 5.3 Current demonstration prompt

The simplified prompt requests a full trajectory and organizes its main instructions into eight rules: correct invocation/blocking representation; two-tick opening protocol; efficient work division; valid locations and hands; atomic contention/handover semantics; wait/release requests; valid termination; and exact tool JSON with one short first-person rationale per real call. It then supplies task-specific facts, goal, initial state, concurrency facts, placement vocabulary, and tool contracts.

“Eight rules” does not mean the complete prompt is eight short sentences: the rules contain several clauses, and task/tool sections add constraints. Do not describe it as an unconstrained natural-language demonstration generator.

The current no-partition branch prefers coherent responsibilities for both agents when useful but permits a single physical worker for a genuinely sequential chain. This is guidance, not a fixed balanced ownership allocation.

### 5.4 Retry cascade

The implementation in `cascade_canary.py`, reused by `production_cascade.py`, attempts:

| Stage | Maximum trajectory attempts | Model/settings |
|---|---:|---|
| Direct generation | 3 | `gemini-3-flash-preview`, low thinking, temperature 0.6 |
| Cooler direct generation | 1 | Flash, low thinking, temperature 0.3 |
| Hotter direct generation | 1 | Flash, low thinking, temperature 0.9 |
| Medium direct generation | 2 | Flash, medium thinking |
| Critic-assisted regeneration | 2 | Flash high-thinking critic, then Flash low-thinking regeneration |
| Pro fallback | 2 | `gemini-3.1-pro-preview`, high thinking |

The maximum is 11 candidate attempts; a critic-assisted attempt can involve two API calls. Later stages retain cumulative observed failures. The current cascade does not use an example-trajectory phase. A candidate must pass the shared validator; a critic's suggested correction is not authoritative. Provider thinking settings govern internal generation effort and are distinct from the explicit per-action `reasoning` field in saved data.

### 5.5 Measured acceptance in the canonical 7,950 trajectories

Scanned from each saved raw record's `cascade_attempt_count` and `cascade_accepted_stage`:

| Accepted stage | Trajectories | Fraction of final dataset |
|---|---:|---:|
| First three Flash-low attempts | 7,574 | 95.27% |
| Flash-low, temperature 0.3 | 160 | 2.01% |
| Flash-low, temperature 0.9 | 59 | 0.74% |
| Flash-medium | 149 | 1.87% |
| Critic-assisted | 2 | 0.03% |
| Pro fallback | 6 | 0.08% |
| Total | 7,950 | 100% |

First-attempt successes are **6,329/7,950 = 79.61%**. The second and third attempts contribute 960 and 285. Every scanned source record has `validation.is_valid=true`; coordinator counts are exactly 3,975 each.

These are retrospective acceptance-stage counts for the final saved population. They are not estimates for unseen task families, and they do not establish that every API request succeeded or exclude failed attempts from the cost. `generation_usage` explicitly accumulates retry input/output/thinking usage. Financial totals should be recomputed from the recorded usage and historical pricing; this document does not claim a current price or elapsed-time total.

The physical-configuration canary's saved verification reports 384/384 valid, with 360 accepted in the first low stage, 14 at changed temperatures, and 10 at medium thinking. That canary is distinct from the full 7,950-trajectory run.

## 6. Simulation grounding, rendering, and scene coverage

### 6.1 Why symbolic validity and simulation replay are separate

The FSM tracks symbolic relations and scheduling. It cannot prove continuous reachability, robot clearance, object placement feasibility, camera visibility, or every native success predicate. A valid symbolic trajectory can fail in the simulator because a symbol resolves to the wrong counter, a placement sampler cannot find a pose, or an executor action fails physically.

Conversely, a permissive low-level tool can physically reposition an agent even when the symbolic call should have been rejected. The live pipeline therefore applies symbolic checks as well as simulation execution. “It rendered” is not equivalent to “it passed concurrent scheduling and native task success.”

### 6.2 Grounding and placement work already implemented

The project has audited and changed several sources of hidden implementation failure:

- Resolve cabinet parent counters explicitly and prioritize those relationships over generic counter matching.
- Canonicalize cabinet navigation/initial positions to parent workspaces while keeping cabinet contents located in the cabinet.
- Allow cabinet access without a front-facing exclusive pose and without displacing the partner through that cabinet-access path.
- Preserve an agent's position at an exclusive child when manipulating parent-counter items.
- Align navigation and `place_next_to` support resolution, including adjacent fixture references.
- Audit placement corridors, candidate sampling, initial robot ordering, countertop appliance approach poses, shared sink/stool-related workspaces, and object-placement strips.

The executor still contains a blocker-clearing fallback for some `require_front` paths. Do not claim “no tool ever moves the partner” globally from the cabinet-specific fix. Physical front-facing rules, collision geometry, symbolic exclusivity, and task support relations need separate descriptions.

Representative evidence is in the `probe_*`, `audit_*`, and `build_*_artifact.py` files under `training/bc_task_vlm`. These include cabinet parent semantics, parent access with an occupied support, shared-source pickups, reference-fixture placement, scene compatibility, and repaired task behavior.

### 6.3 Scene sampling

`scene_sampling.py` defines policy `certified_scenes_v1_no_layout18`. Candidate layouts are **11, 15, 40, 50**; styles are **14, 28, 34, 46, 58**; seeds are **42, 99, 7**. Their Cartesian product has 60 candidate scenes. A compatibility cache selects permitted scenes by task and physical configuration signature, then a seeded sampler chooses among them.

Layout 18 was intentionally excluded after narrow shared-workspace failures. Native RoboCasa layout support is not sufficient to guarantee compatibility with two robots and this executor. A successful reset is also weaker than proving every future shared-workspace access during an episode. The cache records an audited candidate set, not a universal safety proof over all physical states.

Scene selection is separate from symbolic configuration construction. Model-visible IDs remain scene agnostic; simulator grounding resolves concrete fixture identities. The fixed configuration cohort uses `scene_binding="certified_cache_at_evaluation"`. Reproducibility requires retaining the selected scene and cache version, not only a symbolic configuration hash.

### 6.4 Data lifecycle

```text
Task specification + requested configuration + generation variation
  -> candidate concurrent ticks
  -> shared symbolic/concurrent validation and bounded retries
  -> insert get_image calls; revalidate observation-transparent action grid
  -> bind scene and symbolic IDs; replay/render
  -> check execution/image alignment and select trajectories
  -> build private agent-turn examples
  -> model-specific processing/tokenization
  -> SFT checkpoint
  -> independent live-simulation rollout on a fixed cohort
```

Postprocessing and generation are separate transformations, but they should use the same validity contract rather than incompatible definitions of concurrency. Current validation metadata uses `transactional_tick_v3_native_goal_parity`; `require_current_validation` rejects missing/stale contract markers for consumers that invoke it. A metadata check alone does not rerun the simulator.

## 7. Dataset contents and publication representation

### 7.1 Canonical sources

```text
/work/umass/shlomo_umass/dbenhamougol_umass/
  tick53x150_state_grounded_cascade_v1_raw/
    <task>/trajectories/traj_XXXXXX.json
    <task>/cascade_attempts/run_XXXXXX.json
  tick53x150_state_grounded_cascade_v1_rendered/
    <task>/traj_XXXXXX/
      original_trajectory.json
      plan.json
      metadata.json
      adapted_trajectory.json
      images/...
```

The exporter defaults explicitly identify these two roots. The scale-training launchers also identify this rendered root. Historical directories containing 30/task or 100-selected-from-150 data are separate versions and should not be substituted based only on a similar name.

The exporter's task-goal lookup reads the current verified specifications, and the publication metadata carries a policy prompt contract. These do not by themselves freeze the exact historical LLM generation prompt. For generation reproducibility, retain the generation-era specification/prompt source and attempt records in addition to the publication version label.

A raw record includes task and trajectory IDs, agents, initial state, flat steps, tick rows, coordinator, physical configuration/signature, grounding map, validation/final state, accepted cascade stage/attempt count, sampling metadata, and generation usage. Rendering metadata contains tool execution outcomes and image paths. The source example inspected during this review is PlateStoreDinner `traj_000088`.

### 7.2 Trajectory versus tick versus training example

| Unit | Meaning | Count in the inspected source/export |
|---|---|---:|
| Task | Household goal template | 53 |
| Trajectory/episode | Complete generated two-agent execution | 7,950; exactly 150 per task |
| Raw action tick | Concurrent row, possibly with one blocked agent | 58,607 |
| Exported tick-table row | Action tick or final observation-only “Task result” row | 66,557 |
| Agent call | One agent's call within a tick | 104,199 raw calls, before inserted observations |
| SFT example | One next-call target from one private agent context | Not the same as trajectory or tick count; depends on observation insertion and preprocessing |

The 7,950-row difference between exported and raw ticks is one final observation row per episode. Raw trajectory length has minimum 4, median 7, mean 7.372, and maximum 18 ticks. These lengths include communication. They are not counts of physical actions or simulator frames.

### 7.3 Raw tool-call distribution

Counts scanned from raw `tick_rows`; inserted observation calls are excluded:

| Tool | Calls |
|---|---:|
| `communicate` | 40,281 |
| `navigate_to_fixture` | 17,572 |
| `pick_up_object` | 17,297 |
| `wait_for_signal` | 6,043 |
| `place_in_receptacle` | 5,100 |
| `place_on_object` | 4,950 |
| `place_next_to` | 4,797 |
| `open_hinged_part` | 2,711 |
| `place_on_surface` | 2,300 |
| `give_space` | 2,082 |
| `close_hinged_part` | 476 |
| `open_sliding_part` | 290 |
| `place_under` | 150 |
| `press_button` | 150 |

The global interface contains tools not exercised in this dataset. The table is not the distribution of all SFT target tokens: communication/rationale lengths vary, and inserted `get_image` requests add targets.

### 7.4 Hugging Face/export

The publication identity is `DorianAtSchool/RoboTalk`. Local export files are under [`reports/robotalk_hf_full`](../training/bc_task_vlm/reports/robotalk_hf_full). Its `dataset_info.json` records 7,950 episodes, 66,557 rows, and prompt contract `state_grounded_global_tools_no_index_v9_unambiguous_garnish_cake_goal`.

```python
from datasets import load_dataset

trajectories = load_dataset("DorianAtSchool/RoboTalk", "trajectories", split="train")
ticks = load_dataset("DorianAtSchool/RoboTalk", "ticks", split="train")
```

Here Hugging Face's `train` split names the published table, not the project's 43/10 task partition. Media are per-episode tar archives; JSON image maps identify members. Original raw JSON is retained. This inspection verified local publication artifacts, not remote service availability.

The explorer supports task/trajectory selection, forward/backward ticks, per-agent views including wrist, a third global view, observation-call display, rationales, initial state, goal/prompt views, and export facilities. A missing image should remain visibly unavailable until a call produces it. Final observations appear without fabricating final physical actions.

The explorer's `PHYSICAL_TOOLS` includes navigation and `give_space`, whereas the evaluation participation metric excludes them. **Do not compare explorer work-share values directly with evaluation “one physical worker” values.** This difference is visible in `export_robotalk.py` versus `build_43_10_eval_artifact.py`.

## 8. Policy prompts and SFT construction

### 8.1 Shared prompt contents

`build_partial_user_prompt` is called by both private-history dataset construction and live evaluation. It renders task family, task instruction, current agent, coordinator for full communication, observation-view names, symbolic initial state, derived concurrency facts, private history, camera/workspace/communication guidance, and general tool-call rules.

No local or global step indices are included in production prompts. Tick/step indices retained in logs and the explorer are diagnostic metadata, not policy input. Task goals are loaded through task metadata, avoiding a task-family identifier accidentally being used as the full instruction.

For tool-call training, function schemas are passed separately into the chat template. The user prompt does not repeat a second full “Available tools” block. The Qwen template handles tool serialization; Gemini uses native function declarations. The same semantic inputs do not imply identical provider token sequences.

`Observation views attached in order: none` means no images are attached to this invocation. It is expected with consume-once observations, not proof of failed rendering.

### 8.2 Agent histories and causal construction

`dataset.py` builds each acting agent's history from its own executed actions and received `communicate` messages. Same-tick history is buffered and delivered after constructing targets for both agents. This prevents the second serialized agent from seeing information unavailable when the first agent chose its call.

The next target is one tool call, optionally preceded by that call's rationale. For example, one trajectory contributes an agent-0 opening-message example, an agent-1 opening-message example, then further examples for their later calls and inserted observations. Each example reconstructs the appropriate prefix; training does not run the model freely through a complete simulator episode.

Ordinary successful demonstrations have no rejection histories. Live policies can encounter errors and react to their own feedback, but supervised imitation of clean examples does not directly teach those reactions. This motivates reporting both final and error-free success and motivates the correction pipeline.

### 8.3 Loss and weighting

The model receives teacher-forced target tokens. The collator masks prompt, image-context, and padding positions with `-100`, leaving assistant target tokens for cross-entropy. Rationale-enabled samples use `<think>...</think>` assistant content plus a tool-call message. Earlier rationale text is not automatically fed back as future action history.

The implemented dataset is organized by agent-turn examples, not uniformly weighted complete trajectories. Equal trajectories per task therefore do not give equal task weight: longer trajectories generate more examples; longer target messages/rationales contribute different numbers of tokens under the trainer's loss reduction. The paper should not claim per-trajectory loss normalization unless a specific experiment implements it.

### 8.4 Why preprocess separately

Preprocessing validates alignment of raw actions, render plans, and execution records; assembles private contexts; resolves image files; builds schemas/messages; and can pretokenize multimodal inputs. This is substantial repeated CPU/filesystem work if done at every training launch. Reusable artifacts reduce startup cost and ensure matched experiment inputs.

Pretokenized artifacts are processor/template specific. A Thinking model must not blindly consume token IDs produced with an incompatible Instruct template. Saved examples without pretokenization can be processed by the selected model during training. Dataset fingerprints and manifests matter when task goals, tool descriptions, or prompt contracts change.

### 8.5 Context and output budgets

The inspected regularized training runs use `max_length=8192`, image resolution 512, and at most four images per sample. Multimodal sequence budgeting includes image-token expansion, not merely the tokenized natural-language text. Input length and generated-output limits are different settings.

The inspected collator/preprocessor still contains left-truncation paths, and vLLM launchers default to a 16,384-token server context while evaluation defaults may request only 256 new tokens. Thinking experiments override generation budgets. Do not claim universal identical 8,192-token live/SFT budgets or guaranteed `context_limit_exceeded` reporting based solely on prior plans: this inspection did not find that literal status implemented in the current live evaluator. Record the effective configuration for each run.

Teacher-forced Thinking-without-rationale SFT does not secretly sample thousands of free reasoning tokens during optimization. It trains on the provided target tool calls. Using a Thinking base model, enabling a generation-time template option, and supervising rationale tokens are three distinct choices.

## 9. Training experiments and saved configuration

### 9.1 Current scale comparison

The principal comparison uses **43 training task types and 10 held-out task types**, nested subsets of **30, 60, 90, 120, 150 trajectories per training task**, and **one training epoch**. Total original trajectories used are therefore 1,290; 2,580; 3,870; 5,160; and 6,450. The dataset itself still includes all 53 task types; held-out membership is imposed by experiment manifests.

`build_scale_experiment_manifests.py` groups trajectories by physical configuration signature, shuffles within groups reproducibly, and interleaves the groups. Taking prefixes gives nested scale subsets with broader configuration coverage than simply selecting the first trajectory IDs.

Earlier experiments used 47/6 task splits and 27/3 or 90/10 within-task trajectory splits. Those labels are different axes. Current fixed live evaluation measures newly executed episodes on trained task types; it is not imitation scoring against a held-out expert trajectory.

### 9.2 Verified representative hyperparameters

Read from saved `run_config.json` files for regularized 43/10 30/task Instruct, Thinking+rationale, and Thinking-without-rationale runs; scale launchers use the same principal settings:

| Setting | Value |
|---|---|
| Base model | `Qwen/Qwen3-VL-8B-Instruct` or `Qwen/Qwen3-VL-8B-Thinking` |
| Adaptation | LoRA |
| Rank / alpha / dropout | 16 / 32 / 0.05 |
| Target module suffixes | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Learning rate | `1e-4` |
| Weight decay | `0.01` |
| Scheduler / warmup | cosine / 0.03 |
| Epochs | 1 |
| GPUs / examples per GPU / accumulation | 4 / 8 / 2; nominal effective batch 64 |
| Sequence limit / image resolution / image cap | 8,192 / 512 / 4 |
| Gradient checkpointing | enabled |
| Target format | `tool_call` |
| Observation targets | enabled |
| History / indices / images | private / none / consume-once |
| Training seed | 42 |
| Checkpoints | two per epoch in inspected runs; selected figures use epoch 1.0 |

A saved regularized Instruct checkpoint (`checkpoint-324`) contains **504 adapter tensors and zero tensor names containing `visual` or `vision`**, inspected directly from the safetensors header. Its adapter settings alone list module suffixes, so tensor names are the stronger evidence for that checkpoint. Do not describe all historical variants as vision-backbone fine-tuning. Parameter scope should be verified per checkpoint family.

W&B is configured by launchers, with explicit login checks in current training scripts. Checkpoint watcher and Slurm dependency launchers support evaluation after a checkpoint appears; current plots deliberately use selected one-epoch checkpoints rather than all saved half-epoch checkpoints.

### 9.3 Four 30/task model/target combinations

1. Instruct base, tool-call targets only.
2. Instruct base, rationale plus tool-call targets.
3. Thinking base, tool-call targets only.
4. Thinking base, rationale plus tool-call targets.

The rationale is the saved demonstration sentence, not newly generated Qwen reasoning. It is accurate to say “training with rationale traces.” The Thinking-without-rationale evaluation has a documented missing-opening-tool-tag parser recovery and must be labeled accordingly; it is not strictly the same parsing condition as every other arm.

## 10. Fixed-cohort evaluation

### 10.1 Population and sampling

The configuration manifest stores unique symbolic configurations, including coordinator identity, separately from any expert action sequence. An evaluation is a fresh policy rollout toward task success. A “carrier” used in older manifests supplied a known initial scene/state; the current configuration-native entries need no target expert sequence for policy evaluation.

Two modes are implemented:

- `sampled`: 10 episodes per task by default, adjustable with `--episodes-per-task`. Configurations are shuffled without replacement until exhausted, then another shuffled pass begins.
- `full_config`: one episode per configuration by default; `--episodes-per-config` adds repeats and `--max-configs-per-task` caps the selected configurations.

The default 43/10 population is 430 trained-task episodes and 100 held-out-task episodes. Repeated symbolic configurations do not necessarily provide independent behavioral diversity under deterministic decoding. Episode IDs distinguish trials; IDs alone do not cause different actions. vLLM requests include a generation seed, but temperature zero can still repeat the same continuation.

The selection seed hashes split name, task, mode, and evaluation seed. Consequently, regrouping a saved 47/6 run into 43/10 preserves those original episodes, while freshly selecting under 43/10 can change the episode population for moved tasks. The regrouped results are useful comparisons but should not be called perfectly paired native-43/10 rollouts without checking configuration/scene identity.

### 10.2 Runtime and failure accounting

Fixed-cohort launchers explicitly use private history, no step indices, consume-once observations, uniform durations, FSM success, raster maps, and active observations. Multiple simulator workers share a vLLM server for open-weight evaluation. Hosted Gemini evaluation uses native function calls. The present vLLM container path names version 0.27.1; historical evaluations may have used other serving environments.

The default evaluation can provide a rejected-call error to the responsible agent and continue until goal or termination budget. That is different from silently retrying the same model output. Transport/API retries also exist and should not be confused with policy recovery. Failure messages are private to the responsible agent.

The evaluator records proposals, legality, execution/simulator status, time/cycle fields, rejections, termination reasons, FSM/native outcomes, and private reasoning/parser diagnostics. The raw JSONL is essential when interpreting a failure. The last printed error is not necessarily the first causal divergence.

### 10.3 Metrics

Let N be the number of evaluated episodes, S the number reaching FSM success, and E the number reaching FSM success without any recorded rejection.

- **Final FSM success:** S/N.
- **Error-free FSM success:** E/N.
- **Success requiring recovery:** (S−E)/N.
- **Error-free success with exactly one physical worker:** episodes satisfying both conditions, divided by N.
- **Dominant-worker success:** successful episodes where one agent performs at least 80% of counted task-changing actions, divided by N when shown as an inset subset.

These last two are different metrics. The current communication comparison's inset uses **exactly one** physical-work agent, not the earlier ≥80% threshold. The evaluation excludes communication, observation, navigation, `give_space`, waiting, failure reports, and completion markers from counted task-changing actions. A single worker metric therefore does not say that the other agent never communicated or helped with navigation/handover.

Error-free success is a useful proxy for success if terminating at the first rejection. It is computed from completed logs; it is not a separate intervention that reruns the policy. Harness problems and historical silent interventions need their own audit before interpreting that proxy.

### 10.4 Micro, macro, confidence intervals, and error categories

Micro success pools successful episodes over all episodes. Macro success averages task success rates. When every task has exactly the same episode count they are equal, but the uncertainty calculations can differ.

The current 43/10 artifact uses 95% Wilson intervals for binary rates. With sample proportion p̂=k/n and z=1.959963984540054:

```text
center = (p̂ + z²/(2n)) / (1 + z²/n)
half_width = z * sqrt(p̂(1−p̂)/n + z²/(4n²)) / (1 + z²/n)
interval = center ± half_width
```

The standalone fixed-cohort summarizer also implements task/episode hierarchical bootstrap and paired comparisons. Do not label Wilson whiskers as hierarchical-bootstrap intervals. Wilson intervals describe binomial uncertainty under independence; episodes share tasks, configurations, and one checkpoint, so they do not capture every source of experimental variation. Multiple training seeds and more held-out task families would strengthen generalization claims.

Three error categories are implemented in `summarize_fixed_live_sim.py`: **call construction/grounding**, **state/action execution**, and **coordination/progress**. The classifier retains raw error subtypes. The state/action fallback includes physical simulator errors, so that category alone is not proof of a learned planning failure. Error plots separate errors in eventual successes from errors in failed episodes. Task-by-task charts group task splits and show split-level means.

## 11. Communication-guidance ablations

The intended experimental progression is below. Physical action rules, state information, and camera vocabulary remain relevant in every condition.

| Condition | Communication tool | Encouragement/guidance | Communication protocol |
|---|---|---|---|
| None | removed | Cannot exchange messages; other agent still pursues the same goal | absent |
| Unguided | present | Basic communication mechanics, no recommendation to communicate | absent |
| Minimal | present | Brief instruction to communicate; basic request/wait/release guidance | absent |
| Intermediate | present | More complete work-sharing, release, handover, and own-portion completion guidance | absent |
| Full | present | Canonical SFT/default evaluation guidance | sampled coordinator, two opening communication ticks, phase checks |

`wait_for_signal` remains available in None for permanent self-suspension. In the intended communicative conditions, a matching later release wakes the agent. A runtime-enforced protocol changes admissible behavior in addition to changing prompt text; this is not solely a test of extra prompt length.

### 11.1 Wait-description discrepancy, fixed September 14

The initial inspection found that every `CommunicationProfile` had `permanent_wait=True`. `apply_communication_profile` returned early for Full, but used that flag to describe **all non-full modes** as having no communication and no possible release. That was appropriate for None but contradictory for Unguided, Minimal, and Intermediate. The live scheduler still used actual matching-release mechanics.

After user authorization on September 14, the description override was fixed to depend on communication being disabled. Unguided and Minimal now describe basic matching-release waits; Intermediate retains the canonical detailed wait description. Full and None prompts/tool metadata were checked to remain exactly unchanged. New regression tests cover all modes, and reviewed prompt/schema snapshots are saved in `training/bc_task_vlm/reports/communication_waitfix_v3/prompt_snapshots.json`. Qwen Instruct and Gemini reruns use `_waitfix_v3` output names and the same selected episode IDs as the earlier runs. The result tables in this document remain the historical September 6 snapshot pending those reruns; they must not be presented as corrected-condition results.

### 11.2 Intended Intermediate contrast versus literal implementation

Intermediate removes coordinator identity, opening-phase fields, handshake enforcement, and initial-communication gating, while adding its own communication-guidance block and retaining physical handoff rules. Full includes phase-tagged `portion_complete` wording. Therefore “Full minus protocol” describes the intended conceptual contrast, but current text is not produced by a single literal deletion of a protocol block. Exact prompt and schema snapshots are needed for a clean paper table or appendix.

Do not claim that supplying a protocol alone always helps every model. In the stored artifact, Gemini has substantially higher error-free Full success, while base Qwen Instruct struggles with the protocol. Also distinguish final success with errors permitted from error-free success: they give quite different pictures of what communication buys.

## 12. Results snapshot for writing

**Source:** [`fixed_live_sim_43_10_results_artifact/artifact.json`](../training/bc_task_vlm/eval_runs/fixed_live_sim_43_10_results_artifact/artifact.json), snapshot dated September 6, 2026. These are stored artifact results, not a new evaluation or a claim that the artifact reflects every subsequent code edit. Its rows include source metric paths. Main figures prioritize error-free FSM success.

### 12.1 One-epoch scale comparison

Each row uses 430 trained-task and 100 held-out-task episodes. Entries are **error-free successes / episodes (percent)**.

| Trajectories per training task | Instruct, trained tasks | Instruct, held-out tasks | Thinking+rationale, trained tasks | Thinking+rationale, held-out tasks |
|---:|---:|---:|---:|---:|
| 30 | 352/430 (81.86%) | 45/100 (45%) | 345/430 (80.23%) | 71/100 (71%) |
| 60 | 376/430 (87.44%) | 57/100 (57%) | 364/430 (84.65%) | 73/100 (73%) |
| 90 | 386/430 (89.77%) | 67/100 (67%) | 393/430 (91.40%) | 66/100 (66%) |
| 120 | 400/430 (93.02%) | 48/100 (48%) | 388/430 (90.23%) | 77/100 (77%) |
| 150 | 405/430 (94.19%) | 65/100 (65%) | 398/430 (92.56%) | 73/100 (73%) |

The stored Instruct results improve monotonically with scale on trained task types, while held-out-task rates are nonmonotonic. Thinking+rationale improves held-out performance over matched Instruct at four of five scales; 90/task is the exception. These results support a distinction between trained-task gains and task generalization. They do not prove a universal law that more demonstrations worsen generalization, nor isolate model pretraining from rationale targets in the two-family scale comparison.

The strongest stored one-epoch Instruct trained-task figure is 94.19% at 150/task; the strongest held-out Instruct figure is 67% at 90/task. They come from **different checkpoints/scales**. A phrase such as “up to 94% and 67%” must not imply a single model simultaneously attained both maxima. Thinking+rationale reaches 77% held-out error-free success at 120/task in this snapshot.

### 12.2 Thirty/task target/base ablations and corrected-data follow-up

| Variant | Error-free trained | Error-free held-out | Final trained | Final held-out |
|---|---:|---:|---:|---:|
| Instruct, tool only | 352/430 | 45/100 | 373/430 | 54/100 |
| Instruct + rationale | 355/430 | 58/100 | 377/430 | 68/100 |
| Thinking, tool only; parser recovery enabled | 308/430 | 50/100 | 339/430 | 67/100 |
| Thinking + rationale | 345/430 | 71/100 | 364/430 | 78/100 |
| DAgger-100, Flash expert | 355/430 | 47/100 | 367/430 | 56/100 |
| DAgger-100, Gemini 3.1 Pro expert | 331/430 | 52/100 | 361/430 | 60/100 |

DAgger rows are additional training from a specified SFT base and a corrected/original mixture, not another pure 30/task-from-base run. Establish checkpoint provenance before attributing differences to correction quality. Older DAgger-89 results were intentionally removed from this comparison because their base checkpoint differed.

### 12.3 Out-of-the-box communication results

Entries are **error-free / final successes**, with denominators 430 for trained task types and 100 for held-out task types. Split membership is regrouped from existing evaluations where indicated by artifact provenance.

| Model | Mode | Trained task types | Held-out task types |
|---|---|---:|---:|
| Qwen3-VL-8B-Instruct | None | 1 / 127 | 0 / 11 |
| Qwen3-VL-8B-Instruct | Unguided | 2 / 211 | 0 / 34 |
| Qwen3-VL-8B-Instruct | Minimal | 4 / 197 | 0 / 30 |
| Qwen3-VL-8B-Instruct | Intermediate | 3 / 172 | 0 / 25 |
| Qwen3-VL-8B-Instruct | Full | 0 / 40 | 0 / 8 |
| Gemini 3 Flash | None | 33 / 367 | 5 / 75 |
| Gemini 3 Flash | Unguided | 41 / 397 | 6 / 95 |
| Gemini 3 Flash | Minimal | 197 / 407 | 32 / 85 |
| Gemini 3 Flash | Intermediate | 197 / 387 | 32 / 88 |
| Gemini 3 Flash | Full | 282 / 387 | 72 / 90 |

Minimal and Intermediate have identical Gemini **error-free counts** in this artifact, but their final-success counts differ. Equality of one metric is not evidence that the same outputs were reused. The source runs and prompt snapshots, not rounded graph labels, settle that question.

Appropriate interpretation: communication guidance substantially affects reliable task completion, with different responses across model families. A broad claim that “communication is necessary for any task success” is contradicted by nonzero no-communication final success. A claim that “the protocol is insufficient without specialized training” is also too broad given Gemini's Full results. The current-code discrepancies in section 11 should be resolved against run snapshots before a definitive causal interpretation.

## 13. DAgger-inspired collection and training

### 13.1 Purpose and scope

The implemented system uses learner-visited states to produce corrected demonstrations. It supports both saved failed evaluation trajectories and fresh live rollouts from the learner. Current collection is restricted to `train_task_types`; held-out tasks must not enter correction training.

This is best described as **DAgger-inspired expert correction and data aggregation**, rather than an exact implementation of every detail of classic DAgger. It does not label every visited state with an independently queried oracle action. It can replace an earlier causal branch, insert communication/resynchronization, and roll out the learner again.

### 13.2 Loop

```text
Learner rollout, stopping at the first rejection
  -> deterministic diagnosis and candidate branch states
  -> LLM expert chooses local fix / coordination correction / resync
  -> structural validation
  -> shared concurrent-FSM replay from the branch
  -> live learner continuation
  -> later rejection: repeat correction
  -> clean success: materialize for training
  -> simulator-only rejection or exhausted budget: report separately
```

The branch replaces the relevant concurrent tick; it is not inserted after a rejected call or after a wait that would already block the coordinator. The partner's same-tick call must also be represented consistently. `get_image` observations must be preserved with causal alignment. A pre-deadlock branch is necessary if a resynchronization would otherwise ask an already blocked agent to speak.

The expert receives task goal, initial state, current authoritative FSM snapshots at candidate branches, objects/held objects/locations, blocked agents, task constraints/effects, original plan messages, full trajectory, and cumulative correction errors. The prompt asks for an avoidance correction and preservation of a valid original plan where possible. Repeated valid short corrections followed by learner failure can trigger a complete goal-reaching expert suffix.

Structural rejection means the correction object/tick/protocol representation is invalid. FSM rejection means its calls violate state, resource, scheduling, or goal conditions when replayed. Passing both still does not guarantee simulator execution or successful learner continuation.

### 13.3 Strict data and simulator failures

The fresh collector requests `terminate-first-rejection`. Strict clean successes have FSM success and no rejection in the corrected rollout. Simulator-only failures—symbolically legal calls whose physical execution fails—are separately categorized rather than corrected as if they were planning errors. Historical permissive collections exist and should not be mixed silently with strict data.

Correction-type summaries are implemented in `fix_distribution.py`. Fresh batch selection varies evaluation seeds/configuration ordering, and collection admission uses task-aware balancing/caps. Repeated trials do not automatically guarantee novel policy behavior; decoder and initial-state diversity matter.

### 13.4 Materialization and mixture

`dagger/data_generation` replays corrected trajectories, saves durable observations, writes canonical training records, checks validity/alignment, and invokes the shared preprocessor. Its mixture rule is **N corrected trajectories plus 4N original trajectories**: 100 valid corrected trajectories give 400 originals, not a fixed original pool independent of collected count.

This is 20% corrected data by trajectory count, not necessarily by agent turns or target tokens. Rejected learner calls remain diagnostic records; they are not the desired targets in an avoidance dataset. The accepted prefix, valid correction, and successful continuation supply the training trajectory. The pipeline checks images, provenance, and shared prompt construction before training.

## 14. Concrete coordination example for a paper figure

**PlateStoreDinner / `traj_000088`** is a useful real demonstration. Its actual opening plan assigns agent 0 the fridge opening plus meat1/plate, and agent 1 meat2/bowl/fridge storage. The current goal is “place one steak on the plate and the other steak in the bowl, then store the bowl in the fridge.” Initial symbolic locations are agent 0 at fridge and agent 1 at counter, with the fridge door closed. The record uses coordinator agent 0.

The first physical tick opens the fridge while the partner navigates to the stove. Next, agent 0 gives space from the fridge while agent 1 picks meat2 from the pan. Later handover/wait communication coordinates stove access, and bowl storage follows filling. The source validation records a lock-step makespan of 15 and goal at 14; those are logical timing fields, not measured seconds.

Here is the complete raw action grid, with communication text shortened for readability. Tool names, arguments, tick boundaries, and blocked periods follow the saved record; inserted observation calls are omitted from this particular table.

| Tick | Agent 0 | Agent 1 |
|---:|---|---|
| 0 | Communicate proposal: open fridge; take meat1 to plate; partner handles meat2/bowl/storage | Communicate `await_plan` |
| 1 | Communicate `await_confirmation` | Confirm that division |
| 2 | `open_hinged_part(fridge, door)` | `navigate_to_fixture(stove)` |
| 3 | `give_space(fridge)` | `pick_up_object(meat2, source_id=pan)` |
| 4 | Request stove release; report fridge open and vacated | `give_space(stove)` |
| 5 | `wait_for_signal(from=agent_1, about=stove)` | `navigate_to_fixture(counter)` |
| 6 | Not invoked: blocked on stove | `place_in_receptacle(meat2, bowl)` |
| 7 | Not invoked: still blocked during release tick | Communicate `releases=stove`; request fridge release |
| 8 | `navigate_to_fixture(stove)` | `wait_for_signal(from=agent_0, about=fridge)` |
| 9 | `pick_up_object(meat1, source_id=pan)` | Not invoked: blocked on fridge |
| 10 | `navigate_to_fixture(counter)` | Not invoked: blocked on fridge |
| 11 | Communicate `releases=fridge` | Not invoked: still blocked during release tick |
| 12 | `place_on_object(meat1, plate)` | `pick_up_object(bowl, source_id=counter)` |
| 13 | Report own portion complete; name `counter` release keyword | `navigate_to_fixture(fridge)` |
| 14 | `wait_for_signal(from=agent_1, about=counter)` | `place_in_receptacle(bowl, fridge)`; FSM goal reached |

This is a valid demonstration of delayed release and blocked-agent behavior, but not an optimal schedule: agent 0 physically vacates the fridge at tick 3, while agent 1 later waits for an explicit release before returning there. That extra delay is appropriate to mention if discussing efficiency. Validation proves legality under the stated contract, not minimum communication or minimum makespan.

The selected initial room view reproduces layout 50, style 34, seed 7. Camera refinement was only for the explanatory figure. It does not change agent observations in the dataset or evaluation. The chosen view is [`plate_store_dinner_traj088_initial_interior_lower.png`](../training/bc_task_vlm/reports/plate_store_dinner_traj088_overview/plate_store_dinner_traj088_initial_interior_lower.png), generated by [`capture_plate_store_dinner_overview.py`](../training/bc_task_vlm/capture_plate_store_dinner_overview.py).

For a motivation figure, distinguish actual policy-generated failure logs from fabricated illustrations. Useful failure types include duplicate ownership, incompatible initial plans, entry into an occupied exclusive fixture, moving a receptacle while the partner uses it, premature global-completion messages, and missing/private wait requests. Check original cycle metadata before interpreting a table: serialized `get_image` rows are not additional physical ticks, and “Not Invoked” needs a preceding wait or episode termination.

## 15. Historical changes relevant to interpretation

The following progression explains why old and new result artifacts cannot simply be pooled:

| Earlier issue | Current implementation direction | Paper implication |
|---|---|---|
| Centralized history/agent prediction | Separate contexts with fixed acting agent | Specify the evaluated architecture, not old README defaults |
| Global/local step-index differences | Shared no-index prompt builder | Diagnostic ticks must not be mistaken for model input |
| Divergent simultaneous opening plans | Sampled coordinator and confirmation protocol | Protocol motivation; not proof every plan remains consistent |
| Same-tick release/entry or precondition dependence | Frozen beginning-of-tick checks and next-tick release semantics | Expert replay and live decisions use matched logical timing |
| Idle unblocked agents in demonstrations | Canonical invocation checks and explicit blocked markers | Flat valid-looking trajectories can still be invalid concurrently |
| Premature global completion/filler communication | Portion-specific report followed by wait | Completion is evaluator-controlled |
| Independent live resource locks | Shared concurrent contention classifier | Shared-source pickup should not be misreported as model error |
| Cabinets as exclusive workspaces | Shared parent/cabinet aliases and revised navigation | Requires version-aware data/replay interpretation |
| Generic counter grounding and spawn collisions | Support-aware grounding and scene/placement audits | Some failures were simulator/harness defects |
| Task-local policy schemas and missing state | Global tool interface plus symbolic initial state | Current models do receive explicit entity information |
| Tool-output/reasoning parser incompatibilities | Native serving support and documented recovery modes | Formatting failures can be harness-sensitive |
| Silent mutual-wait clearing | Explicit deadlock termination | Old final-success logs may need correction |
| Expert-trajectory-derived evaluation splits | Configuration-native fixed cohorts | Use “trained task types,” not “held-out trajectories” |

Historical discussions of low/medium/high thinking, K trajectories per API call, work partitions, or universal context-overflow handling are not enough to establish what the production run did. Prefer its saved metadata and current executable launch parameters.

## 16. Limits, discrepancies, and checks before submission

1. **FSM versus native success:** primary scores use the FSM. Native geometry, gripper clearance, contact/stirring, and pruning can differ. The [native/FSM audit](../training/bc_task_vlm/reports/native_fsm_success_audit/report.md) records fixed GarnishCake and PrepareCoffee semantic mismatches, a remaining deliberately pruned OrganizeCondiments distractor, and physical abstractions such as CheeseMixing and bread contact. Do not claim literal native-goal equivalence across all continuous states.
2. **CheeseMixing is explicitly abstracted:** its current instruction says placing the spatula in the pot counts as stirring. It is not evidence of learned stirring motor control.
3. **Non-full wait-description contradiction:** section 11 records the September 14 fix and replacement evaluation family. Treat old Unguided/Minimal/Intermediate results as potentially affected until the corrected runs are incorporated.
4. **Prompt parity has multiple layers:** a shared function helps, but task instruction, initial state, history delivery, tool projection, chat template, image timing, truncation, and checkpoint-era source versions must also agree. Existing basic tests do not prove byte-for-byte parity of every historical request.
5. **Metric vocabulary differs across tools:** explorer work share includes navigation/handover; evaluation task-changing work excludes them. Exactly-one-worker and ≥80%-worker are not interchangeable.
6. **Cohort overlap:** trained-task evaluation configurations can overlap training configurations. This is intentional task-instance evaluation, not a guarantee of unseen symbolic configurations. Held-out task types are the explicit generalization test.
7. **Regrouped versus native cohort:** moving task labels changes aggregate membership but does not regenerate episode/configuration selections. Account for split-dependent sampling seeds.
8. **Scene coverage is finite:** the certified set is not all RoboCasa scenes, and a successful initialization is not proof of every future interaction. Retain task/configuration/scene audit evidence.
9. **Physical executor side effects:** cabinet access avoids silent partner clearing, but other front-facing fallback code still exists. Describe scoped guarantees accurately.
10. **One checkpoint is not a training-seed distribution:** Wilson error bars do not quantify variability over retraining, prompt revisions, provider changes, or scene-policy alternatives.
11. **Context budgets:** inspected training truncation and server/output defaults are not a universal fail-closed context-limit implementation. Document effective limits and any discarded/truncated examples for reported runs.
12. **Publication metadata:** the local dataset card still says licensing must be completed before public release, despite the session's subsequent public-release request. Resolve the license text and dataset/model asset terms before treating the card as final publication metadata.
13. **Source version:** save the exact working-tree diff or commit, task specs, prompt text/schema snapshots, processor versions, scene cache, selection manifests, run configs, parser flags, checkpoint identities, and JSONL result sources for the paper release.

## 17. Suggested paper organization and supported claims

**Introduction:** high-level embodied planning is increasingly common, while independent agents need communication to coordinate under separate evolving contexts. Motivate temporal dependencies and shared resources with an actual rollout example.

**Environment and interface:** describe task/state schema, two agents, scripted execution, observation interface, and the distinction between logical concurrency and simulator operations. Define the FSM success metric and its physical-abstraction limits.

**RoboTalk construction:** explain task specifications, balanced configuration exposure, structured generation variation, the validator/retry cascade, observation insertion/rendering, and the 7,950-trajectory release. Include first-attempt and bounded-cascade acceptance with denominators.

**Coordination protocol:** state the two opening ticks and later request/wait/release mechanics. Make the reason for next-tick message effects explicit. Do not claim natural-language plan correctness is formally verified just because phase order is checked.

**Learning/evaluation:** give nested 43/10 scale manifests, agent-turn targets, LoRA configuration, rationale variants, fixed episode selection, effective serving settings, and metric definitions.

**Results:** separate communication ablations, scale, 30/task base/target ablations, and DAgger follow-up. Report error-free and final success explicitly. Discuss single-worker successful episodes as a qualification to interpreting success as collaboration.

**Limitations:** limited task/scene coverage, symbolic physical abstraction, finite test episodes/training seeds, overlap of trained configurations, simulator/harness sensitivity, and exact ablation prompt discrepancies pending resolution.

Supported concise claim: “RoboTalk provides validated concurrent demonstrations for two embodied agents with private execution contexts. Fine-tuning improves reliable task completion, while generalization and the benefit of communication guidance vary by model and training configuration.”

Avoid unsupported stronger claims: “communication is always necessary,” “more data universally reduces generalization,” “all task success is native physical success,” “the agents never share global information,” or “every comparison differs only by one prompt sentence.”

## 18. Reproduction and evidence checklist

For a paper result, record the task split and nested trajectory manifest; raw/rendered source roots; task-spec and prompt/schema versions; whether rationale targets and image calls are trained; base model and adapter; epoch/learning rate/LoRA settings; serving/parser/output/context settings; configuration/scene seed and cache; rejection mode; and the exact aggregate/trajectory JSONL paths.

The existing [`43/10 HTML artifact`](../training/bc_task_vlm/eval_runs/fixed_live_sim_43_10_results_artifact/report.html) provides the visual comparison. [`export_communication_ablation_png.py`](../training/bc_task_vlm/export_communication_ablation_png.py) and [`export_sft_scale_pngs.py`](../training/bc_task_vlm/export_sft_scale_pngs.py) produce compact paper figures. Caption metadata must match the selected denominator, split, checkpoint, and inset definition.

Relevant tests available for follow-up include `tests/test_concurrent_fsm.py`, `tests/test_workspace_semantics.py`, `tests/test_shared_workspace.py`, `tests/test_task_level_tool_interface.py`, `tests/test_simplified_generation_prompt.py`, `tests/test_scene_sampling.py`, and the training tests for prompt contracts, communication profiles, and fixed cohort selection. This documentation inspection did not rerun simulation audits or training; it examined their code, stored evidence, and representative records.

## Appendix A. Current task catalogue

The following table quotes the current verified-spec `task_goal`, not necessarily the untouched upstream RoboCasa instruction. Configuration counts are unique physical signatures observed in the canonical raw dataset; coordinator identity and scene parameters are excluded. Every task has 150 trajectories. Split labels refer to the 43/10 experiment.

| Task / specification | Split | Physical configurations | Current goal instruction |
|---|---|---:|---|
| [AddLemonToFish](../data_generation/task_level/tasks/specs/verified/addlemontofish.json) | Trained | 6 | retrieve the lemon_wedge from the fridge and place it on the fish_plate with the fish on the dining counter. |
| [AddSugarCubes](../data_generation/task_level/tasks/specs/verified/addsugarcubes.json) | Trained | 1 | place both sugar_cube_1 and sugar_cube_2 onto the cake_plate which contains the cake on the dining_counter. |
| [AlcoholServingPrep](../data_generation/task_level/tasks/specs/verified/alcoholservingprep.json) | Trained | 16 | Move the alcohol and the cup from the cabinet to the dining counter with the decoration on it; open the cabinet first if it is closed. |
| [AlignSilverware](../data_generation/task_level/tasks/specs/verified/alignsilverware.json) | Trained | 4 | place the fork on the left side of the plate and the spoon on the right side of the plate on the dining_counter. |
| [ArrangeBreadBowl](../data_generation/task_level/tasks/specs/verified/arrangebreadbowl.json) | Held out | 8 | pick up the heated bread from the toaster oven, place it in the bowl on the counter, then move the bowl containing both breads to the dining counter. |
| [ArrangeDrinkware](../data_generation/task_level/tasks/specs/verified/arrangedrinkware.json) | Trained | 4 | Place the pitcher and cup on the dining counter for serving. |
| [ArrangeTea](../data_generation/task_level/tasks/specs/verified/arrangetea.json) | Trained | 2 | Pick the kettle from the counter and place it on the tray. Then pick the mug from the cabinet and place it on the tray. Finally, close the cabinet doors. |
| [BeverageOrganization](../data_generation/task_level/tasks/specs/verified/beverageorganization.json) | Held out | 4 | Move the drinks to the dining counter. |
| [BowlAndCup](../data_generation/task_level/tasks/specs/verified/bowlandcup.json) | Trained | 4 | place the cup inside the bowl on the dining table, then move the bowl (containing the cup) to the kitchen counter. |
| [CandleCleanup](../data_generation/task_level/tasks/specs/verified/candlecleanup.json) | Trained | 8 | Pick the two decorations from the dining table and place them in the cabinet, opening it first if it is closed, then leave the cabinet closed. |
| [CheeseMixing](../data_generation/task_level/tasks/specs/verified/cheesemixing.json) | Trained | 3 | Put both the cheese and the spatula in the pot filled with milk. Placing the spatula in the pot counts as stirring. |
| [ClusterItemsForClearing](../data_generation/task_level/tasks/specs/verified/clusteritemsforclearing.json) | Held out | 1 | Move the two items to cluster with the anchor item on the dining counter for efficient clearing. |
| [ColorfulSalsa](../data_generation/task_level/tasks/specs/verified/colorfulsalsa.json) | Trained | 1 | Place the avocado, onion, tomato, and bell pepper on the cutting board. |
| [CondimentCollection](../data_generation/task_level/tasks/specs/verified/condimentcollection.json) | Trained | 2 | Pick the condiments from the counter and place them in the cabinet. |
| [CutBuffetPizza](../data_generation/task_level/tasks/specs/verified/cutbuffetpizza.json) | Trained | 6 | Take the pizza cutter from the drawer and place it on the dining counter for cutting. |
| [DateNight](../data_generation/task_level/tasks/specs/verified/datenight.json) | Trained | 8 | Get the decoration and the alcohol from the cabinet and move them to the dining counter. |
| [DeliverStraw](../data_generation/task_level/tasks/specs/verified/deliverstraw.json) | Trained | 6 | Take a straw from the drawer and place it inside the glass cup on the dining counter. |
| [DessertAssembly](../data_generation/task_level/tasks/specs/verified/dessertassembly.json) | Trained | 1 | place the dessert_bowl (containing dessert1) and the cupcake onto the tray on the counter. |
| [DisplayMeatVariety](../data_generation/task_level/tasks/specs/verified/displaymeatvariety.json) | Trained | 6 | Retrieve the chicken, shrimp, and beef from the fridge and place them on the serving tray on the dining counter. |
| [DistributeChicken](../data_generation/task_level/tasks/specs/verified/distributechicken.json) | Held out | 3 | Take the chicken drumsticks from the pan and place one on each plate on the dining counter. |
| [GarnishCake](../data_generation/task_level/tasks/specs/verified/garnishcake.json) | Held out | 1 | Place exactly one cherry on either the cake or cake_plate. Place exactly one strawberry on the cake_plate in order to position it beside the cake. |
| [GarnishCupcake](../data_generation/task_level/tasks/specs/verified/garnishcupcake.json) | Trained | 8 | retrieve cinnamon from cabinet and place it next to cupcake_plate on dining_counter, then place chocolate onto cupcake_plate. |
| [GarnishPancake](../data_generation/task_level/tasks/specs/verified/garnishpancake.json) | Trained | 6 | retrieve the strawberry from the fridge and place it on top of the pancake which is on a plate on the dining counter. |
| [GatherMarinadeIngredients](../data_generation/task_level/tasks/specs/verified/gathermarinadeingredients.json) | Trained | 12 | Retrieve the bottle and the shaker from the cabinet and place them next to the mixing bowl on the counter. Then add garlic from the fridge to the mixing bowl. |
| [HotDogSetup](../data_generation/task_level/tasks/specs/verified/hotdogsetup.json) | Held out | 60 | place the hotdog_bun and sausage on the plate on the dining_table, and place the condiment bottle next to the plate. |
| [LemonSeasoningFish](../data_generation/task_level/tasks/specs/verified/lemonseasoningfish.json) | Trained | 6 | Retrieve a lemon from the fridge and place it on the cutting board with the fish. |
| [LineUpCondiments](../data_generation/task_level/tasks/specs/verified/lineupcondiments.json) | Trained | 16 | Take two condiments from the cabinet and move them on the counter near the stove, lined up next to each other. |
| [MatchCupAndDrink](../data_generation/task_level/tasks/specs/verified/matchcupanddrink.json) | Trained | 4 | Place the wine bottle next to the wine glass and the juice bottle next to the glass cup on the dining counter. |
| [MeatSkewerAssembly](../data_generation/task_level/tasks/specs/verified/meatskewerassembly.json) | Held out | 1 | Move the kebab skewers from the plate to the oven tray for oven preparation. |
| [OrganizeCondiments](../data_generation/task_level/tasks/specs/verified/organizecondiments.json) | Trained | 2 | Move the three condiments from the counter into the cabinet. |
| [PlateStoreDinner](../data_generation/task_level/tasks/specs/verified/platestoredinner.json) | Trained | 14 | place one steak on the plate and the other steak in the bowl, then store the bowl in the fridge. |
| [PortionHotDogs](../data_generation/task_level/tasks/specs/verified/portionhotdogs.json) | Trained | 1 | place exactly one hotdog_bun and one sausage on plate1, and exactly one hotdog_bun and one sausage on plate2, all retrieved from the bowl on the dining_counter. |
| [PortionYogurt](../data_generation/task_level/tasks/specs/verified/portionyogurt.json) | Trained | 1 | place exactly two yogurt containers on plate1 and exactly two yogurt containers on plate2. |
| [PrepareCheeseStation](../data_generation/task_level/tasks/specs/verified/preparecheesestation.json) | Held out | 12 | Pick the cheese from the fridge and the cheese grater from the cabinet and place them next to the salad bowl on the counter. |
| [PrepareCocktailStation](../data_generation/task_level/tasks/specs/verified/preparecocktailstation.json) | Trained | 16 | Place the lemon wedge in the bowl on the dining counter, then place the liquor bottle and glass cup next to the bowl. |
| [PrepareCoffee](../data_generation/task_level/tasks/specs/verified/preparecoffee.json) | Trained | 6 | Pick the mug from the cabinet, place it under the coffee machine dispenser, and press the start button. |
| [PrepareDrinkStation](../data_generation/task_level/tasks/specs/verified/preparedrinkstation.json) | Trained | 8 | Prepare a drink station by placing the cup and mug on the tray and the pitcher next to the tray on the dining counter. |
| [PrepareSandwichStation](../data_generation/task_level/tasks/specs/verified/preparesandwichstation.json) | Held out | 6 | retrieve ingredient_bowl and baguette from ingredient_source_fixture, then place both next to toaster_oven on staging_surface to stage them near the toaster oven. |
| [PrepareSausageCheese](../data_generation/task_level/tasks/specs/verified/preparesausagecheese.json) | Trained | 6 | Retrieve the sausage and cheese from the fridge and place them on the cutting board on the counter. |
| [PrepareSoupServing](../data_generation/task_level/tasks/specs/verified/preparesoupserving.json) | Trained | 16 | Move the ladle from the cabinet to the pot, opening the cabinet first if it is closed. Leave the cabinet closed at the end. |
| [SeasoningSteak](../data_generation/task_level/tasks/specs/verified/seasoningsteak.json) | Trained | 8 | Retrieve the shaker from the cabinet and place it next to the steak on the dining counter. |
| [ServeMealJuice](../data_generation/task_level/tasks/specs/verified/servemealjuice.json) | Held out | 4 | Pick up the orange juice cups from the counter and place one next to each plate on the dining counter. |
| [ServeSteak](../data_generation/task_level/tasks/specs/verified/servesteak.json) | Trained | 3 | Pick up the pan with the steak in it and place it on the dining table. Then, place the steak on the plate. |
| [SetBowlsForSoup](../data_generation/task_level/tasks/specs/verified/setbowlsforsoup.json) | Trained | 8 | Move the bowls from the cabinet to the plates on the dining table. |
| [SetupBowls](../data_generation/task_level/tasks/specs/verified/setupbowls.json) | Trained | 8 | Pick up the bowls from the cabinet and place each bowl in front of a stool on the dining counter. |
| [SetupButterPlate](../data_generation/task_level/tasks/specs/verified/setupbutterplate.json) | Trained | 16 | retrieve butter from fridge and place it on plate, then retrieve butter_knife from counter and place it next to plate on dining_counter. |
| [SetupSodaBowl](../data_generation/task_level/tasks/specs/verified/setupsodabowl.json) | Trained | 6 | Grab two sodas from the fridge and place both inside the bowl with ice on the dining counter; open the fridge first if it is closed. |
| [SetUpSpiceStation](../data_generation/task_level/tasks/specs/verified/setupspicestation.json) | Trained | 16 | Collect the spice, bottle, and shaker from the cabinet and place them next to the stove. |
| [SetupWineGlasses](../data_generation/task_level/tasks/specs/verified/setupwineglasses.json) | Trained | 4 | Retrieve the wine glasses from the counter and place one next to each plate on the dining counter. |
| [SpicyMarinade](../data_generation/task_level/tasks/specs/verified/spicymarinade.json) | Trained | 2 | Place the bowl and condiment from the cabinet on the counter, opening the cabinet first if it is closed. Then place the lime and garlic on the cutting board. |
| [SweetenCoffee](../data_generation/task_level/tasks/specs/verified/sweetencoffee.json) | Trained | 6 | Take the sugar cube from the saucer and place it inside the coffee mug. Then, get the milk from the fridge and place it next to the coffee on the counter. |
| [TongBuffetSetup](../data_generation/task_level/tasks/specs/verified/tongbuffetsetup.json) | Trained | 6 | Take the tongs from the drawer and place them on the dining counter next to the tray containing the food. |
| [VeggieDipPrep](../data_generation/task_level/tasks/specs/verified/veggiedipprep.json) | Trained | 1 | place the cucumber, carrot, and bowl onto the tray on the counter. |
