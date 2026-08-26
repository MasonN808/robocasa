# Fixed Live-Sim Evaluation and Initial Training Ablations

## Objective

Build one configuration-first, model-independent live-simulator cohort for fair
comparison of the current 27/3 and 90/10 SFT models and future training
ablations. Keep the existing dataset-derived evaluations as a separate
historical surface.

The benchmark must not score a model on simulator facts hidden by an overly
coarse symbolic initial state. Before freezing the cohort, expand safe access
state randomization and audit whether an agent initialized at a broad work
surface physically blocks an exclusive colocated appliance or sub-fixture.

## Terminology

- `train_task_types`: task types included in SFT training (47 current tasks).
- `heldout_task_types`: task types excluded from SFT training (6 current tasks).
- **Configuration**: task, full symbolic initial state, coordinator, scene, and
  simulator/environment seeds. It excludes expert actions, communication
  wording, and work assignment.
- **Deterministic rollout**: temperature 0, sampling disabled, rollout seed 0.
- **Stochastic repeat**: the same configuration evaluated with sampling
  enabled, one fixed nonzero temperature, and a distinct decoder seed.
- **Carrier trajectory**: a saved expert sequence used to ground/prune and
  prove that a configuration has an executable solution. The evaluated model
  never receives the carrier's plan or actions.

An episode is not an expert trajectory target. It is a live-sim configuration
plus a frozen decoding policy and rollout seed on which the model produces its
own trajectory.

## Preconditions before cohort construction

### Safe access-state randomization

Extend the global initial-state sampler—not per-task handwritten variants—to
randomize open/closed access state for:

- hinged cabinets;
- refrigerator doors;
- drawers/sliding access parts.

A fixture/part is eligible when:

1. Its open/closed state exists in the symbolic initial state.
2. The FSM models that state and the relevant transition.
3. The task exposes the exact valid opening action for that fixture/part.
4. Changing only that access state neither satisfies the task goal nor makes
   the task logically impossible.

No existing expert replay is required to declare such a state eligible: an old
trajectory that assumed an open fixture would naturally lack the newly needed
opening call. The generator must instead see the sampled state and produce a
new trajectory, which is then validated by the ordinary concurrent FSM and may
later receive live-sim oracle QA.

Do not randomize appliance operating states, object locations, held objects, or
food/preparation states in this phase. For example, starting a coffee machine
already running could violate prerequisites or partially satisfy its goal.

Record state factors, not a permanent `out_of_distribution` boolean. At report
time, compare each episode's factors with the evaluated model's training
manifest and label exact-configuration and factor-value coverage separately.
Closed cabinets/fridges/drawers are currently unseen by the existing models,
but will be in-distribution for future models trained with this sampler.

### Access-state prompt consistency

Before using randomized access states for generation or evaluation, remove or
make conditional any task prose that hardcodes an initial fixture state. The
structured `initial_state` is canonical. A task must never simultaneously say,
for example, `cabinet = closed` in `initial_state` and “the cabinet is already
open” in its task-specific instructions or examples.

The current specifications requiring correction include at least
`CondimentCollection`, `OrganizeCondiments`, and `SetUpSpiceStation`, whose
extra execution rules explicitly say that the cabinet is already open. Audit
implicit assumptions in task goals and examples as well, including wording
such as “place the objects in the open cabinet.” Replace initial-state claims
with concise conditional guidance, for example: “Check the cabinet state in
`initial_state`; open it before access if it is closed.” Preserve genuine final
goal requirements separately, such as requiring a cabinet to be closed after
objects are placed inside.

Existing models have useful cross-task supervision for access transitions—the
current rendered pool contains many cabinet/fridge opening examples, drawer
opening examples, and some cabinet-closing examples—but generally does not
show both initial states for the same task. Treat performance on newly sampled
opposite-state configurations as cross-task state generalization and report it
separately from configurations covered by that model's training manifest.

### Exact prompt parity and the default state-grounded interface

Training preprocessing and live evaluation must call one shared prompt builder
with the same canonical inputs. For every frozen evaluation carrier, add a
parity test that compares the complete system message, user message, function
schemas, coordinator, agent-local history, observation labels, and no-index
mode byte for byte at an equivalent decision point. Also compare the final
post-chat-template text and token IDs; matching pre-template dictionaries is
not sufficient if preprocessing and inference invoke the template differently.
In particular, both paths must use the verified task goal as the task
instruction; preprocessing must not
read the natural-language instruction from `metadata.json` while evaluation
uses the composite-task name from `original_trajectory.json`.

The default SFT/live-evaluation interface is the state-grounded, generic-tool
design:

1. Include the verified task goal and the full symbolic initial state. This
   exposes initial agent locations, object/support relationships, fixture-part
   states, held objects, and machine state explicitly; later changes remain
   available only through that agent's observations, messages, private action
   history, and reported failures.
2. Reuse and enrich `TASK_LEVEL_ALLOWED_TOOL_SPECS` as the same global
   model-facing tool set for every task; do not introduce a second registry.
   Each function schema includes
   only its name, general description, required arguments, optional arguments,
   argument types, and structural requirements such as one-of groups.
3. Do not expose task-specific allowed-ID enums, object/source pairings,
   destination allowlists, or a task-specific subset of tool names. The shared
   concurrent FSM and task validator continue enforcing those constraints
   internally, so an invalid selection is a policy error rather than an option
   removed from the model's decision space.
4. Do not duplicate tool schemas in an `Available tools` prose block. Pass the
   canonical global schemas once through the backend interface. Qwen's chat
   template supplies its `<tools>` and `<tool_call>` syntax; Gemini uses native
   function calling. Backend-specific serialization must not be written into
   the canonical user prompt.
5. Keep `get_image`, `communicate`, `wait_for_signal`, and the physical tools in
   the same global interface. Continue omitting model-predicted global
   completion from partial-observation evaluation because the FSM owns episode
   termination.

Future tasks may require explicit symbolic sub-sites even though none of the
current 53 verified tasks does. Examples include selecting a particular stove
burner, left/right sink basin, oven rack, refrigerator shelf, or toaster slot.
Do not merely begin emitting `source_site_id` or `target_site_id`: first expose
stable site keys in the symbolic initial state, track each object's
`location_site` in the FSM, add site-aware goal conditions, map those stable
keys to scene-specific simulator regions, generate site-grounded expert calls,
and verify exact train/eval prompt parity across layouts. Until then, current
tasks omit site arguments and the executor's inferred/default physical site is
an internal placement detail.

Before enabling this interface, audit the union of all declared task tools for
conflicting required/optional arguments, argument types, and descriptions.
There must be exactly one canonical schema per tool name. Validate every
task-specific expert call against the new global schema, then validate it again
against the hidden task-specific FSM contract. A schema that is globally valid
but task-invalid must produce the ordinary `report_failed` path during live
evaluation.

The earlier visual/history-only plus constrained-schema interface remains a
historical control, not the default. If resources permit, use the original 2x2
state/schema experiment only as a diagnostic after the default interface is
working. The primary new model is full initial state plus generic global tools.
Compare it on the same trajectories, checkpoint family, optimizer settings,
decoding contract, and fixed live-sim episodes. Report overall and per-split
success, first-rejection counterfactual success, navigation/location
rejections, invalid-ID/tool selections, redundant open/close calls, and seen
versus held-out access-state configurations.

### Colocated access-blocking audit

Before adding any prompt or FSM field, build a simulator-backed audit over
unique physical spawn configurations:

1. Identify exclusive appliances/sub-fixtures spatially colocated with an
   agent's broad starting fixture (for example, a toaster oven on a counter).
   Do not test unrelated fixtures merely because they appear in the task; an
   agent at a counter is not presumed to block a refrigerator elsewhere.
2. Attempt navigation to the exclusive fixture with the other agent present.
3. Reset, move the suspected blocker away, and retry.
4. Reset, move only the attempted navigator away while leaving the suspected
   blocker, and retry.
5. Classify blocking only when removing the other agent makes navigation
   succeed and moving the navigator alone does not.
6. Save initial and post-probe top/room/map views, robot coordinates, resolved
   fixture IDs, navigation outcomes, and give-space outcomes.
7. Build an HTML review artifact with one entry per detected unique physical
   configuration.

The user manually reviews every automatically detected case. Only confirmed
cases enter the state schema. Manipulation success is not used to infer
interaction readiness because executor actions may reposition the robot.

Use one field only:

```json
"interaction_anchor": "toaster_oven"
```

Its semantics are intentionally narrow:

- the anchored agent is already positioned to interact with that exclusive
  fixture, even when its broad `location` is the parent counter;
- another agent cannot navigate to or interact with the anchored fixture until
  the anchored agent yields it;
- the anchored agent may still explicitly navigate to its current anchor; this
  is a valid redundant repositioning and preserves the anchor;
- `give_space` may name either the agent's `interaction_anchor` or its broad
  `location`; yielding the anchor clears the anchor, while yielding the broad
  location also clears any anchor nested at that location;
- navigating elsewhere clears the old anchor.

Do not infer this field from a task name or broad symbolic location alone.
Derive it from reviewed simulator geometry keyed by the task, scene, simulator
seed, and joint robot-start assignment. For `ArrangeBreadBowl`, audit every
supported scene/seed combination before deciding whether the observed
`(counter, counter) -> agent_1 anchored at toaster_oven` mapping is universal
for that task configuration or scene-specific.

After manual approval, implement this identically in generated initial state,
prompt/training/evaluation context, concurrent FSM, shared scheduler,
`give_space`, live evaluation, and tests. The prompt currently contains the
initial-state dictionary directly; add only one concise canonical instruction
explaining the field's operational meaning.

### Atomic physical handovers and validation gating

Treat every tick as atomic for physical workspace ownership. If one agent
occupies an exclusive fixture at the start of a tick, another agent may not
navigate to or act at that fixture in the same tick that the occupant calls
`give_space`. The departure takes effect only after that tick; entry becomes
legal on the following tick. Enforce this in the canonical shared scheduler so
generation validation, postprocessing validation, concurrent FSM replay, and
live evaluation cannot disagree because of within-tick execution order.

The current rendered 27/3 and 90/10 pools contain at least one confirmed
counterexample: `arrange_bread_bowl/traj_000024` simultaneously navigates one
agent to the toaster while its occupant calls `give_space`. It is marked
`validation.is_valid = false` in both rendered pools but was nevertheless used
for training by the current 27/3 and 90/10 one- and three-epoch SFT models. It
contributed 55 agent-turn examples to each model. Report these checkpoints as
trained with known invalid-carrier contamination until clean replacements are
selected and new models are trained.

Validation metadata must be a hard pipeline gate, not merely diagnostic
annotation. Dataset selection, rendering, preprocessing, and training must
refuse every record whose current validation does not have `is_valid == true`.
After adding the atomic handover rule, revalidate all 5,300 source trajectories,
but do not replace or replay the 408 configuration carriers that already passed
the completed all-task live-sim audit. Try alternate carriers only for the one
failed configuration currently represented by
`arrange_bread_bowl/traj_000024`, and rerun live-sim only for that configuration
until one clean carrier passes or its existing alternates are exhausted. If the
new symbolic rule unexpectedly rejects a carrier that already passed live sim,
investigate that disagreement as a validator regression rather than silently
swapping the carrier or expanding the live audit.

Before bulk revalidation, make the shared tick transition transactional for
all FSM preconditions and effects, not only physical handovers. Every agent's
call is validated against one frozen beginning-of-tick state; no call may rely
on a cabinet being opened, an object being placed, an appliance being changed,
or another prerequisite being created by its partner during that same tick.
Only after the complete batch passes precondition and cross-effect checks are
accepted effects committed. Offline concurrent validation and live evaluation
must call this same transactional implementation.

After revalidating the 5,300 trajectories with the atomic-handover and general
transactional fixes, group newly invalid trajectories by root cause, task, and
tick pattern. For each recurring pattern, inspect the canonical prompt and
task-specific facts before regenerating data. If the required ordering is
missing or ambiguous, add one concise general or task-specific clarification
and a focused validator-backed example. If the prompt already states the rule
clearly, treat the invalid trajectory as generator noncompliance and rely on
targeted retry feedback rather than adding more prompt rules. Record this
decision so prompt complexity does not grow automatically with every rejected
trajectory.

## Fixed cohort contract

The cohort is configuration-first and may contain more than 20 total episodes
for tasks with more than 20 feasible configurations.

For each task:

1. Include every known feasible unique configuration once as a deterministic
   rollout (`rollout_seed=0`, temperature 0).
2. If fewer than 20 unique configurations exist, add stochastic repeats,
   distributed as evenly as possible across configurations, until the task has
   20 balanced-report episodes.
3. All stochastic repeats use one fixed nonzero temperature and distinct
   decoder seeds. Do not vary temperature across repeats; that would confound
   sampling variation with distribution sharpness.
4. If more than 20 feasible configurations exist, retain all of them in the
   all-configuration metric and choose a deterministic, stratified subset of 20
   for the balanced metric. Do not discard the remaining configurations from
   evaluation.

Every episode record freezes:

- stable `episode_id`, configuration ID, and within-task rank;
- membership in the all-configuration and balanced-20 metrics;
- task and split;
- full symbolic initial state and grounding/pruning carrier;
- access-state factors and approved `interaction_anchor` facts;
- coordinator assignment;
- layout, style, environment seed, task/sample seed, rollout seed, temperature,
  and whether sampling is enabled;
- evaluator, prompt, contention-policy, and initial-state-schema versions;
- configuration and scene signatures;
- carrier sequence signature and validation provenance;
- future-compatible scene membership.

The current scene remains layout 11, style 34, and environment seed 42 because
those produced the current training visuals. Fields remain episode-local so a
future cohort can mix scenes without redesigning the manifest.

## Cohort construction

Revise `training/bc_task_vlm/fixed_live_sim_cohort.py` into a configuration-first,
resumable builder.

Inputs:

- the full generated/rendered carrier pool;
- train/held-out task selection plus an audit for unassigned verified tasks;
- reviewed access-blocking annotations;
- access-state sampling policy;
- balanced-report target (default 20);
- cohort and stochastic decoding seeds;
- scene specification;
- output/shard/resume options.

Construction rules:

1. Enumerate configurations before sampling carrier trajectories.
2. Normalize and hash task-relevant initial state, approved blocking facts,
   scene, coordinator, and simulator seeds.
3. Group all carriers by configuration and retain their distinct action-sequence
   signatures.
4. For existing configurations, search carriers until at least one passes live
   oracle replay under the shared concurrent scheduler. A failed carrier does
   not invalidate a configuration if another sequence for it passes.
5. For newly randomized access states, generate new solutions rather than
   replaying unchanged old trajectories; validate them with the concurrent FSM
   and later live-sim QA.
6. Cover every known feasible configuration before adding repeats.
7. When adding repeats, prefer additional passing, structurally distinct
   carrier sequences for provenance, but remember that carrier identity alone
   does not vary model behavior.
8. Require zero harness errors, zero rejected oracle calls, and FSM goal success
   for accepted carriers. Record native success separately but do not require
   it.
9. Freeze the manifest independently of trained-model outcomes.

Produce a candidate ledger, configuration ledger, carrier-attempt ledger,
structured rejection ledger, reviewed blocking ledger, final manifest, summary,
and content hash.

## Seeded decoding

Wire rollout seeds into the decoder; storing a seed as an index is insufficient.

- At temperature 0, sampling is disabled and seed has no effect.
- At fixed temperature greater than 0, seed initializes the random generator
  and selects a reproducible sample from the same distribution.
- Different seeds can produce different tool trajectories but are not
  guaranteed to do so when the distribution is highly concentrated.

For local HF and vLLM, derive a stable call seed from the episode rollout seed,
agent ID, and local turn so worker scheduling does not change outputs. For
Gemini or another API without guaranteed seed control, record that the rollout
is stochastic but not exactly reproducible; do not pretend its seed is
equivalent to local seeded decoding.

## Fixed-cohort evaluation

The evaluator supports:

- one immutable fixed-cohort manifest;
- split selection (`train_task_types` or `heldout_task_types`);
- all episodes, all deterministic configurations, balanced-20 membership, or a
  deterministic per-task prefix for cheap screening;
- optional task/shard filters;
- backend/model/adapter arguments and safe resume into a distinct
  model/cohort/split output directory.

The evaluator verifies manifest hash and refuses incompatible step indexing,
prompt contract, scene, initial-state schema, success criterion, decoding
settings, or contention policy. Current evaluations use private partial history,
no step indices, consume-once observations, FSM success, uniform durations, and
the canonical concurrent-FSM contention classifier.

## Metrics and confidence intervals

Revise `training/bc_task_vlm/summarize_fixed_live_sim.py` to report two primary
views from the same evaluation run.

### All-configuration deterministic result

- one temperature-0 rollout for every known feasible configuration;
- configuration success and coverage by task;
- macro-by-task is primary because task configuration counts differ;
- micro success is also reported with its exact denominator.

### Balanced 20-episode result

- exactly 20 marked episodes per task;
- all unique configurations plus seeded repeats when a task has fewer than 20;
- a deterministic stratified subset when a task has more than 20;
- macro and micro success.

Additional metrics:

- deterministic seed-0 versus stochastic-repeat success;
- deterministic failures recovered by at least one stochastic repeat;
- configurations that always fail or always succeed;
- seen versus currently unseen access-state factors, computed relative to each
  model's training manifest;
- exact-configuration coverage;
- paired model differences on identical episode IDs;
- first-rejection-cutoff success, rejection incidence, resource conflicts,
  termination reasons, partial-goal fraction, and harness-error rate.

Primary 95% intervals use hierarchical bootstrap (tasks, then episodes within
task) and paired hierarchical bootstrap for matched models. Wilson episode
intervals remain descriptive references. The six held-out task types still
limit inference about the population of unseen tasks.

## Training launcher and initial ablations

Retain explicit `NUM_EPOCHS` and `LEARNING_RATE` overrides, unique output/W&B
names for non-default cells, four-B200 effective batch, mandatory online W&B,
and the no-index partial-observation contract.

Initial matrix:

| Dataset | Epochs | LR | Status | Estimated four-GPU time |
|---|---:|---:|---|---:|
| 27/3 | 3 | 2e-4 | existing | 4h19m measured |
| 90/10 | 3 | 2e-4 | existing | 13h25m measured |
| 27/3 | 1 | 2e-4 | new/independent of cohort revision | about 1h27m |
| 90/10 | 1 | 2e-4 | new/independent of cohort revision | about 4h28m |

The existing one-epoch and three-epoch jobs remain historical diagnostics, but
their checkpoints were trained with task-constrained tool schemas and without
the symbolic initial state. They cannot serve as the primary models for the new
interface. Future training data generation must use the same access-state
randomization and approved blocking annotations; existing validated action
trajectories may be reused because the SFT prompt/interface is constructed at
preprocessing time.

## Revised execution order

1. Preserve current candidate/oracle outputs as diagnostics. Stop or hold only
   obsolete downstream fixed-cohort model evaluations; allow approved training
   ablations to continue.
2. Implement the default state-grounded interface: one canonical verified goal,
   full symbolic initial state, one global generic tool schema set with no
   task-specific enums or tool filtering, and no duplicate prose tool list.
   Fix and test byte-identical SFT/live-evaluation prompt construction.
   Freeze prompt snapshots for representative cabinet, drawer, refrigerator,
   appliance, and object-placement tasks.
3. Audit the global schema union for conflicting argument contracts; validate
   every existing expert call first structurally against the global schemas and
   then semantically against its hidden task FSM. Verify Qwen and Gemini use
   their own backend-native tool-call serialization without canonical prompt
   markup.
4. Implement and test global cabinet/fridge/drawer access-state sampling, then
   audit task goals, task-specific rules, and examples for hardcoded initial
   access states. Make those statements conditional on the canonical
   `initial_state` before generating or evaluating randomized configurations.
5. Build the colocated access-blocking audit and its visual artifact without
   yet changing prompts, FSM behavior, or data schema.
6. User reviews every detected blocking case.
7. Implement approved `interaction_anchor` facts end to end and validate shared
   offline/live semantics.
8. After the all-task first-pass audit completes, add the atomic same-tick
   `give_space`/exclusive-entry rule and general transactional pre-tick
   validation to the canonical shared scheduler. Make
   `validation.is_valid == true` mandatory in selection, rendering,
   preprocessing, and training, and revalidate the complete source pool.
   Preserve the 408 carriers that already passed live sim. Try alternate
   carriers and repeat live-sim only for the failed ArrangeBreadBowl
   configuration; treat any new symbolic-versus-live disagreement among the
   408 passing carriers as a validator regression to investigate.
   Classify newly invalid trajectories by root cause and determine whether
   each recurring pattern needs clearer prompt/task wording or only targeted
   retry feedback before generating replacements.
9. Generate supplemental demonstrations for every task/configuration newly
   introduced by randomized cabinet/fridge/drawer states. Use the deterministic
   balanced joint sampler: enumerate valid agent-start assignments and all
   eligible open/closed access-state combinations, shuffle their order once per
   task with a stable seed, pair each physical configuration with every
   coordinator assignment, and cycle through their Cartesian product. For N
   requested runs and K valid configurations, every configuration must appear
   either `floor(N/K)` or `ceil(N/K)` times. Keep structured-random language
   variation and coordinator sampling independent. Reuse unaffected existing
   carriers only after the new configurations are represented; revalidate and
   rerender every genuinely new or replacement trajectory. Do not launch paid
   generation until the 150-run preflight manifest passes this balance check.
10. Inventory all configurations and passing carriers. Build a small revised
   smoke cohort that includes open/closed access states, a confirmed blocking
   case, stochastic repeats, and a task with more than 20 configurations.
11. Smoke-test oracle validation, deterministic/stochastic decoding,
   reproducibility, resume, configuration coverage, and both metric views.
12. Build and freeze the full revised cohort configuration-first.
13. Re-preprocess the validated trajectories with the new interface, train the
    primary full-state/global-tools model, and evaluate it on the frozen cohort.
    Treat the old constrained/no-state checkpoints as historical controls.
14. Optionally run the remaining cells of the 2x2 state visibility x tool-schema
    constraint matrix after the primary model establishes that the new default
    works.
15. Evaluate existing 27/3 and 90/10 three-epoch checkpoints and completed
    one-epoch ablations on both splits. Use a predeclared fixed subset for early
    cost screens; do not set outcome-dependent thresholds after viewing results.
16. Add base Qwen 8B and Gemini 3 Flash after any required external-API
    approval.
17. Summarize and rebuild the fixed-cohort artifact automatically.

### Transactional pool revalidation result (2026-08-15)

The first read-only revalidation of all 5,300 selected trajectories under
`transactional_tick_v1` accepted 5,242 and rejected 58.  Every rejection has
the same semantic cause: one agent enters or uses an exclusive fixture during
the same tick in which its beginning-of-tick occupant calls `give_space`.
Because departure is committed at the end of the atomic tick, entry is legal
only on the following tick.  The failures span 19 tasks; the largest group is
`prepare_soup_serving` (11), followed by `organize_condiments` (7).

This is a recurring, understandable protocol gap rather than 58 unrelated
schema errors.  Before replacement generation, add one concise canonical rule
and example saying that `give_space` frees the resource on the next tick, then
use targeted retry feedback that identifies the actual agents, fixture, and
tick.  Do not add task-specific wording for this cross-task rule.  Keep the
source pool immutable; the audit ledger is stored under
`training/bc_task_vlm/eval_runs/transactional_pool_revalidation_v1/`.

### State-grounded generation preflight (2026-08-16)

The obsolete queued ArrangeBreadBowl evaluation jobs `388462` and `388463`
were cancelled. The production partial-history default is now explicitly
`none`, and preprocessing artifacts plus live-evaluation metrics record prompt
contract `state_grounded_global_tools_no_index_v1`.

The 150-run dry-run manifest covers 53 tasks and is balanced for every task.
Thirty-three tasks expose at least one safely randomizable cabinet,
refrigerator, or drawer access part. This includes the three
`cabinet_double_door` tasks that were initially missed by an overly narrow
fixture-type matcher. Including coordinator assignment, `PrepareCoffee` has 8
joint configurations and `HotDogSetup` has 112; at 150 runs each valid configuration
appears within one count of every other configuration for its task. The frozen
manifest is
`training/bc_task_vlm/reports/new_access_state_generation_preflight/initial_config_manifest_150.json`.

All 5,300 trajectories in the legacy `v1` selected source pool are
structurally valid under the global model-facing tool schemas. Retrospective
current-FSM revalidation accepts 5,099 and rejects 201: 56 atomic same-tick
resource handovers, one held-object error, and 144 completion-tail protocol
errors. This is a diagnostic of the pre-repair source pool, not the final clean
dataset. The replacement dataset
`tick53x100_from150_partition_none_temp06_v4_transactional` contains 5,300 of
5,300 trajectories marked valid under `transactional_tick_v1`. New generation
must likewise satisfy the current validator. The legacy audit ledger is under
`training/bc_task_vlm/eval_runs/global_tool_interface_audit_v1/`.

### Frozen configuration targets and all-configuration canary (2026-08-17)

The corrected sampler exposes 824 distinct joint configurations across 53
tasks after pairing valid ordered agent starts, eligible access-state
combinations, and both coordinator assignments. Configuration selection was
frozen before generation or model evaluation at
`training/bc_task_vlm/eval_manifests/fixed_live_sim_state_grounded_v1/configuration_targets.json`.
It contains an 824-episode deterministic all-configuration view and a separate
balanced 20-episode/task view. Carrier and scene bindings remain explicitly
pending until generation and oracle replay make the manifest executable.

Slurm array `391162` attempts every one of the 824 configurations exactly once.
Full 150/task generation is blocked until
`training.bc_task_vlm.verify_all_configs_canary` confirms complete run-index
coverage, exact configuration identity, global tool-schema validity, and
current concurrent-FSM validity for all outputs. The gate must also be combined
with the no-index SFT/live prompt-parity tests before full generation is
submitted.

### Canonical cabinet-parent workspace revision (2026-08-19)

The 824-target manifest predates the approved cabinet workspace semantics and
is obsolete. Cabinets and their declared parent counters are one agent
workspace: new initial-position sampling and model-facing navigation use the
parent, while legacy `location=cabinet`, `navigate_to_fixture(cabinet)`, and
`give_space(cabinet)` remain accepted aliases. Object containment is unchanged;
an object inside a cabinet is not on its parent counter.

Production behavior must satisfy all of the following before a replacement
cohort is frozen:

- cabinets are shared and non-front-facing;
- cabinet navigation physically routes to the parent and never silently moves
  a partner;
- cabinet interactions are accepted from either alias;
- cabinet door transitions cannot overlap another agent's cabinet-content
  access, although distinct objects in an already-open cabinet may be accessed
  concurrently;
- an agent at an exclusive child appliance may manipulate its parent surface
  and objects on it without physical movement or a symbolic location change;
- configuration signatures canonicalize agent cabinet locations to parents in
  generation, canary verification, and every fixed-cohort builder;
- the manifest stores every canonical configuration exactly once.

The first production manifest under this policy contains 768 joint
configurations. This differs from the earlier read-only projection of 712
because 13 newly declared cabinet-parent workspaces now legitimately allow the
previously absent case where both agents start at the shared parent. The old
824 manifest contained 112 cabinet/parent alias duplicates and also omitted
those 56 valid shared-parent starts.

The existing 5,300-trajectory selected pool contains 1,734 trajectories across
20 tasks with cabinet navigation or cabinet-specific handover behavior (2,656
navigations, 1,259 give-space calls, and 1,157 cabinet waits). Treat these as
affected by the policy change. Simple cabinet navigation can be canonicalized,
but obsolete waits/handover tails must not be silently rewritten; regenerate or
explicitly retain them only after measuring the resulting supervision policy.

The replacement cohort exposes two runtime evaluation modes. `sampled` is the
default and runs 10 episodes per task unless `--episodes-per-task` changes the
count. It samples by reproducible uniformly shuffled passes, so every available
configuration appears before any repeats. `full_config` runs every configuration
once by default; `--episodes-per-config` requests intentional labelled repeats,
and `--max-configs-per-task` selects a reproducible limited subset. Duplicate
configurations never enter the underlying configuration universe.

## Reporting artifacts

Maintain three durable artifacts:

1. Access-blocking audit: simulator evidence and manual decisions for every
   detected physical configuration.
2. Fixed-cohort model/ablation comparison: sampled and full-configuration
   views, clustered CIs, paired differences, model-relative training coverage,
   and cohort provenance.
3. Historical dataset-derived 27/3 and 90/10 report: current split-specific and
   out-of-box results, explicitly separate from the fixed benchmark.

Artifacts are incremental: incomplete models appear as running/incomplete and
completed runs populate without changing cohort definitions.

## Future scene randomization

Do not randomize scenes in the first revised cohort because current training
visuals use layout 11/style 34/seed 42. Preserve the metadata-first design:

- rendering records layout, style, scene seed, and scene signature;
- scene inventory reports coverage by task and split;
- future rendering samples RoboCasa scenes deterministically per trajectory;
- cohort construction stratifies allowed scenes;
- evaluation reports training-scene and unseen-scene strata;
- access-blocking annotations are recomputed per physical scene/spawn rather
  than copied across layouts.

This avoids evaluator and metrics redesign when visual diversity is introduced.

### Certified scene compatibility gate (implemented 2026-08-19)

Scene randomization is now mediated by one shared policy in
`data_generation/task_level/scene_sampling.py`; rendering and live evaluation
must not choose scenes independently. The first candidate pool contains layouts
11, 15, 40, and 50 crossed with five styles and three simulator seeds (60 scene
triples). Layout 18 is excluded because its narrow `counter_1_main_group`
cannot safely host both robots for otherwise valid configurations.

Before generation or evaluation uses this pool, a simulator audit checks every
unique physical configuration against every candidate scene. The resulting
cache maps `(task, physical_configuration_signature)` to its certified scenes.
Sampling is uniform over that compatible set and reproducible from an explicit
sampling seed. A configuration with zero certified scenes is a hard error, not
silently dropped or replaced. New trajectory records carry both the physical
configuration and its signature so rendering and evaluation resolve the same
contract.

The production stovetop geometry was also aligned with other countertop
appliances: direct stovetop navigation and parent-counter reservation use the
same aisle-facing working corridor. This change must pass both the focused
390-case regression audit and the complete scene/configuration compatibility
audit before the cache is approved.

The gate is now complete. The final initialization audit accepted 15,420 of
15,420 task/configuration/scene records, and the shared-workspace audit accepted
2,870 of 2,870 applicable probes. Focused action, initialization, and workspace
regressions passed 60/60, 1,080/1,080, and 240/240 respectively. The merged
cache contains all 53 tasks, all 384 physical configurations, and exactly the
60 certified scenes; every configuration has either 30 or 60 compatible
scenes. The frozen 768-row target cohort (384 physical states crossed with two
coordinator assignments) contains no duplicate canonical configurations.

Both rendering and live evaluation now consume this same cache through
`sample_compatible_scene`. Contract verification sampled all 768 targets with
a fixed seed, reproduced every choice exactly, returned only certified scenes,
and exercised all 60 scenes. The default evaluation remains 10 uniformly
sampled configurations per task; `full_config` evaluates every configuration,
with explicit controls for repeats and caps.

The partial-observability SFT and live-evaluation paths now call one shared
prompt constructor. Under the production `none` mode it includes the identical
task goal, symbolic initial state, coordinator assignment, global tool
interface, private history, and observations while exposing no local or global
step index. The pre-generation contract suite passes 46 tests. The remaining
blocking step is the API-backed all-configuration generation canary, followed
by current concurrent-FSM verification and carrier binding; bulk generation
must not begin before that gate passes.
