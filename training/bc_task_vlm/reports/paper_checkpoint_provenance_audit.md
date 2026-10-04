# Paper checkpoint provenance audit

Date: 2026-09-15. Existing evaluations and figures were not modified.

## Scope and evidence

Inspected all 54 split results (27 distinct checkpoints) in the existing 43/10
artifact, covering Instruct scaling, Thinking+rationale scaling, 30/task rationale
ablations, and the DAgger evaluations. The companion
`paper_checkpoint_provenance_audit.json` retains resolved source runs, checkpoint
paths, saved epochs, training settings, evaluation arguments, episode identities,
task-level counts, and checks. Reproduce with
`python -m training.bc_task_vlm.audit_paper_checkpoint_provenance`.

All results contain 430 trained-task or 100 held-out-task episodes, ten per task,
with no duplicated episode IDs. Recomputed error-free success agrees with both
aggregate metrics and the artifact values. Regrouped rows agree with their
source except the intended cohort_split field. All adapter bases match training
configuration. All 27 safetensors files have valid header/file extents and 504
tensors; an additional safetensors/numpy scan found no nonfinite tensor values.
These integrity checks do not prove complete historical simulator correctness.

## Confirmed epoch-label swaps

Only Instruct 60/task and 90/task have reversed half/one-epoch labels. Both epochs
were evaluated. Server launch logs agree with the checkpoint paths in
parallel_run.json; trainer_state.json establishes the actual epoch.

| Scale | Actual one-epoch checkpoint | Directory label | Trained success | Held-out success |
|---|---|---|---|---|
| 30 | checkpoint-648 | ep1p0 | 352/430 | 45/100 |
| 60 | checkpoint-1298 | ep0p5 | 378/430 | 64/100 |
| 90 | checkpoint-1945 | ep0p5 | 396/430 | 55/100 |
| 120 | checkpoint-2593 | ep1p0 | 400/430 | 48/100 |
| 150 | checkpoint-3240 | ep1p0 | 405/430 | 65/100 |

The artifact builder derives epoch labels from directory names. Correct that
mapping from checkpoint metadata, with a mismatch guard. No reruns are needed
for these swaps. Other checkpoint epoch labels match, allowing rounding to the
nearest training step for half-epoch saves.

## Instruct 120/task

This is the actual one-epoch checkpoint. Its training settings match the other
Instruct scales, and every Instruct scale/epoch uses the same evaluation arguments
apart from checkpoint/server endpoint and identical episode configurations,
coordinator assignments, scenes, and reference lengths.

The 48/100 held-out score is reproduced from complete episodes. Notable outcomes:

- GarnishCake: 0/10, all budget_exhausted.
- ClusterItemsForClearing: 0/10, all mutual_wait_deadlock.
- PrepareCheeseStation: 2/10; the other eight exhaust the budget.
- ServeMealJuice: 0/10; eight deadlocks and two budget exhaustion.

At actual one epoch, 90/task has 2, 6, 5, and 0 successes respectively on these
tasks; 150/task has 10, 0, 10, and 0. This is evidence of task-dependent rollout
performance, not proof of overfitting or evidence that the evaluator is flawless.

## Additional fixed-cohort comparability issue

Thinking+rationale 60/90/120 (both available epochs) use correct checkpoints, but
the artifact regrouped their original 47/6 evaluations into 43/10. Training used
the intended 43 training tasks; regrouping is not training leakage.

Trained-task episode configurations match the native 43/10 cohort. In the
held-out split, four tasks differ: ArrangeBreadBowl, DistributeChicken,
PrepareCheeseStation, ServeMealJuice. By episode rank, 36 configuration signatures
and 35 scenes differ. Ignoring rank, only 66/100 full configuration/coordinator/
scene instances match as a multiset. This is not merely reordered episodes.

The existing values remain valid measurements on their sampled episodes, but
must not be described as exactly the same fixed episodes as other variants.
For exact matching, evaluate these four tasks using the native 43/10 cohort for
each affected checkpoint, preserving the existing results as a separate source.
For the paper's one-epoch models this is 40 episodes per model, 120 total, before
recombining with the 60 already matching held-out episodes.

## Other settings and limitations

All non-DAgger training variants share the inspected training hyperparameters
and 43 training-task identities, with intentional changes to model variant,
rationale targets, processor and scale-specific preprocessed data.
Thinking inference uses different sampling/output settings from Instruct;
Thinking without rationale additionally enables missing-opening-tool-tag recovery.
DAgger training has its own learning rates and initialization, as expected.
No new claim of identical inference settings across model variants is warranted.

Historical code was not proven byte-identical from immutable source snapshots.
This audit establishes recorded provenance, completeness and metric consistency,
not universal validity of every simulator or protocol implementation.
