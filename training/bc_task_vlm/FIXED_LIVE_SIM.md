# Fixed live-sim benchmark

The benchmark separates episode configuration from expert behavior. A rendered
trajectory supplies a known-buildable initial simulator state; evaluation never
asks a model to imitate that trajectory. Every model receives the same initial
state, coordinator assignment, scene, seed, prompt contract, and budget.

Split names are deliberately precise:

- `train_task_types`: task types represented during SFT
- `heldout_task_types`: task types absent from SFT

The configuration cohort stores every canonical configuration exactly once.
Evaluation expands those configurations into episodes in one of two modes:

- `--cohort-mode sampled` (default): 10 episodes per task by default. It uses a
  seeded, uniformly shuffled pass over all available configurations. A caller
  can set `--episodes-per-task N`; configurations repeat only after every unique
  configuration for that task has appeared once.
- `--cohort-mode full_config`: one episode for every configuration by default.
  `--episodes-per-config N` repeats each configuration, while
  `--max-configs-per-task N` runs a seeded subset of configurations.

The available-configuration table itself never contains duplicates.

## Lifecycle

Build candidates:

```bash
python -m training.bc_task_vlm.fixed_live_sim_cohort \
  --dataset-root /work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3 \
  --selection data_analysis/analysis/held_out_task_selection/held_out_task_selection.json \
  --cohort-seed 20260817 \
  --output training/bc_task_vlm/eval_manifests/fixed_live_sim_v1/candidates.json
```

Run both oracle splits with `fixed_live_sim_eval.sbatch`. Oracle runs must use
`--concurrent-expert-replay`; the launcher adds it automatically so serialized
steps recover their original atomic tick grid. Freeze only after every episode
has FSM success, no rejection, and no harness error:

```bash
python -m training.bc_task_vlm.fixed_live_sim_cohort \
  --candidate-manifest training/bc_task_vlm/eval_manifests/fixed_live_sim_v1/candidates.json \
  --oracle-results TRAIN_RESULTS.jsonl --oracle-results HELDOUT_RESULTS.jsonl \
  --output training/bc_task_vlm/eval_manifests/fixed_live_sim_v1/frozen.json
```

Evaluate by setting `COHORT_SPLIT`, `COHORT_MODE`, and the corresponding count
variables. All fixed-cohort model runs use partial history, no local
or global step indices, consume-once observations, uniform durations, FSM
success, and the concurrent-FSM contention policy.

Summarize one model (repeat `--results` for worker files or splits):

```bash
python -m training.bc_task_vlm.summarize_fixed_live_sim \
  --results RESULTS.jsonl --label MODEL \
  --output-json summary.json --output-md summary.md
```

Pass `--baseline-results` and `--baseline-label` to add a paired hierarchical
bootstrap comparison over identical episode IDs. The primary report contains
micro and macro success per split; CIs resample tasks and then episodes within
task. Wilson intervals are retained only as a descriptive reference.

Scene fields already include membership and signatures. A future randomized
renderer/evaluator can add sampled layout/style/scene seeds and mark scenes as
seen or unseen without changing episode identity or summary code.
