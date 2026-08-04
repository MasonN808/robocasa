# Making tick generation cheaper

## The measurement that frames everything

Per trajectory, old flat run (`data/raw/20260324T031125Z`) vs the tick pilot
(job 266619), same model (`gemini-3-flash-preview`):

| per trajectory | old flat | new tick | ratio |
|---|---|---|---|
| **reasoning tokens** | 1,153 | **21,335** | **18.5x** |
| prompt tokens | 566 | 4,865 | 8.6x |
| output tokens | 1,355 | 1,001 | 0.7x |
| cost | $0.0078 | $0.0694 | 8.9x |
| trajectories per API call | **4** | **1** | 0.25x |

**Reasoning tokens are ~21x the output tokens.** Anything that reduces
*thinking* is worth roughly 20x more than anything that shortens the answer.
The written output is actually SHORTER on tick, so the format is not producing
more text — it is thinking far more, and it was configured to be allowed to.

The two runs differed in config, not just format:

| | old flat | new tick |
|---|---|---|
| sampling strategy | `verbalized`, k=4 | `base` (k=1) |
| thinking_level | `medium` | `null` (uncapped) |

Neither difference was deliberate. How much of the 8.9x is config and how much
is tick reasoning being intrinsically harder is **unmeasured** — that is what
the probes below are for.

## Ranked levers

### Tier 1 — expected to dominate, and they compound

- **`--thinking-level medium`.** A direct cap on the dominant cost. The old
  flat runs used it.
- **`--verbalized-k 4`.** Better than the 4x round-trip saving it appears to
  be: the model reasons ONCE about the task and emits 4 variants, so the
  thinking amortises. May approach 4x on reasoning too. Partial validity is
  fine — 2 valid of 4 still beats 1 per call.

### Tier 2 — real but smaller

- **Prompt caching.** `cached_input_tokens: 0` in both runs; never used. The 15
  tick rules and the tool specs are byte-identical across every run of a task.
  Cost only, no latency win.
- **`--max-workers`.** Currently 6. Wall-clock only, bounded by API rate
  limits. See the existing `scripts/full_traj_gen/probe_*_rate.sh`.
- **Lower `--max-retries`.** Each retry is a FULL regeneration with full
  reasoning. The pilot recovered all 3 retries early; the dead4 failures burned
  all 5. Dropping 5 -> 3 cuts the expensive tail.

### Tier 3 — plausible, needs its own A/B

- **Repair instead of regenerate.** The retry prompt says *"Regenerate the full
  trajectory from step 0. Do not continue or patch the previous attempt."*
  Patching would be dramatically cheaper — the model already holds a nearly
  correct plan and one error. That instruction may encode a real finding
  (patching produced worse results), so measure before changing it.
- **Deterministic pre-repair.** Some rejections are mechanical (id typos,
  missing `message`). Fixing those in code costs zero tokens.
- **Trim the prompt.** 15 rules plus the tool descriptions lengthened during
  spec re-materialization. More text to digest means more reasoning. Trim LAST:
  every one of those rules was earned by a failure.

## Probe protocol

One variable at a time, same two tasks every time so the numbers compare.

Baseline is job **266619**: PrepareCoffee + AddSugarCubes, 10 runs each, 4
workers, `base` strategy, thinking `null` -> 20 trajectories, **$1.3888**,
**7:24 wall** (sacct; an earlier "~4 min" eyeball estimate was wrong).

## Results

All probes: PrepareCoffee + AddSugarCubes, validity 100% in every cell.

| probe | traj | $/traj | reason/traj | wall | **s/traj** |
|---|---|---|---|---|---|
| baseline (uncapped, `base`) | 20 | $0.0694 | 21,335 | 7:24 | 22.2 |
| **`--thinking-level medium`** | 20 | **$0.0500** | 15,068 | **5:25** | **16.2** |
| `verbalized-k 4` | 12 | $0.0268 | 7,240 | 8:22 | 41.8 |
| `verbalized-k 2` | 18 | $0.0318 | 8,953 | 15:57 | 53.1 |
| medium + k=2 + 12 workers | 18 | $0.0354 | 10,382 | 12:08 | 40.4 |

### Recommendation: `--thinking-level medium`, and nothing else

**-28% cost, -27% wall, validity untouched.** The uncapped run was spending
~6k reasoning tokens per trajectory that bought no correctness.

**Verbalized sampling is a cost lever that COSTS latency, and it does not
combine well.** It halves $/traj but doubles-to-triples s/traj, because each
call returns k trajectories in one much longer request and there are k times
fewer calls to overlap. Raising workers to 12 recovered only part of it (53.1
-> 40.4 s/traj, still worse than doing nothing). Combining it with `medium` was
no cheaper than k=2 alone.

It also brings failure modes the base path does not have:
- **k=4 is unusable on PrepareCoffee**: Vertex `400 INVALID_ARGUMENT`, response
  count exceeds request limits. The tick response schema is already large and
  verbalized multiplies it, so this fails per-task unpredictably across 53.
- `UnexpectedStepIndexSemanticValidationError: step 0 does not match expected
  index 1` appeared 5x -- candidate step numbering looks mishandled on the
  verbalized+tick path. Unresolved.
- 1 of 5 runs failed even after retries, in both verbalized probes.
- It surfaced a crash that discarded a whole run's payload (fixed: type-stable
  `_error_event_key`).

**Caveat on precision:** 18-20 trajectories per cell, so a few percent of $/traj
is noise. The latency differences are large and consistent; the small cost
differences between verbalized variants are not meaningful.

### Actual scale-out cost, for future budgeting

Job 267262 (53 tasks x 30, uncapped, base): **1543 trajectories, 100% valid,
$124.41**, 3.0% of runs exhausted retries. A pre-hoc estimate of $50-70 from
pilot per-trajectory cost was wrong because it ignored retry multiplication.
At `medium` the same corpus should land near $90.

Report for each probe: wall clock, total cost, cost/valid trajectory, valid
fraction, and retries used. **Validity is the guard** — a config that halves
cost while dropping validity is not a win, and tick reasoning may genuinely
need more thinking than flat did.
