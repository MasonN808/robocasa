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
~4 min wall.

Report for each probe: wall clock, total cost, cost/valid trajectory, valid
fraction, and retries used. **Validity is the guard** — a config that halves
cost while dropping validity is not a win, and tick reasoning may genuinely
need more thinking than flat did.
