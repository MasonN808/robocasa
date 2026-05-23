# Sampling Method Data Generation

Use these commands from the repository root to generate comparable raw
trajectory data under `raw/sampling_methods/` for post-hoc sampling analysis.

Set the output root first:

```bash
export ROBOCASA_TASK_LEVEL_DATA_ROOT="$PWD/data_generation/task_level/data"
```

For runs meant to sit alongside the existing k3 sampling-method dataset, use:

```bash
export ROBOCASA_TASK_LEVEL_DATA_ROOT="$PWD/data_generation/task_level/data_k3"
```

## All Analysis Methods

This runs `base`, `random`, `verbalized`, and `high_temperature` into sibling
directories under `$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/`.
Existing method directories are resumed in place.

```bash
bash scripts/generate_raw_sampling_methods.sh \
  --num-trajectories 30 \
  --verbalized-k 3 \
  --max-workers 4 \
  --max-retries 5 \
  -- --enable-validation
```

## Strategies

- `base`: one trajectory per model call using the task's native prompt and
  response schema.
- `random`: one trajectory per model call with a UUID sample ID prepended to
  perturb the prompt.
- `verbalized`: K trajectories per model call using a structured
  `responses[]` output where each item contains `probability` and `trajectory`.
- `structured_random`: one base-style trajectory per model call. The code
  derives one deterministic random seed per task/run and uses it to select one
  low/medium/high configuration for
  `communication_message_length`, `communication_message_complexity`, and
  `tool_call_diversity` before prompting the model. The model returns the
  normal single-trajectory JSON shape; the seed stays internal, and the prompt
  only includes the selected levels.
- `high_temperature`: one trajectory per model call using the base prompt shape
  with a higher sampling temperature.

## Individual Methods

Use individual commands when you only need to resume or regenerate one method.

```bash
# Base: one trajectory per run.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 30 \
  --random-start-location true \
  --sampling base \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/base"

# Random: UUID prompt tag, one trajectory per run.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 30 \
  --random-start-location true \
  --sampling random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/random"

# Verbalized: K trajectories per run, so num-runs = num-trajectories / K.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 10 \
  --random-start-location true \
  --sampling verbalized \
  --verbalized-k 3 \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/verbalized"

# Structured random: one base-style trajectory per run. Increase --num-runs
# to draw more seeded control configurations through separate model calls.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 30 \
  --random-start-location true \
  --sampling structured_random \
  --temperature 0.6 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/structured_random"

# High temperature: base prompt shape with temperature 1.0.
python -m data_generation.task_level.generation.raw.cli \
  --tasks verified \
  --num-runs 30 \
  --random-start-location true \
  --sampling high_temperature \
  --temperature 1.0 \
  --model gemini-3-flash-preview \
  --location global \
  --thinking-level low \
  --max-workers 4 \
  --max-retries 5 \
  --enable-validation \
  --resume "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/high_temperature"
```

If a method directory does not exist yet, replace `--resume ...` with
`--summary-path "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/<method>/summary.json"`.

## Delta CPU Slurm

Submit structured random sampling on a Delta CPU node with:

```bash
sbatch scripts/generate_structured_random_delta_cpu.sh
```

Common overrides:

```bash
NUM_RUNS=20 MAX_WORKERS=8 \
sbatch scripts/generate_structured_random_delta_cpu.sh
```

The structured-random Delta wrapper defaults to the k3 dataset root:
`data_generation/task_level/data_k3/raw/sampling_methods/structured_random`.
It also defaults to `NUM_RUNS=30`, matching the existing base/random/verbalized
sampling-method trajectory count convention of 30 trajectories per verified
task.

## Validity Checks

After generation, check completion and saved invalid trajectory counts:

```bash
jq '{is_complete, pending_tasks, num_trajectories}' \
  "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/base/summary.json"

jq -r '.task_summaries[] | select((.pending_run_indices | length) > 0 or (.trajectory_stats.invalid_trajectories // 0) > 0) | [.composite_task, (.pending_run_indices | length), (.trajectory_stats.invalid_trajectories // 0)] | @tsv' \
  "$ROBOCASA_TASK_LEVEL_DATA_ROOT/raw/sampling_methods/base/summary.json"
```

`summary_errors.json` is historical attempt metadata. It can contain validation
errors from failed attempts even when the saved trajectory files are valid.

## Analysis Metrics

After generation, summarize tool-call distributions, stitched conversation
length, and optional Qwen3 trajectory-conversation cosine-similarity metrics:

```bash
python data_analysis/analyze_sampling_methods.py \
  --input-root data_generation/task_level/data_k3/raw/sampling_methods \
  --output-dir data_generation/task_level/data_k3/raw/sampling_method_analysis
```

That writes:

- `sampling_method_analysis.json`: full nested metrics for the whole dataset,
  each sampling method, each task aggregated across sampling methods, and each
  `(sampling_method, task)` pair.
- `task_all_methods_metrics.csv`: one row per task aggregated across sampling
  methods.
- `task_metrics.csv`: one row per `(sampling_method, task)` pair.
- `sampling_method_metrics.csv`: one row per sampling method.
- `dataset_metrics.csv`: one row for the whole input dataset.

`environment_metrics.csv` and `environment_all_methods_metrics.csv` are also
written as aliases for older notebooks.

The embedding metric stitches all `communicate` messages in each trajectory
into one ordered conversation string, embeds that string once, and computes
exact unique-pair cosine similarity summaries for each `(sampling_method, task)`
group. By default, the analyzer uses at most 30 sorted trajectories per group,
so complete groups produce `30 choose 2 = 435` cosine scores. Use
`--max-trajectories-per-task 0` to compare all saved trajectories in each group.
Trajectory embeddings are cached in the output directory by default using the
model, settings, and exact stitched texts as the cache key; reruns skip the
embedding model when the cache matches. Use `--embedding-cache refresh` to
force recomputation or `--embedding-cache off` to disable caching.
Trajectory JSON loading is parallelized with `--load-workers 16` by default;
increase it on a high-throughput filesystem or lower it if shared storage is
under pressure.

To compute embedding similarities with Qwen/Qwen3-Embedding-4B through vLLM
directly in the script:

```bash
python data_analysis/analyze_sampling_methods.py \
  --input-root data_generation/task_level/data_k3/raw/sampling_methods \
  --output-dir data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3 \
  --embedding-provider vllm \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --embedding-batch-size 256 \
  --dtype float16
```

On Slurm, submit the same Qwen3 cosine analysis with:

```bash
sbatch scripts/analyze_sampling_method_cosine_qwen3.sbatch
```

Common overrides:

```bash
EMBEDDING_BATCH_SIZE=512 \
SAMPLING_ANALYSIS_OUTPUT_DIR=data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3 \
sbatch scripts/analyze_sampling_method_cosine_qwen3.sbatch
```

The sbatch wrapper uses `--embedding-cache auto` by default, so reruns reuse
matching `trajectory_embeddings.npy` files in the output directory. Use
`EMBEDDING_CACHE=refresh` to force recomputation.

To use a running vLLM OpenAI-compatible embeddings server instead:

```bash
vllm serve Qwen/Qwen3-Embedding-4B \
  --runner pooling \
  --served-model-name Qwen/Qwen3-Embedding-4B

python data_analysis/analyze_sampling_methods.py \
  --input-root data_generation/task_level/data_k3/raw/sampling_methods \
  --output-dir data_generation/task_level/data_k3/raw/sampling_method_analysis_qwen3 \
  --embedding-provider vllm-openai \
  --embedding-model Qwen/Qwen3-Embedding-4B \
  --vllm-base-url http://localhost:8000/v1
```
