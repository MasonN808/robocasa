# Release manifest (Step 2 output)

**Generated:** 2026-10-04, on `publication-cleanup` at `9e97a2f`.
**Method:**
- Static import closure (AST walk, including function-local, relative and sibling-file imports) from the entry points that produced the paper's results.
- Entry points were read from the launchers behind the paper runs.
- Data dependencies were taken from the saved run arguments of the paper's evaluations and training runs.

**Scope** (see [`publication_readiness_plan.md`](publication_readiness_plan.md) B.1):
- generation of the 7,950-trajectory dataset;
- SFT;
- live-sim evaluation;
- Figs. 3 and 5–7;
- the HF export.

Out of scope: the task-spec pipeline, off-sim evaluation, DAgger, diversity analysis (Table III / Fig. 4, deferred to Step 12) and all other exploration.

This is a review document; nothing has been deleted yet.

## 1. Entry points (what `reproduce.py` and the configs will wrap)

| Stage | Entry point | Paper launcher it came from |
|---|---|---|
| Generate raw trajectories (Gemini cascade) | `data_generation.task_level.generation.raw.production_cascade` | `generate_tick150_cascade_v1.sbatch` |
| Insert observations / postprocess | `data_generation.task_level.generation.image.cli` | `postprocess_tick150_cascade_v1.sbatch` |
| Render in simulation | `scripts/sweep_trajectories.py` | `render_tick150_cascade_v1.sbatch` |
| Scene compatibility cache | `training.bc_task_vlm.build_scene_compatibility_cache`, `merge_shared_workspace_compatibility` | `finalize_certified_scene_cohort.sbatch` |
| Evaluation cohort | `build_fixed_cohort_targets`, `fixed_live_sim_cohort`, `freeze_configuration_cohort`, `fixed_cohort_selection` | `finalize_*_cohort.sbatch` |
| 43/10 split and nested 30–150 selections | `training.bc_task_vlm.build_scale_experiment_manifests` | — |
| Materialize scale subsets | `training.bc_task_vlm.materialize_scale_artifacts` | `materialize_tick150_*` |
| Preprocess | `training.bc_task_vlm.preprocess` | `training/scripts/launch_cpu_preprocess.sh` |
| Train (LoRA) | `training.bc_task_vlm.main` via `accelerate` | `training/scripts/launch_multigpu.sh` |
| Live-sim eval | `training.bc_task_vlm.live_sim_parallel_eval` → `live_sim_eval` | `fixed_live_sim_eval.sbatch`, `fixed_live_sim_open_weight_vllm.sbatch` |
| Summaries and Figs. 5–7 | `summarize_fixed_live_sim`, `build_43_10_eval_artifact`, `export_communication_ablation_png`, `export_sft_scale_pngs` | `build_communication_ablation_artifact.sbatch` |
| Fig. 3 | `scripts/plot_task_phase_distribution.py`, `export_task_phase_pngs` | — |
| HF dataset and adapters | `robotalk_hf.export_robotalk`, `robotalk_hf.publish_models` | — |

### 1.1 The exact paper evaluation and training settings

These are recovered from the saved runs and become the shipped configs.

**Evaluation**, from `eval_runs/fixed_live_sim/sft_s30_noreason_…/train_task_types/parallel_run.json`. The other cells differ only in the adapter, endpoint and cohort split.

```text
--backend vllm --cohort-split {train_task_types|heldout_task_types} --cohort-mode sampled
--episodes-per-task 10 --episodes-per-config 1 --evaluation-seed 20260817
--partial-history --partial-step-index-mode none --partial-observation-mode consume-once
--train-get-image --success-criterion fsm --uniform-durations --no-forced-json
--map-renderer raster --map-dpi 60 --gl-backend egl --no-record-firsts
--max-new-tokens 256 --vllm-temperature 0.0
--scene-compatibility-cache training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json
--scene-sampling-seed 20260819
--manifest scale_experiments/tick150_reasoning_43train_10held_seed20260821/fixed_live_sim_cohort.json
--dataset-root <rendered dataset>   # → HF download in the release
```

Things to carry into the configs:
- Thinking runs override `--max-new-tokens`.
- The Thinking-without-rationale arm adds the `_recovery_v1` parser recovery.
- Gemini runs (Fig. 5) use the native function-calling backend.
- The step budget is the default `--step-budget-factor 2.0`, doubled by `--train-get-image`, which gives the paper's 4×. Pin it explicitly.

**Training**, from `runs/qwen3vl8b-tick150-s30-noreason-…/run_config.json`:
- `lora_r 16`, `lora_alpha 32`, `lora_dropout 0.05`;
- `learning_rate 1e-4`, `weight_decay 0.01`, `cosine`, `warmup_ratio 0.03`, `num_epochs 1`;
- `per_device_batch_size 8` × `grad_accum 2` × 4 GPUs;
- `max_length 8192`, `image_resolution 512`, `max_images_per_sample 4`, `gradient_checkpointing`, `bf16`, `flash_attention_2`;
- `sft_format tool_call`, `partial_history`, `partial_step_index_mode none`, `partial_observation_mode consume_once`, `train_get_image`, `supervise_last_assistant_turn_only`, `seed 42`;
- `train_reasoning` is toggled for the rationale arms.

`eval_generation_max_samples 0` means the in-training off-sim evaluation was **off** in the paper runs, so removing it changes no result.

## 2. Keep: our code (87 modules)

| Area | Modules |
|---|---|
| Tasks, FSM, concurrency | `data_generation/task_level/tasks/{__init__,base}.py`, `tasks/shared/{concurrent_fsm,constants,errors,fsm,instances,prompting,render_order,scheduling,schema,state,types,validation_contract,workspace_semantics}.py`, `tasks/specs/{__init__,runtime}.py` |
| Tool interface and grounding | `subatomic_tool_specs.py`, `subatomic_tool_calls.py`, `grounding_specs.py`, `object_type_families.py`, `scene_sampling.py`, `data_generation/utils.py` |
| Generation | `generation/raw/{cli,config,costs,errors,llm_critic_repair,orchestrator,outputs,production_cascade,cascade_canary,progress,runtime_support}.py`, `generation/image/{cli,processor}.py`, `runtime/{batch_generation,client,on_demand_generation,render_env}.py`, `sampling/{base,structured_random}.py` |
| Root modules that must move into the package | `tick_format.py` (imported by `specs/runtime.py`, `shared/prompting.py`, `image/processor.py`, `raw/runtime_support.py`), `insert_waits.py` (`raw/runtime_support.py`), `scripts/sweep_trajectories.py` (render entry point) |
| Training | `dataset.py`, `preprocess.py`, `preprocessed_data.py`, `main.py`, `peft_compat.py`, `prompting.py`, `communication_profiles.py`, `tool_calling.py`, `schema_utils.py`, `task_registry.py`, `build_scale_experiment_manifests.py`, `materialize_scale_artifacts.py` |
| Evaluation | `live_sim_eval.py`, `live_sim_parallel_eval.py`, `fixed_cohort_selection.py`, `fixed_live_sim_cohort.py`, `freeze_configuration_cohort.py`, `build_fixed_cohort_targets.py`, `build_scene_compatibility_cache.py`, `merge_shared_workspace_compatibility.py`, `summarize_fixed_live_sim.py` |
| Figures | `build_43_10_eval_artifact.py` (reduce to table and plot logic), `export_communication_ablation_png.py`, `export_sft_scale_pngs.py`, `export_task_phase_pngs.py`, `plot_style_utils.py`, `plot_caption_metadata.py`, `scripts/plot_task_phase_distribution.py` |
| Release | `robotalk_hf/export_robotalk.py`, `robotalk_hf/publish_models.py`, `robotalk_hf/static_space/` |

**Reached, but to be pruned during Step 3** (imported only through registries or dead options):
- `sampling/{high_temperature,random_number,verbalized}.py`: Table III-only strategies. Drop their registry entries.
- `tasks/shared/partitions.py`: work partitions. Production uses `partition_policy="none"`, so prune it once the call sites are confirmed dead.

### 2.1 Keep: data and config files

| File(s) | Why |
|---|---|
| `data_generation/task_level/tasks/specs/verified/` (53 JSON + README) | The task suite. Drop `specs/older/`, `specs/deferred_scene_matching/` and the 8 loose files in `specs/`. |
| `training/bc_task_vlm/scale_experiments/tick150_reasoning_43train_10held_seed20260821/` (`43_train_10_heldout.json`, `selection_{30,60,90,120,150}.json`, `fixed_live_sim_cohort.json`) | The exact split, nested scale subsets and evaluation cohort used by every paper run → `configs/`. Drop `scale_experiments/tick150_v1/` (old 47/6). |
| `reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json` | Scene cache passed to every paper evaluation → `configs/`. The other 12 files in that folder are preflight evidence; drop them. |
| `training/bc_task_vlm/accelerate_multigpu.yaml` | Training launcher config. |
| Rendered dataset (`tick53x150_state_grounded_cascade_v1_rendered`) | Not in git. The release reads it from HF `DorianAtSchool/RoboTalk`. **Check:** the evaluator reads `--dataset-root` for configurations and reference lengths, so confirm the HF layout satisfies it, or add an adapter. |

### 2.2 Upstream `robocasa/` package

The closure reaches about 450 upstream files: `robocasa/environments/kitchen/__init__` imports every kitchen task.

**Recommendation:** keep the `robocasa/` package as a whole fork and do not prune upstream tasks. Pruning would break the upstream registry for little gain. Do the following:
- Keep our 13 reached additions: `sim_tool_executor*`, `trajectory_runner*`, `trajectory_adapter`, `trajectory_pruning`, `sim_tool_specs`, `placement*`, `occupancy_grid`.
- Keep our 10 modified upstream files (`kitchen.py`, three composite tasks, `camera_utils`, `env_utils`, two wrappers, `__init__`, `demo_kitchen_scenes`).
- Keep the 118 added or modified asset files.
- Delete our unreached addition `robocasa/utils/generate_llm_task_descriptions.py`.
- List all of the above in `docs/FORK_CHANGES.md`.
- Upstream `robocasa/scripts/*` stays only where reached (`dataset_scripts`) or needed for asset download (`download_kitchen_assets`, `setup_macros`).

## 3. Move before delete

| Shipped code | Depends on excluded code | Action |
|---|---|---|
| `live_sim_eval.py:1005` | `evaluation.VisionGenerationCollator` (HF-transformers generation backend) | Move the class to `robotalk/evaluation/hf_backend.py`, or drop the HF backend if only vLLM and Gemini are shipped. All paper runs used `--backend vllm` or Gemini. |
| `live_sim_eval.py:1336` | `eval_standalone._GEMINI_MIME_BY_SUFFIX`, `_message_text`, `build_vertex_function_declarations` | Needed for Fig. 5 Gemini runs. Move to `robotalk/evaluation/gemini.py`. |
| `live_sim_eval.py:3659` | `eval_standalone._load_few_shot_blocks`, `_load_task_spec_blocks` | Only used under `--few-shot`/`--task-spec-detail`, which no paper run sets. Delete the options. |
| `main.py:52,1794` | `evaluation.evaluate_structured_generation` | In-training off-sim eval, off in the paper runs. Delete the hook and its `eval_generation_*` args. |
| `fixed_live_sim_open_weight_vllm.sbatch:127-141` | `dagger.loop`, `dagger.collect` branches | Strip when generalizing into `scripts/slurm/`. |
| `tasks/specs/runtime.py` and others | Root `tick_format.py`, `insert_waits.py` | `git mv` into the package and rewrite imports. |

Once these are done, `evaluation.py`, `eval_standalone.py`, `prediction_io.py` and `metrics.py` have no importers left and can be deleted.

## 4. Delete

By directory. Counts are tracked `.py` files unless noted.

- **Repository root (25 `.py`, 259 `.sbatch`, 7 `.md`, misc).** Delete everything except `README.md`, `LICENSE`, `pyproject.toml`, `uv.lock` and `MANIFEST.in`, after the root-module moves in §3.
  - `.py` examples: `build_v3_artifact.py`, `calibrate_partitions.py`, `concurrent_fsm_demo.py`, `corpus_audit.py`, `coverage_*`, `lockstep_check.py`, `order_ab.py`, `validity_ab.py`, `replay_report.py`, `stage_tick_for_sim.py`, `weight_work_partitions.py`, `push_*_hf.py`, `test_google_cloud.py`, `preflight_er2.py`, `setup.py` (replaced by `pyproject.toml`).
  - Other files: `requirements.txt` (`-e .`), `typescript`, `nvtop-x86_64.AppImage`, `MUJOCO_LOG.TXT`, `slurm_submit_with_qos_retry.sh`, `run_*.sh`, `tick_postprocess_all.sh`.
- **`data_generation/task_level/`:**
  - `pipeline/` (20, task-spec pipeline);
  - `analysis/report_text_diversity.py`;
  - `generation/{exemplar_normalization,grounding,thinking_level_canary}.py`;
  - `tasks/migrate_cabinet_parents.py`;
  - `build_initial_config_manifest.py`;
  - the tracked `data/raw/20260324T031125Z/` sample outputs (84 files);
  - `tasks/TASKS.md` (fold into docs).
- **`training/bc_task_vlm/` (84 unreached).** Delete:
  - all `probe_*`, `audit_*`, `diagnose_*`, `verify_*` and `build_*_artifact.py` scripts except `build_43_10_eval_artifact.py`;
  - the off-sim group: `evaluation.py`, `eval_standalone.py`, `prediction_io.py`, `metrics.py`, `judge_communications.py`, `divergence_analysis.py`, `export_results_table.py`, `plot_*_comparison.py`;
  - one-offs: `merge_thinkrat_native43_repairs.py`, `regroup_fixed_live_sim_results.py`, `merge_pretokenized_shards.py`, `live_sim_timing_*`, `calculate_average_tokens_for_tasks.py`, `analyze_trajectory_tokens.py`, `build_task_split_variant.py`, `build_tick100_eval_manifests.py`, `capture_plate_store_dinner_overview.py`, `render_robotalk_paper_video.py`, `render_communication_ablation_examples.py`, `audit_paper_checkpoint_provenance.py` (run it for the model cards first);
  - `robotalk_hf/anonymize_review.py`;
  - all `*.md` notes, `artifact_results_explorer.html`, `plans/`, `hf_repo_lists/`, `eval_data_subset/`, `eval_manifests/` (superseded by `scale_experiments/…/fixed_live_sim_cohort.json`), `reports/`, `eval_runs/`, `runs/qwen35_9b_lora/`, `requirements*.txt`.
- **`training/scripts/`.** Delete everything after generalizing `launch_multigpu.sh` and `launch_cpu_preprocess.sh` into `scripts/slurm/` and `scripts/`.
- **`scripts/`.** Delete everything except `sweep_trajectories.py` and `plot_task_phase_distribution.py` (both move into the package).
- **Whole directories:** `data_analysis/`, `model_evals/`, `policies/`, `external/` (and the `.gitmodules` entries), `scratch/`, `eval_task_subsets/`, `artifacts/`, `tests/rl/`, the upstream Sphinx `docs/` tree (keep `docs/paper/`), `docs/dorian-scale-tasks-changes/`.
- **Not tracked, so nothing to delete:** `training/bc_task_vlm/dagger/`, which is gitignored. See open question 1.

## 5. Tests

| Bucket | Files |
|---|---|
| **Keep (51 import only kept code).** Retarget them after the restructure, then drop the ones exercising pruned options. | `test_concurrent_fsm`, `test_workspace_semantics`, `test_wait_release_rule`, `test_wait_releases`, `test_scene_sampling`, `test_task_level_tool_interface`, `test_simplified_generation_prompt`, `test_tick_format`, `test_tick_image_injection`, `test_trajectory_validation_contract`, `test_cascade_canary`, `test_bc_task_vlm_{dataset,preprocess,partial_observability,live_sim_partial,live_sim_parallel_eval,causal_training,render_order_alignment}`, `test_access_state_sampling`, `test_sim_tool_specs`, `test_task_level_{grounding,trajectory_generation,progress_fallback}`, `test_tasks_validity`, `test_spec_parity`, `test_trajectory_{adapter,pruning,runner_parent_fixture}`, `test_render_order`, `test_handoff_examples`, `test_occupancy_grid*`, `test_placement_map`, `test_dataset_playback`, `test_env_determinism`, `test_novel_instruction`, `test_post_traj_generation`, `test_datasets`, and in `training/bc_task_vlm/tests`: `communication_profiles`, `fixed_cohort_selection`, `prompt_contract_parity`, `scene_compatibility_cache_tools`, `paper_checkpoint_epoch` |
| **Keep-list tests that still go** (their subject is out of scope) | `test_bc_task_vlm_evaluation`, `test_bc_task_vlm_metrics`, `test_dagger_data_generation`, `test_sampling_method_analysis`, `test_work_partitions`, `test_merge_sweep_summaries`, `test_placement_sweep`, `test_sweep_trajectories` (keep only the parts covering the render entry point) |
| **Mixed: fix or trim** | `test_bc_task_vlm_live_sim_eval` (drop the `divergence_analysis` cases), `test_bc_task_vlm_preprocessed_data` (drop the shard-merge cases), `test_sim_tool_executor` (drop the `generate_llm_task_descriptions` cases), `test_fixed_live_sim` (drop the `build_fixed_live_sim_artifact` cases) |
| **Mixed: delete** | `test_derived_coordination` (`lockstep_check`), `test_wait_for_signal` (`insert_wait_for_signal`), `test_task_level_phase1`/`phase4`/`prompts`, `test_task_specs` (pipeline). `test_door_fixtures`/`test_fixtures`/`test_layouts` need upstream `robocasa/scripts` entry points: keep them as upstream tests only if those scripts stay, otherwise delete. |
| **Delete (16)** | `test_asset_load_speed`, `test_env_speed`, `test_bc_task_vlm_average_tokens`, `test_exemplar_normalization`, `test_list_hf_robocasa_repos`, `test_lockstep_check`, `test_low_level_data_export`, `test_merge_hf_stage_shards`, `test_stage_hf_sweep_datasets`, `test_shared_workspace` (imports the pipeline's `phase1`), `test_sim_normalization`, `test_task_groups`, `test_task_level_phase{2,3,5}`, `test_task_level_pipeline_cli` |

## 6. Decisions (2026-10-04)

1. **DAgger:** not committed anywhere; it stays only on disk.
2. **Backends:** only vLLM and Gemini ship.
   - `HfPolicy`, `DegeneratePolicy`, the `--backend hf`/`degenerate` choices and `--model-name-or-path` are removed.
   - `OraclePolicy` (expert replay) stays for now because tests and render checks use it.
3. **Checks:** the prompt-consistency gate (`verify_generation_training_eval_prompts.py`) is the only check tool kept. It becomes a test in Step 9; `validate_scale_artifacts` and `verify_scene_cohort_contract` are deleted.

## 7. Step 3 progress

**Pass A (done).**
- §3 code moves:
  - Gemini helpers → `training/bc_task_vlm/gemini_function_calling.py`;
  - HF, degenerate and DAgger-prefix paths removed from `live_sim_eval.py`/`live_sim_parallel_eval.py`;
  - `--few-shot`/`--task-spec-detail` removed;
  - in-training off-sim eval removed from `main.py` (trainer renamed `LengthGroupedTrainer`).
- 1,016 files deleted per §4/§5.
- Tests trimmed:
  - divergence-analysis and LLM-task-description cases removed;
  - `test_fixed_live_sim` keeps its summarizer and Wilson assertions but drops the HTML-report half;
  - `test_dataset_playback` deleted, because its `robocasa.scripts.playback_dataset` import was already missing upstream at the fork point.

**Pass B (done, 2026-10-04).** Every removal is behaviour-checked against golden snapshots taken before the change. The snapshot scripts live outside the repo, in `robotalk-cleanup-scratch/golden*.py`.

| Commit | Removed | Check |
|---|---|---|
| `b9483e7` | The 82 stale tests, which failed identically on the snapshot | Suite green (761) |
| `d66e4f0` | The High Temp, Random/UUID and Verbalized sampling strategies, and the `--verbalized-k` plumbing | Suite green |
| `1268ab2` | Work partitions: selector, flags, FSM partition checks, prompt rules, retry-feedback branches, and `work_partitions` in 49 specs | 212 production generation and SFT prompts byte-identical; 530 dataset trajectories plus 1,590 mutants give identical validator verdicts |
| `4e10706` | Centralized history, agent prediction, task_complete, the causal single cache, step-index modes, the `cache` observation mode, the flag to turn off `get_image` training, the plain SFT format, and the centralized live runner | 10,284 SFT examples byte-identical; prompts and verdicts unchanged; the prompt-consistency gate passes on the paper cohort |
| `3c478d5` | Orphaned helpers: agent-argument schemas, the task_complete tool, `pop_agent_argument` | Suite green (696) |

Two side effects of `4e10706`:
- Preprocessed artifacts and example caches are stamped `train_example_format=private_history_consume_once`, so artifacts built by the old code are rejected.
- A latent `NameError` in the collator's empty-label check is fixed.

**Pass C (done, 2026-10-04).** This pass covers generation.
- A third golden snapshot, `golden_cascade.py`, rebuilds 318 recorded trajectories' task instances. All 318 initial states match exactly. It then replays each trajectory's ticks through the cascade's prompt, response schema, parse and `_validate_candidate` path, plus 2 mutations per trajectory.
- The first golden check (`golden.py`) also stayed unchanged throughout.

| Commit | Removed |
|---|---|
| `ed35129` | The pre-cascade generator: raw CLI, orchestrator, on-demand and Vertex-batch runners, costs, output writers, error summaries. `runtime_support` goes from 59 definitions to 12, and `progress.py` is reduced to the render sweep's components. |
| `e5430e0` | Legacy/simplified/simplified_v2 prompt styles, the flat (non-tick) path, the `base` registry entry, and unread `RuntimeConfig` fields |
| this commit | The Azure OpenAI client and the pricing layer |

**Test status after Pass A.**
- Run on a compute node, over all tracked `tests/test_*.py` and `training/bc_task_vlm/tests/test_*.py`.
- Every remaining failure is **pre-existing**: the identical files at `pre-publication-snapshot` give 65 failed / 19 errors.
- After Pass A, the failures are exactly that baseline set, minus the 3 tests of removed features (deleted) and the 2 removed cases.
- The 82 inherited failures and errors are concentrated in:

| File | Failures + errors |
|---|---|
| `test_sim_tool_executor` | 33 |
| `test_bc_task_vlm_preprocessed_data` | 11 |
| `test_occupancy_grid*` | 11 |
| `test_bc_task_vlm_dataset` | 7 |
| `test_task_level_trajectory_generation` | 4 |
| `test_task_level_grounding` | 3 |
| `test_wait_release_rule`, `test_trajectory_adapter`, `test_sim_tool_specs`, `test_bc_task_vlm_partial_observability`, `test_bc_task_vlm_live_sim_parallel_eval` | 2 each |
| 5 other files | 1 each |

- These are stale tests, written against older behaviour. Step 9 must:
  - triage each one as stale (update or delete) or as a real bug (fix code);
  - get the suite green before release.
- Login-node caveat: some tests need a compute node, because a native extension raises SIGILL on the login CPU.
