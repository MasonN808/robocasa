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

**End-to-end checks after Passes B and C.**
- **Oracle replay.** I replayed 4 tasks × 2 expert trajectories (including the held-out ArrangeBreadBowl) in the live simulator, on both `pre-publication-snapshot` and the cleaned code. The results are identical: outcome, termination, step counts, rejections, native success, makespan, and every executed call.
- **vLLM smoke test.** I served the released Instruct 30/task adapter (`checkpoint-648`) with the paper's vLLM 0.27.1 settings and ran 6 paper-cohort episodes (PlateStoreDinner, PrepareCoffee, SetupBowls × 2) with the paper's evaluation flags. All 6 match the archived paper records on FSM success and error-free success.
  - 3 of the 6 have identical executed call sequences.
  - The other 3 diverge mid-episode, after identical earlier calls and observations, and only in the wording of a free-text message; the tool and coordination phase are the same.
  - That pattern is consistent with greedy-decoding nondeterminism under different request batching: the paper ran 4 workers on one server, the smoke test ran 1. It does not point to a prompt change, which would diverge at the first call.

## 8. Step 4 (restructure) progress

**Package move (`0d86582`).** The paper code now lives in `robotalk/`:
- `tasks`, `tools`, `generation`, `training`, `evaluation`, `analysis` and `release` subpackages;
- `configs/` holds the 43/10 split, the nested selections, the evaluation cohort, the scene compatibility cache and the accelerate config.

All golden checks are unchanged after the move.

**Reproduction entry point.**
- `configs/experiments.yaml` defines every Fig. 5–7 cell with the exact settings recovered from the paper's saved `run_config.json` / `parallel_run.json`:
  - all 12 SFT runs share every hyperparameter and differ only in base model, `train_reasoning` and scale;
  - Thinking models were evaluated with sampling (temperature 0.6, top-p 0.95, top-k 20, 2048 new tokens), so they are stochastic.
- `scripts/reproduce.py {fig3,fig5,fig6,fig7,figures} [--train]` runs download-or-train → vLLM serve → parallel eval → aggregate → plot. `scripts/slurm/reproduce.sbatch` wraps it.
- `robotalk/analysis/paper_results.py` replaces the archive-specific 43/10 artifact builder.
  - Run against the archived paper evaluation logs, it reproduces all 24 Fig. 6–7 values exactly.
  - The exporters then regenerate Figs. 3, 6 and 7 pixel-identical to the paper's PNGs.

**Fig. 5 caveat.** The paper's communication ablation was run on the 47/6 manifest and regrouped to 43/10. Configuration lists are identical per task, but episode sampling seeds include the split name. A native 43/10 rerun therefore draws different episodes for the 4 tasks that moved to held-out (40 of 100 held-out episodes). Expect agreement within CI, not identical held-out episodes. Gemini is also not bit-reproducible through the API.

**Fig. 7 adapters published (2026-10-05).** `DorianAtSchool/RoboTalk-Qwen3-VL-8B-Instruct-Rationale-30traj` and `DorianAtSchool/RoboTalk-Qwen3-VL-8B-Thinking-30traj` are public and hash-verified. Both are the epoch-1.0 checkpoint (`checkpoint-648`). Their cards carry the paper's numbers, and the Thinking card documents the parser-recovery flag used in evaluation. All 12 Fig. 6–7 adapters are now on the Hub.

**Training data on the Hub: what the published dataset contains (checked 2026-10-05).** Training reads three files per rendered trajectory: `original_trajectory.json`, `plan.json` and `metadata.json`.
- `original_trajectory.json` is fully derivable from the published `raw/` records. Re-running `robotalk.generation.image.processor.post_process_trajectory` on all 7,950 raw records reproduces the steps (inserted observation calls with their templated rationale and image paths) and the tick rows identically.
- Image files in the published media archives are the same files under the same names.
- Not in the published dataset: per-step simulator execution status. 85 steps in 80 trajectories (72 pickups, 13 navigations) failed in the render, and training skips them. This lives only in `metadata.json`. `plan.json` adds only image paths and alignment bookkeeping, both derivable.
- Pre-existing bug found: strict revalidation of 26 re-derived PlateStoreDinner trajectories crashes with `StopIteration` in `ConcurrentTaskValidator.replay` (an occupant lookup via `next()` without a default). The identical failure occurs on `pre-publication-snapshot`. Training is unaffected because it reads the stored verdict. Track it for Step 9.

**Open item: training data on the Hub.**
- `--train` needs the rendered per-trajectory layout: `original_trajectory.json`, `plan.json`, `metadata.json` and the images.
- The HF export has the images (identical files) and the raw generation records, but not those three JSON files.
- Proposed: publish them as an extra HF config (well under 1.8 GB) and add a converter that rebuilds `data/robotalk_rendered/` from the Hub.
- Evaluation from released adapters does not need the dataset; the cohort is configuration-native.

## 9. Dataset v1.1 (published 2026-10-05)

`DorianAtSchool/RoboTalk` now has two tags:
- `v1.0` = `b20be4f`: the paper's data, unchanged.
- `v1.1` = `89a7771` = `main`.

**What changed in v1.1.** 95 trajectories (1.2%) were replaced:
- **66 re-rendered.** These were stove pickup or navigation steps that failed in the August render. The current executor renders them with every step succeeding, and their plans are unchanged.
- **29 regenerated** with the production cascade, for the same task, run index, initial state and coordinator:
  - 24 PlateStoreDinner and 2 PrepareSoupServing plans entered an exclusive fixture in the tick its occupant left. That breaks the paper's rule, and the validator used at generation time missed it.
  - 3 more still failed in simulation after re-rendering.

**Replaced share by split.**
- 43 trained tasks: 65 / 6,450 (1.0%).
- 10 held-out tasks: 30 / 1,500 (2.0%).
- Nested 30–150/task training subsets: about 1% each.

**Upload.**
- One commit: 29 raw records, 95 media archives, both tables, `dataset_info.json`, and an Apache-2.0 card with a short Versions note. `preview.html` was deleted.
- All 128 uploaded files were hash-verified.
- The export diff confirmed that nothing outside the 95 changed.

**Checks.**
- The 29 regenerated trajectories pass strict revalidation, and all 95 render with every step succeeding.
- `robotalk.release.materialize_training_layout` rebuilds the training layout from the published files alone. It gives identical training examples on 835 unchanged trajectories (63,242 examples) and on all 95 repaired ones (7,998 examples).
- The validator crash behind the 26 failures (an occupant lookup that assumed `give_space`) is fixed in `2154ca0`.

**Paper-era archive.**
- The repair working folder is `robotalk_v1_1_repair/`, holding the repair sets, logs, the card source and the receipt.
- The v1.1 roots are symlink overlays: `tick53x150_state_grounded_cascade_v1_1_{raw,rendered}`.

**Explorer Space.**
- `DorianAtSchool/RoboTalk-Explorer` now serves the v1.1 bundle (Space commit `a7a243e`).
- The catalog and episode files of the 29 regenerated trajectories were updated, and episodes now carry `media_archive`.
- Media is fetched from the dataset's `v1.1` tag, not from `main` (`e0db464`).
- The unused leftovers `media/`, `data/ticks.json` and `style.css` were removed.
- All 7,954 files were hash-verified.

## 10. Step 5 (portability and hygiene)

**Paths.**
- No tracked file outside the internal planning docs mentions a cluster path, account, partition or personal address.
- `tests/test_portability.py` enforces this.

**Defaults.**
- The dataset root comes from `ROBOTALK_DATA_ROOT`, default `data/robotalk_rendered`.
- Every output goes under `ROBOTALK_OUTPUT_ROOT`, default `outputs/`, via `robotalk.utils.output_root()`. This covers training runs, `reproduce.py`, the aggregator, the figure exporters and the HF export.
- The wandb project defaults to `robotalk`, with no entity.
- `.env.example` documents every variable, and `reproduce.py` now loads `.env` too.
- Dependency messages point at the `[train]` extra (Step 6).
- The figure pipeline under a moved output root reproduces the archived `artifact.json`; only the recorded source paths differ.

**Configs.**
- Provenance paths in `configs/` were rewritten. The split and scene cache now point at their repo copies, whose hashes matched the recorded ones. Everything else is relative to the paper-era workspace and keeps its recorded SHA-256.
- The dependent hashes were updated: `selection_*.task_split_sha256` and the cohort's `scene_cache_sha256`.
- The cohort `content_hash` did not reproduce under any formula even before this step, so it was stale at commit time. It is now recomputed over the whole file.
- The evaluation content of the cohort and the scene cache is unchanged, and scene sampling does not depend on file contents.

**Release tooling.**
- `export_robotalk` requires `--raw-root` and `--rendered-root` and writes the published card (`robotalk/release/dataset_card.md`). It no longer writes `preview.html`.
- `publish_models.py` was removed. It was hard-wired to the archived run layout, and all 12 adapters are published.
- `tasks/TASKS.md`, an obsolete list of task candidates, was removed. The specs README now describes what ships, without pipeline provenance.

**Tests.**
- Corpus-backed tests read `ROBOTALK_DATA_ROOT` and skip cleanly without it. `test_render_order` previously errored at collection.
- The render fixture's image paths are relative.
- Finding: published (tick-format) trajectories are already in concurrent order (300/300 sampled), so `concurrent_step_order` is the identity on them. The old guard "the corpus really reorders" became the invariant "published trajectories are already in concurrent order".
- Suite: 574 passed on a compute node, with a 72-trajectory corpus materialized from v1.1.

**Credentials.** A regex scan found no keys or tokens. `gitleaks` is not installed on the cluster, so it moves to Step 11 on the exported tree.

## 11. Step 6 (dependencies and installation)

**Packaging.**
- A single `pyproject.toml` (`robotalk` 1.0.0, Apache-2.0, the paper's authors) packages `robotalk` and the vendored `robocasa`.
- Core dependencies are the simulator stack pinned to the paper's versions (`mujoco` 3.3.1, `numpy` 2.2.5, `numba` 0.61.2, `scipy` 1.15.3) plus what `robotalk` imports.
- Extras:

  | Extra | Contents |
  |---|---|
  | `gen` | `google-genai`, `google-auth`, `datasets` |
  | `train` | `torch` 2.7.1, `transformers` 4.57.6, `peft` 0.19.1, `accelerate` 1.14.0, … (the paper environment's versions) |
  | `flash` | prebuilt flash-attn 2.8.3.post1 wheel (Linux, Python 3.11) |
  | `dev` | `pytest` |

- Packages that nothing in the release imports were dropped: `tianshou`, `lerobot`, `pynput`, `hidapi`, `tensorboard`, …
- vLLM is not an extra. The paper ran vLLM 0.27.1 in a container, and `VLLM_LAUNCHER` selects it.
- `uv.lock` was regenerated.

**GPU assumption.**
- `torch` and `torchvision` come from the PyTorch CUDA 12.8 index, as in the paper environment.
- Those wheels cover compute capability 7.5–12.0 (Turing through Blackwell) and need a driver that supports CUDA 12.8 (≥ 570). They do not support Volta or older GPUs.
- The default PyPI wheel (CUDA 12.6) fails on Blackwell with "no kernel image is available".
- This is stated in `pyproject.toml`, and the README must state it too (Step 8).

**robosuite.**
- The `MasonN808/robosuite@dev` submodule was removed.
- The fork is upstream ARISE `aaa8b9b` plus one change that prefixes the `manipulator_mount` body name per robot, which two mobile manipulators need. Upstream master still lacks it.
- Decision (2026-10-05): mirror the fork under `DorianAtSchool/robosuite` and pin the exact commit the experiments used (`94f9aa2`, the fork's `dev`).
- A wrapper replacing the fork (`robosuite_compat`) was tried first and tested byte-identical, then dropped in favour of the exact paper code.
- Licenses: the root `LICENSE` is now Apache-2.0, RoboCasa's MIT text moved to `robocasa/LICENSE`, and a `NOTICE` was added. It points at `docs/FORK_CHANGES.md` (Step 8).

**From-scratch install** (`uv sync --frozen --all-extras` into a fresh venv):
- Tracked tests: 574 passed with the mirror pin (576 under the interim wrapper, which had 2 extra tests).
- Oracle live-sim replay (8 trajectories): identical to the cleaned code in the research env, apart from timing fields, under both the interim wrapper and the mirror.
- One LoRA training step on Qwen3-VL-8B-Instruct, on an RTX PRO 6000 (Blackwell), passed with both `sdpa` and `flash_attention_2`.
- Still to check (Step 10, on the exported tree): the RoboCasa asset download, and `pip`-only installation.

## 12. Step 8 (documentation)

**New docs.**
- `README.md` replaces the upstream RoboCasa README. Its sections:
  - installation, with a marked **GPU requirements** box;
  - a Quickstart;
  - results tables for Figs. 3 and 5–7, each with its `reproduce.py` command and measured compute;
  - repository structure, generating new data, limitations, license, citations (RoboTalk, RoboCasa365, RoboCasa, robosuite) and acknowledgements (the paper's funding text).
- Also added: `docs/{concurrency_semantics,tool_interface,data_format,evaluation}.md`, `docs/FORK_CHANGES.md` (against upstream `1b19563`) and `CITATION.cff`.

**Checks on the README.**
- Every number was recomputed from `paper_results`.
- The Fig. 3 counts come from `export_task_phase_pngs._load_counts`: 12/12/19 training, 2/2/6 held-out.
- Tool names and arguments in `tool_interface.md` were checked against `build_model_tool_specs`.
- The budget (2 × factor 2 = 4× the reference length) and termination rules were checked against `live_sim_eval`.

**Compute figures.**
- Training: `train_runtime` of the released runs on 4× B200 ranges from 1.96 h (30/task) to 9.7 h (150/task); all ten Fig. 6 adapters took 58.6 h, or 235 GPU-hours.
- Evaluation: about 3 episode-hours per Instruct model and about 4 per Thinking model (sum of `elapsed_s`), so about 1 h of wall time with 4 workers on one RTX PRO 6000.
- No Gemini token usage was recorded, so the README states no API cost.

**Quickstart verified from scratch** in the fresh env:
- `hf download --revision v1.1` of one task;
- `materialize_training_layout` (150/150);
- oracle replay with `configs/eval/oracle_smoke.json`: 2/2 reached the goal with 0 rejections, and videos were written.

**Fixes found while doing this.**
- `imageio-ffmpeg` was missing from the dependencies. Without it, episode videos crash on the PyAV plugin (`expected bytes, NoneType`). It is now added and locked.
- The "118 modified CC BY assets" were 117 absolute symlinks into another checkout's downloaded assets, plus the download's README, committed by accident in `2f1a64c`.
  - They are now untracked (kept on disk) and git-ignored.
  - `tests/test_portability.py` now fails on absolute symlinks, and it flags them on the previous commit.
  - `NOTICE`, `FORK_CHANGES.md` and the plan's license table are corrected: no asset is modified or redistributed.
- The README has no teaser image yet: there is no exported figure file. Add one from the paper sources if wanted (Step 10).

## 13. Step 9 (tests and CI)

**Layout.**
- `tests/unit` holds 34 files plus 2 new ones. They run on any CPU with core + `dev` dependencies only: no torch and no RoboCasa assets. About 580 tests run in about 2 s.
- `tests/sim` holds 7 files (172 tests). They build MuJoCo scenes, are auto-marked `sim` by `tests/conftest.py`, and need the downloaded assets.
- The `bc_task_vlm_` name prefix and the `sys.path` hacks were removed.
- Two upstream RoboCasa scripts posing as tests were deleted:
  - `test_datasets.py`, an HDF5 demo-dataset checker that collects no tests;
  - `test_tasks_validity.py`, random rollouts over all 365 upstream tasks.

**New tests.**
- `test_prompt_snapshot.py` replaces `verify_generation_training_eval_prompts.py`, whose SFT-vs-eval check called the same function twice.
  - It checks 212 generation and policy prompt hashes (53 tasks × 4 configurations) against the snapshot recorded from the paper-era code (`golden_before`).
  - A prompt change that would break the released adapters' contract now fails CI.
- `test_reproduce_dry_run.py`: every figure, with and without `--train`, must resolve offline. Every config path must exist, and every model cell must be evaluated; for example, `fig6 --train` resolves 72 commands. The existing fixture-based SFT example test (`test_dataset.py`) covers the no-network smoke item.

**Lint.**
- `ruff` 0.16.10 runs with `E9` + `F` on `robotalk/`, `scripts/` and `tests/`; the vendored `robocasa/` is excluded. It reports 0 findings.
- Fixes:
  - 121 unused imports removed;
  - 5 unused variables removed; `_require_dependency` kept as a check;
  - a `TYPE_CHECKING` import added for the sweep's `datasets` annotations;
  - the dead `robotalk/tasks/base.py` re-export facade deleted (4 names used by 3 files, which now import from the defining modules).
- Behaviour-neutral: golden snapshots g1, g2 and g3 (prompts, verdicts and mutants; cascade; 10,284 SFT examples) are byte-identical, every module imports, and unit and sim tests pass.

**CI.**
- `.github/workflows/ci.yml` runs `uv sync --frozen --extra dev`, ruff and `pytest tests/unit` on Python 3.11.
- The pre-commit hook is now ruff instead of upstream's black.

## 14. Step 10: vLLM server parity fix

**Finding.** `reproduce.py` served every model with `--tool-call-parser hermes` and no reasoning parser. The paper's servers, as their startup logs record, differed by model type:

| Cells | Tool-call parser | Reasoning parser |
|---|---|---|
| Instruct cells, incl. Instruct + rationale, and the Fig. 5 base model | `hermes` | none |
| Thinking + rationale (s30–s150) | `qwen3_xml` | `qwen3` |
| Thinking, no rationale | `qwen3_xml` | none |

- The cells were matched to their logs by exact adapter path; the Thinking-no-rationale cell was matched by its recorded port.
- The paper's servers also ran with `VLLM_USE_FLASHINFER_SAMPLER=0`, and passed `--max-lora-rank` only with an adapter.
- The earlier vLLM smoke test used an Instruct model, so it could not catch this.

**Fix.**
- `configs/experiments.yaml` now has `vllm.profile_serve_args`, `lora_serve_args` and `env`.
- `tests/unit/test_reproduce_dry_run.py` pins each cell's parsers.
- The client arguments of every recorded cell match the paper. The only differences are flags whose values are now built in (private history, no step indices, consume-once observations, trained `get_image`) and the default `--communication-mode full`.

## 15. Step 10: acceptance run on a clean export

The acceptance run uses `git archive HEAD` with a fresh `uv sync --frozen --all-extras` and a fresh asset download.

**Assets.**
- All 6 RoboCasa asset packages download and extract: 23 GB on disk; the README now says so.
- None of the 384 tracked files under `robocasa/models/assets` is changed by the extraction.

**Tests.** 768 passed, unit plus sim. The 7 skips need a local dataset.

**Fig. 3.** It failed on a fresh clone, and two fixes were needed:
- The exporter now creates its output folder; `tests/unit/test_fig3.py` covers this.
- `matplotlib` resolved to 3.10.8 instead of the paper environment's 3.11.1, and the figure-styling code depends on 3.11.
  - All direct dependencies are now pinned to the paper environment's versions.
  - That also corrected `opencv-python` (4.13 → 5.0.0.93), which the live-sim image pipeline uses, plus `pyarrow`, `safetensors`, `tqdm`, `lxml` and `imageio`.
  - The 31 remaining lock-versus-paper-env differences are patch-level utility packages (HTTP clients, certificates, fonts, pydantic).
- After both fixes, Fig. 3 rebuilds byte-identical to the paper-equivalent render.

**Fig. 7 from the released adapters** (clean export, `reproduce.py fig7`, 1× RTX PRO 6000, 4 h 45 min). Every cell falls inside the reproduced 95% Wilson interval of the paper value:

| Cell | Reproduced | Paper |
|---|---|---|
| Instruct, train | 349/430 | 352/430 |
| Instruct, held-out | 45/100 | 45/100 |
| Instruct + rationale, train | 361/430 | 355/430 |
| Instruct + rationale, held-out | 59/100 | 58/100 |
| Thinking, train | 316/430 | 308/430 |
| Thinking, held-out | 50/100 | 50/100 |
| Thinking + rationale, train | 351/430 | 345/430 |
| Thinking + rationale, held-out | 68/100 | 71/100 |

The vLLM server logs confirm the paper's parsers for each profile.

**Plotting bug found.**
- `reproduce.py fig7` crashed when plotting: the shared Fig. 6/7 exporter drew every panel and failed on the missing Fig. 6 cells. Under `set -e` that also ended the job before the spot-checks.
- The exporter now draws only panels whose cells are all evaluated; `tests/unit/test_partial_figures.py` covers this.
- The spot-checks were resubmitted as job 1199098.

**Spot-checks** (job 1199098):

| Cell | Reproduced | Paper | Reproduced 95% interval |
|---|---|---|---|
| Fig. 6, Thinking + rationale 120/task, train | 391/430 | 388/430 | 87.8–93.3% |
| Fig. 6, Thinking + rationale 120/task, held-out | 73/100 | 77/100 | 63.6–80.7% |
| Fig. 5, base Qwen Minimal, train | 5/430 | 5/430 | exact |
| Fig. 5, base Qwen Minimal, held-out | 0/100 | 0/100 | exact |

**Second plotting bug found.** The job again failed at plotting. `reproduce.py` decided whether a figure was complete from the `--models` subset only, so a one-model run plotted an incomplete figure.
- Completeness is now checked against every cell of the figure.
- `test_partial_figures.py::test_a_single_model_run_skips_the_plot_of_an_incomplete_figure` covers it.
- In the export, `fig5 --models …`, `fig6 --models …` and `fig7` all exit 0: evaluated cells are skipped and incomplete figures are not plotted.

### Step 10 status

**Done** (clean export of `HEAD`, fresh env and assets, this cluster):
- tests: 768 passed;
- Fig. 3 byte-identical;
- Fig. 7 from the released adapters: all 8 cells within the 95% intervals;
- the Fig. 6 and Fig. 5 spot-checks above.

**Not done:**
- `dataset-stats`: there is no such command; the dataset counts are covered by the dataset card and the loader.
- A non-cluster machine: not available to me.
- The co-author README walkthrough: for the authors.
- Gemini Fig. 5 cells: skipped to avoid API cost.

**Fixed along the way:**
- vLLM parsers (`cef8e03`);
- dependency pins and the Fig. 3 output directory (`7a827d0`);
- partial-figure plotting (`916acca` and this commit).

**Gemini smoke** (requested 2026-10-06; minimal API use):
- Command: `reproduce.py fig5 --models gemini_full --tasks add_lemon_to_fish,garnish_cake --episodes-per-task 1`, i.e. 2 episodes, one per split.
- First attempt: it failed before any API call. `reproduce.py` passed the whole `--tasks` list to each split's evaluator, which rejects tasks outside its split. Each split now gets only its own tasks; a dry-run test covers this.
- Second attempt: both episodes ran with native function calling, the full opening handshake and FSM checks.
  - `add_lemon_to_fish` reached the goal, with 1 rejected call.
  - `garnish_cake` ended in a mutual-wait deadlock.
- No token usage is recorded by the evaluator.
- **Consecutive episodes:** the same command with `--episodes-per-task 2`, 4 episodes. Each task's two episodes ran in one worker process.
  - `add_lemon_to_fish`: success, then success in a new scene.
  - `garnish_cake`: mutual-wait deadlock, then success in a new scene with 0 rejections.
  - Episode 2 starts from a clean state: a fresh handshake under the other coordinator, and no waits left over from the deadlocked episode.
