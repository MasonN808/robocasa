# RoboTalk code release: expectations and cleanup plan

**Date:** 2026-10-03
**Working branch:** `concurrent-FSM-pipeline-and-comm-ablations` at `af38f5f`, plus uncommitted changes.
**Paper:** *RoboTalk: Learning Multi-Robot Communication and Coordination from Multimodal Demonstrations*, arXiv:2609.23997. The PDF is at [`docs/paper/RoboTalk_arxiv.pdf`](paper/RoboTalk_arxiv.pdf) and ships in the public repo.
**Companion reference:** [`robotalk_project_and_implementation.md`](robotalk_project_and_implementation.md), the internal implementation reference.

This document has two parts:

- **Part A** sets out what a publishable codebase for this paper must provide. It draws on comparable releases and community guidelines.
- **Part B** is the ordered work plan for getting this repository there. Step 1 freezes the current state so that nothing removed during trimming is lost.

---

## Part A — What a publishable research codebase looks like

### A.1 Reference releases

| Repository | What to borrow |
|---|---|
| [robocasa/robocasa](https://github.com/robocasa/robocasa) (our upstream) | Single installable package. Assets are downloaded by a script, never committed. MIT license for code, CC BY 4.0 for assets and data. README order: Install, Usage, License, Citation. |
| [robocasa/robocasa-gr1-tabletop-tasks](https://github.com/robocasa/robocasa-gr1-tabletop-tasks) | The cleanest published *fork* of RoboCasa. Its README opens with "built upon RoboCasa…". It keeps the upstream layout and LICENSE, and installs robosuite separately rather than vendoring it. |
| [openvla/openvla](https://github.com/openvla/openvla) | Closest analogue for LoRA SFT of a VLM. It has a "Repository Structure" section, a literal `torchrun …` command per paper experiment, checkpoints on the HF Hub with a `from_pretrained` snippet, and a pinned `flash-attn`. |
| [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) | `uv sync` with a lockfile, `third_party/` as git submodules, named Python training configs, stated hardware requirements, and a separate license file for each upstream model. |
| [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), [octo-models/octo](https://github.com/octo-models/octo) | A dataset download script or HF Hub link, MIT code with CC BY 4.0 data, and short eval examples. |
| [huggingface/lerobot](https://github.com/huggingface/lerobot) | Dataset format: Parquet plus media on the Hub, and a dataset card. Our HF export already follows this. |
| [UMass-Foundation-Model/Co-LLM-Agents](https://github.com/UMass-Foundation-Model/Co-LLM-Agents) (CoELA) | Multi-agent LLM coordination in an embodied setting. A useful counter-example: it has no LICENSE and no pinned environment. Do not repeat that. |

Community standards:

- [Papers with Code ML Code Completeness Checklist](https://github.com/paperswithcode/releasing-research-code). It has five items: dependency spec, training code, evaluation code, pretrained models, and a README results table with the commands that produce it.
- [NeurIPS code submission policy](https://neurips.cc/public/guides/CodeSubmissionPolicy)
- [HF dataset cards](https://huggingface.co/docs/hub/datasets-cards) and [model cards](https://huggingface.co/docs/hub/model-cards)
- [Datasheets for Datasets](https://arxiv.org/abs/1803.09010)
- [GitHub `CITATION.cff`](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-citation-files)
- ACM artifact badges: *Available*, *Functional*, *Reusable* and *Results Reproduced*. *Functional* plus *Reusable* is a sensible target for us.

### A.2 Expectations checklist

A release is "complete and clean" when every item below holds.

**1. Scope matches the paper.**
- Everything needed for the paper's claims is present:
  - task specifications and simulator;
  - data generation and validation;
  - SFT;
  - live-sim evaluation;
  - metrics and figures.
- Anything the paper does not report is removed, not merely separated: DAgger, low-level policy backends, RL, work-partition experiments and old tick formats.

**2. One obvious structure.**
- An installable package with subpackages for each pipeline stage.
- Separate `configs/`, `scripts/`, `tests/` and `docs/` directories.
- No experiment launchers at the repository root.

**3. Installation works from a clean machine.**
- `pyproject.toml` with optional extras: `[sim]`, `[gen]`, `[train]`, `[eval]`.
- A lockfile or pinned requirements.
- The Python version is stated, and robosuite is pinned to a commit.
- One asset-download command.
- Hardware stated: GPUs for training and vLLM, CPU or EGL for MuJoCo.

**4. Entry points.**
- One CLI or script per stage: generate → postprocess/render → preprocess → train → evaluate → summarize/plot.
- Each paper experiment is a named config, not an ad-hoc launcher.

**5. Artifacts are linked, not committed.**
- The dataset is on the HF Hub, already done as `DorianAtSchool/RoboTalk`.
- LoRA adapters are on the Hub with model cards: base model, data split, hyperparameters, license.
- Small evaluation *inputs* ship in the repo: task split, nested scale manifests, fixed cohort, and scene compatibility cache.
- Evaluation *outputs* (logs, result JSON, figures) are not shipped; they are regenerated by command.
- The repository holds no large binaries.

**6. Results are reproducible from the README.**
- A table mirrors each quantitative paper result in scope: Fig. 3 (task phases), Fig. 5 (communication ablation; conditions in Table IV), Figs. 6–7 (SFT scaling, rationale ablation), and dataset statistics.
- Each row has the one command that regenerates it.
- The fast path evaluates the released adapters; the full path also retrains.
- The figure scripts run on whatever that command writes to `outputs/`.

**7. Portable.**
- No absolute cluster paths such as `/work/umass/...`, home directories, SLURM accounts, partitions or e-mail addresses in code.
- Paths come from CLI flags, env vars or a `.env.example`.
- SLURM files are generic templates under `scripts/slurm/`, marked "adapt to your scheduler".

**8. Safe to publish.**
- No API keys or tokens in the tree *or in git history*.
- No personal e-mail addresses.
- No internal monitoring or notification scripts.
- If review is double-blind, the review copy contains no author identities: no commit history, no `DorianAtSchool` or `MasonN808` URLs.

**9. Licensed and attributed.**
- A LICENSE for our code.
- The upstream RoboCasa (MIT) and robosuite license notices are kept.
- A fork notice names the upstream commit and lists our modifications.
- Data license on the dataset card.
- Base-model (Qwen3-VL, Apache-2.0) and generator terms (Gemini outputs) are noted.

**10. Citable.**
- A BibTeX block in the README and a `CITATION.cff`.
- RoboCasa and robosuite are cited as upstreams.

**11. Tested.**
- A CPU-only unit-test subset passes in CI: FSM, concurrent scheduler, prompts, tool schemas, cohort selection, metrics.
- Simulator and GPU tests are marked and skippable.
- pre-commit with ruff/black.

**12. Honest documentation.**
- The README describes the *current* architecture: separate per-agent contexts, no step indices, FSM success.
- Limitations from §16 of the implementation reference are stated: FSM vs. native success, scene coverage, single training seed, and so on.

---

## Part B — Plan for this repository

### B.0 Current state (inventory as of 2026-10-03)

| Observation | Evidence |
|---|---|
| Uncommitted work | 16 modified tracked files (+520/−36): pipeline phases 2–3, `sim_normalization`, `communication_profiles`, `live_sim_eval`, `main`, `prompting`, and tests. About 50 untracked paths, including the paper-figure scripts and `robotalk_hf/` exporter. |
| Large untracked outputs | `reports/robotalk_hf_full` (27 GB, 402k files), `reports/robotalk_anonymous_review` (13 GB), `reports/robotalk_model_release` (1.7 GB), `reports/robotalk_hf_smoke` (24 MB), `reports/plate_store_dinner_traj088_overview` (18 MB). **These must not be committed.** |
| Paper-backing results are gitignored | `training/bc_task_vlm/eval_runs/` (37 GB) is ignored. That includes `fixed_live_sim_43_10_results_artifact/` (811 KB), which backs the paper's numbers. |
| Root clutter | 259 `.sbatch` files and 27 one-off `.py` files at the repository root. Also 6 planning notes (`HIGH_LEVEL_PLAN.md`, `WAIT_FOR_SIGNAL_PLAN.md`, `GENERATION_SPEEDUP.md`, …), `typescript`, `nvtop-x86_64.AppImage`, `MUJOCO_LOG.TXT`, `scratch/`, and the `slurm_logs/` and `artifacts/` directories. |
| Non-paper subsystems | `model_evals/` (low-level policy backends), `external/{openpi,rldx-1,Isaac-GR00T}` submodules, `policies/`, `tests/rl`, the RL directories, `data_analysis/vllm_no_flash_attn`, and roughly 40 `probe_*`/`audit_*`/`diagnose_*`/`build_*_artifact.py` scripts in `training/bc_task_vlm`. |
| Portability | 292 tracked files contain `/work/umass/...` paths, 17 of them `.py`. E-mails appear in `scripts/monitor_*_email.sh` and `scripts/sbatch/by_gpu/*`. |
| Secrets | A quick scan found no Gemini, HF or OpenAI key patterns in tracked files or history; `client.py` reads keys from the environment. A full `gitleaks` pass is still required. |
| Packaging | `pyproject.toml` and `setup.py` still describe upstream **RoboCasa365** (name `robocasa`, upstream authors), and `requirements.txt` is just `-e .`. Training and generation dependencies (transformers, peft, vllm, google-genai, …) are not declared. |
| README | Still the upstream RoboCasa README. `training/bc_task_vlm/README.md` describes an obsolete centralized setup (implementation reference §2). |
| License | The MIT LICENSE says "Copyright (c) 2026 the RoboCasa Team". There is no RoboTalk notice, and the dataset card's license is unresolved (implementation reference §16, item 12). |
| Fork lineage | The upstream base is around `1b19563` (2026-03-02). Our changes under `robocasa/` touch 142 files, 118 of them under `models/assets`. robosuite is a submodule pointing at `MasonN808/robosuite@dev`. |
| History | 332 commits mixing upstream and our work. The remote is the personal fork `MasonN808/robocasa`. |
| Tooling | `.pre-commit-config.yaml` exists. There is no `.github/` CI. |

### B.1 Decisions (resolved 2026-10-04)

1. **Release vehicle: a fresh public repository under `DorianAtSchool`, e.g. `DorianAtSchool/robotalk`.**
   - It holds a squashed snapshot of the trimmed tree, with no history imported.
   - This repository (`MasonN808/robocasa`) stays the private archive and keeps the full history.
2. **No anonymization.**
   - Author names and the `DorianAtSchool` HF and GitHub links may appear.
   - Portability rules still apply: no cluster paths, accounts, partitions or e-mail addresses (Step 5).
3. **Scope is strictly what the paper reports.** That means:
   - the multi-agent simulator with its 53 tasks, FSM and coordination mechanisms;
   - the generation pipeline with the production Structured Random sampling (the other sampling strategies and the Table III / Fig. 4 diversity analysis are deferred to Step 13);
   - the dataset export;
   - LoRA SFT: scale 30–150, Instruct/Thinking × rationale;
   - live-sim evaluation;
   - the communication-guidance ablation.

   Out of scope and removed from the release:
   - **DAgger**: `training/bc_task_vlm/dagger`, `tests/test_*dagger*`, and its launchers;
   - low-level policy backends and RL;
   - work partitions and old tick formats;
   - the cross-source and LLM-judge evaluations, unless the paper uses them;
   - the probe and audit scripts;
   - the anonymization tooling;
  - the task-spec generation pipeline (`pipeline/phase1–5`), which is outdated; only its output, the 53 verified specs, ships;
  - all off-sim (teacher-forced) evaluation, including the LLM communication judge.

4. **Licenses:** see B.1a below. These match RoboCasa365.
5. **Results are reported, not shipped.**
   - The README carries the paper's numbers (Figs. 3 and 5–7, dataset statistics).
   - No result files, eval logs or figure PNGs go into the public repo.
   - Each number and figure instead gets a short documented command; see Step 8 and the `reproduce` entry point in Step 4.
   - Before writing the README table, confirm that it uses the camera-ready numbers. Implementation reference §11.1: the communication-ablation numbers predate the `_waitfix_v3` reruns.

### B.1a Licenses

**Decision (2026-10-05): Apache-2.0 for code, dataset and adapters.** All candidate licenses were checked on 2026-10-04.

| Component | License | Notes |
|---|---|---|
| Our code (`robotalk/`, scripts) | **Apache-2.0** | Use the copyright line `Copyright (c) 2026 The RoboTalk Authors`. |
| Forked RoboCasa code (`robocasa/`) | MIT (upstream), kept with its notice | MIT code may be included in an Apache-2.0 project. Keep `Copyright (c) 2026 the RoboCasa Team` and the MIT text in `NOTICE` / `robocasa/LICENSE`. |
| RoboCasa assets we modified (118 files under `robocasa/models/assets`) | **CC BY 4.0** (upstream asset license) | CC BY requires attribution *and an indication that changes were made*. List them in `NOTICE`/`docs/FORK_CHANGES.md`. |
| robosuite (dependency, pinned) | MIT | Installed as a dependency, not vendored. |
| RoboTalk dataset (`DorianAtSchool/RoboTalk`) | **Apache-2.0** | The images are renders of CC BY 4.0 RoboCasa assets. CC BY has no share-alike clause, so Apache-2.0 is allowed with attribution, and the card credits RoboCasa365. |
| LoRA adapters | **Apache-2.0** | Inherited from Qwen3-VL-8B-Instruct/Thinking (both Apache-2.0). |

**Caveat on the Gemini-generated data.**
- The [Gemini API Additional Terms](https://ai.google.dev/gemini-api/terms) (last modified 2026-03-23) say Google does not claim ownership of generated content, and that the user is responsible for its use.
- The same terms say: *"You may not use the Services to develop models that compete with the Services."*
- Training task-specific 8B coordination policies is very unlikely to be "competing with the Gemini API", but this is not legal advice. The dataset card should:
  - state that trajectories were generated with Gemini 3 Flash/3.1 Pro;
  - note that downstream users are responsible for complying with the generator's terms.
- Ask the advisor or UMass tech-transfer whether a review is needed.

### B.2 Steps

#### Step 1 — Commit everything needed, then tag the snapshot (do this first)

Goal: a single recoverable point that contains all code and docs, but no bulk outputs.

1. Add ignore rules for the bulk outputs:
   ```gitignore
   training/bc_task_vlm/reports/robotalk_hf_full/
   training/bc_task_vlm/reports/robotalk_hf_smoke/
   training/bc_task_vlm/reports/robotalk_anonymous_review/
   training/bc_task_vlm/reports/robotalk_model_release/
   ```
   - Their *contents* already live on the HF Hub. Record the HF revision hashes from each `upload_receipt.json` in the commit message.
   - Small provenance files are worth force-adding: `robotalk_anonymous_review/{README.md,audit.json,upload_receipt.json}` and `robotalk_model_release/{manifest.json,upload_receipt.json}`.
2. Force-add the paper-backing evaluation summary despite the `eval_runs/` ignore:
   ```bash
   git add -f training/bc_task_vlm/eval_runs/fixed_live_sim_43_10_results_artifact/
   ```
   - This is for the **private archive only**; it is the single record of the paper's numbers. Results files are not shipped in the public repo (B.1, item 5).
   - Do the same for any other small `artifact.json` or `report.html` cited by the paper.
3. For the figure directory `reports/plate_store_dinner_traj088_overview/` (18 MB), commit only the selected PNG plus its generating metadata, not every camera candidate.
4. Commit in two logical commits on the current branch:
   - *a.* code, tests and launchers: the 16 modified files, new `training/bc_task_vlm/*.py`, `robotalk_hf/` (excluding `__pycache__`), new tests, and new `.sbatch` and `scripts/*`;
   - *b.* docs and small reports: `docs/robotalk_project_and_implementation.md`, this plan, `reports/communication_waitfix_v3/`, `paper_checkpoint_provenance_audit.*`, and `data_analysis/plots/task_phase_distribution/`.
5. Run the CPU test subset before committing so that the snapshot is known-good, or record which tests fail:
   ```bash
   pytest tests/test_concurrent_fsm.py tests/test_task_level_phase2.py tests/test_task_level_phase3.py \
     tests/test_sim_normalization.py training/bc_task_vlm/tests -q
   ```
6. Tag and push:
   ```bash
   git tag -a pre-publication-snapshot -m "State used for arXiv:2609.23997"
   git push origin concurrent-FSM-pipeline-and-comm-ablations --tags
   ```
   Pushing from the cluster may need the GitHub credential workaround.
7. Create the working branch for the cleanup: `git switch -c publication-cleanup`.


#### Step 2 — Fix the public scope (paper → code map)

This table is the whitelist. Anything not reachable from it, by import or by a documented command, is removed in Step 3.

| Paper element | Code that ships |
|---|---|
| Multi-agent simulator, 53 tasks, FSM | `data_generation/task_level/tasks/specs/verified/*.json`, `tasks/specs/runtime.py`, `tasks/shared/{fsm,concurrent_fsm,scheduling,workspace_semantics,prompting,instances}.py`, `subatomic_tool_{specs,calls}.py`, `grounding_specs.py`, `scene_sampling.py`, and the `robocasa/utils/{sim_tool_executor*,trajectory_runner*,trajectory_adapter,sim_tool_specs}.py` changes plus the kitchen env changes. |
| Coordination mechanisms (leader–follower, wait–release, exclusive workspaces) | Same as above, plus `communication_profiles.py`. |
| Generation pipeline and retry cascade | `generation/raw/{production_cascade,cascade_canary}.py`, `sampling/structured_random.py`, `generation/image/processor.py`, render and postprocess entry points, and the scene compatibility cache builder. |
| Fig. 3: task-phase distribution by split | `scripts/plot_task_phase_distribution.py` / `training/bc_task_vlm/export_task_phase_pngs.py`, plus the per-task phase counts from the original single-agent RoboCasa365 implementations. |
| Dataset release | `training/bc_task_vlm/robotalk_hf/{export_robotalk.py,publish_models.py,static_space}`. **Drop** `anonymize_review.py`. |
| SFT (LoRA; Instruct/Thinking × rationale; scale 30–150) | `training/bc_task_vlm/{prompting,dataset,preprocess,preprocessed_data,main,peft_compat,metrics}.py`, `build_scale_experiment_manifests.py`, `materialize_scale_artifacts.py`, `accelerate_multigpu.yaml`. |
| Live-sim evaluation, cohorts and metrics | `live_sim_eval.py`, `live_sim_parallel_eval.py`, `fixed_cohort_selection.py`, `freeze_configuration_cohort.py`, `fixed_live_sim_cohort.py`, `summarize_fixed_live_sim.py`, and the vLLM serving launcher (generalized). |
| Fig. 5 (communication ablation; conditions defined in Table IV), Fig. 6 (SFT scaling), Fig. 7 (rationale ablation) | `summarize_fixed_live_sim.py`, `export_communication_ablation_png.py`, `export_sft_scale_pngs.py`, `plot_style_utils.py`, `plot_caption_metadata.py`. Reduce `build_43_10_eval_artifact.py` to the table and plot logic only; the HTML artifact is not needed. |
| Evaluation details | The paper (§IV, Closed-Loop Evaluation) defines the budget as "four times the length of an example solution stored in the task specification, counting tool calls from both agents and observations". The code gives `--step-budget-factor` 2.0 × 2 when `get_image` is trained (`live_sim_eval.py:1578-1581`). Make 4× explicit in the eval config so it does not depend on that coupling. |
| Fig. 2 (PlateStoreDinner `traj_000088` example) and Fig. 1 (illustration) | Qualitative figures; no reproduction command is needed. Optionally keep `capture_plate_store_dinner_overview.py` as an example of rendering a dataset trajectory. |
| **Excluded (decided 2026-10-04)** | **Task-spec pipeline** (`data_generation/task_level/pipeline/`, phases 1–5, `sim_normalization.py`, and `tests/test_task_level_phase*.py`, `test_task_level_pipeline_cli.py`, `test_sim_normalization.py`): it is outdated, so only the 53 verified JSON specs it produced ship. **Off-sim (teacher-forced) evaluation**: `evaluation.py`, `eval_standalone.py`, `judge_communications.py`, `prediction_io.py`, `divergence_analysis.py`, `export_results_table.py`, the `plot_*_comparison.py` scripts, `eval_*offsim*`/`judge_*` launchers, and the structured-eval paths in `metrics.py`. **Caution:** `main.py`, `dataset.py` and `live_sim_eval.py` import from `evaluation`/`metrics`/`prediction_io`. Move the shared helpers (e.g. tool-call parsing) into `robotalk/evaluation/parsing.py` *before* deleting, and drop the in-training off-sim eval hook in `main.py`. |

**Paper ↔ archive cross-check (2026-10-04).**
- Every success rate in Figs. 5–7 matches the archived Sept 15 artifact (`eval_runs/fixed_live_sim_43_10_results_artifact/artifact.json`, commit `9ac8888`) exactly.
- These are the README numbers. Error-free successes per 430 trained-task / 100 held-out episodes:

| Result | Values (trained / held-out) |
|---|---|
| Fig. 5, Gemini 3 Flash: None / Unguided / Minimal / Intermediate / Full | 33/5 · 41/8 · 186/34 · 198/46 · 282/72 |
| Fig. 5, Qwen3-VL-8B-Instruct (untuned): same order | 1/0 · 2/0 · 5/0 · 7/0 · 0/0 |
| Fig. 6, Instruct at 30/60/90/120/150 per task | 352/45 · 378/64 · 396/55 · 400/48 · 405/65 |
| Fig. 6, Thinking+rationale at 30/60/90/120/150 per task | 345/71 · 364/72 · 393/66 · 388/77 · 398/73 |
| Fig. 7 at 30 per task: Instruct / Instruct+rationale / Thinking / Thinking+rationale | 352/45 · 355/58 · 308/50 · 345/71 |

- The implementation reference's §12 tables (Sept 6 snapshot) are **stale**. Minimal and Intermediate were replaced by the `_waitfix_v3` reruns, and Instruct 60/90 by the actual one-epoch checkpoints. Do not copy numbers from that doc.
- **Instruct 60/90 epochs.** Their source runs are labelled `ep0p5`, but `paper_checkpoint_provenance_audit.md` establishes from `trainer_state.json` that they are the true one-epoch checkpoints (1298, 1945). The directory labels were swapped. The released HF adapters point at these same checkpoints.
- **Held-out cohort.** Thinking+rationale 60/90/120 held-out numbers come from `_native43matched` reruns on the native 43/10 cohort. The reproduction path, which always uses the native cohort, therefore targets the same episodes as the paper.
- **The Thinking-without-rationale arm** was evaluated with parser recovery (`_recovery_v1`). Its released config must enable that flag, and its model card must say so.
- **Task specifications.** The README and docs refer to the 53 verified task specifications as they are. They do not mention how the specs were produced, and the task-spec generation pipeline is not released.
- **Table III / Fig. 4 (diversity) are deferred** to the post-release Step 13. The raw sampling-comparison trajectories are not on Unity: Mason computed the numbers on NCSA Delta from `sampling_methods_data_52Tasks_30Trajectories`, and the archived summary `data_analysis/plots/model_sampling_comparison/model_method_diversity_summary.csv` covers 51 tasks.

#### Step 3 — Trim (on `publication-cleanup`)

- **Repository root.** Delete:
  - all `*.sbatch`, `*.sh` and one-off `*.py` files (`order_ab.py`, `validity_ab.py`, `tick_*`, `regen_*`, `gate_*`, …);
  - the planning notes (`HIGH_LEVEL_PLAN.md`, `WAIT_FOR_SIGNAL_PLAN.md`, `GENERATION_SPEEDUP.md`, `FUTURE_AB_POLLING_WAITS.md`, `PIPELINE.md`, `CONCURRENCY_LIMITATIONS.md`; fold the relevant limitations into the docs);
  - `typescript`, `nvtop-x86_64.AppImage`, `MUJOCO_LOG.TXT`, `scratch/`, `artifacts/`, `eval_task_subsets/`, `test_google_cloud.py`, `preflight_er2.py`.
- **Out-of-scope subsystems.** Delete:
  - `training/bc_task_vlm/dagger/` and every DAgger test and launcher (`tests/test_*dagger*`, `launch_dagger_*`);
  - `model_evals/`, `policies/`, the `external/*` submodules and their `.gitmodules` entries;
  - `tests/rl` and the RL directories;
  - work-partition code and tests (`coverage_work_partitions.py`, `weight_work_partitions.py`, `tests/test_work_partitions.py`, …);
  - old tick and lock-step formats, once confirmed unused by the shipped path.
- **`scripts/`.** Delete `monitor_*`, `tmp_push_*`, `sbatch/by_gpu/*`, `cleanup_after_push.sh`, `smoke_push_and_clean.sh` and the sweep or HF-staging helpers not used by the shipped path.
- **`training/bc_task_vlm/`.** Delete:
  - the `probe_*`, `audit_*`, `diagnose_*`, `verify_*` and one-off `build_*_artifact.py` scripts;
  - `merge_thinkrat_native43_repairs.py`, `regroup_fixed_live_sim_results.py`, `capture_plate_store_dinner_overview.py` and `render_robotalk_paper_video.py`;
  - `*.md` trackers and diagnostics, `artifact_results_explorer.html`, `plans/`, `hf_repo_lists/`, `eval_data_subset/`;
  - all of `reports/` and `eval_runs/`.
- **`eval_manifests/` (4.5 MB).** Keep only the manifests that define the paper's evaluation population: the 43/10 split, the nested scale manifests, and the fixed configuration cohort. Move them to `configs/eval/`.
  - Better still, check that `freeze_configuration_cohort.py` and `build_scale_experiment_manifests.py` regenerate them byte-identically from a seed. Then ship only the seed, plus the files as a checksum test.
- **`data_analysis/`.** Delete all of it from the release, including the diversity analysis (deferred to Step 13) and `select_held_out_tasks.py`. The 43/10 split ships as a config file. Also delete the non-production sampling modes (High Temp, Random, Verbalized) from `sampling/` and the generation prompts.
- **Dead code paths** flagged in the implementation reference:
  - the legacy flat-example replay;
  - the centralized-history and step-index prompt options;
  - the non-consume-once image modes;
  - the permissive (non-terminating) rejection modes, if no paper config uses them.

  Confirm with `vulture` or coverage over the shipped commands before deleting.
- **Upstream RoboCasa.** Drop the upstream Sphinx `docs/` tree and the unrelated upstream scripts (teleop, upstream dataset tooling) not needed by our env. Link upstream docs instead. Keep only the `robocasa/` package code our environment imports.

#### Step 4 — Restructure

Target layout, following OpenVLA and openpi:

```text
robotalk/                      # github.com/DorianAtSchool/robotalk
├── README.md  LICENSE  NOTICE  CITATION.cff  pyproject.toml  uv.lock  .env.example
├── robocasa/                 # trimmed fork of upstream RoboCasa365 @ <sha>; changes listed in NOTICE / docs/FORK_CHANGES.md
├── robotalk/
│   ├── tasks/                # specs/verified/*.json, runtime, fsm, concurrent_fsm, scheduling, workspace_semantics
│   ├── tools/                # tool specs/calls, grounding
│   ├── generation/           # structured-random sampling, prompts, retry cascade, observation insertion, render/postprocess
│   ├── training/             # dataset, preprocess, prompting, main (LoRA SFT)
│   ├── evaluation/           # live_sim_eval, parallel eval, cohorts, metrics, summarize
│   ├── analysis/             # metrics summary; figure builders (Figs. 3, 5–7)
│   └── release/              # HF dataset export, adapter publishing
├── configs/
│   ├── generation/           # production cascade (structured random)
│   ├── train/                # scale{30..150} × {instruct,thinking} × {rationale,no_rationale}
│   └── eval/                 # split, cohort, comm_mode ∈ {none,unguided,minimal,intermediate,full}
├── scripts/
│   ├── download_assets.sh
│   ├── reproduce.py          # one entry point per paper result (see below)
│   └── slurm/                # 4–5 generic templates (generate, preprocess, train, vLLM eval), marked "adapt to your cluster"
├── tests/                    # unit (CPU) / sim (marked) / gpu (marked)
└── docs/                     # data format, tool interface, concurrency semantics, evaluation, limitations
```

- Moving `data_generation/task_level` and `training/bc_task_vlm` into `robotalk/` is a large rename. Do it with `git mv` in one commit, and fix imports with a scripted rewrite before any other edits.
- Collapse the roughly 40 historical training and eval launchers into the parameterized configs above.
- **`scripts/reproduce.py` is the minimal-work path.** One subcommand per paper result runs the whole chain and writes the table or figure to `outputs/`:

  ```bash
  python scripts/reproduce.py fig3                         # task-phase distribution (CPU, seconds)
  python scripts/reproduce.py fig5                         # comm ablation: Gemini 3 Flash + untuned Qwen3-VL-8B × 5 conditions
  python scripts/reproduce.py fig6 --from-adapters         # SFT scaling: eval the 10 released adapters, then plot
  python scripts/reproduce.py fig6 --train                 # same, but retrain all 10 adapters first
  python scripts/reproduce.py fig7 --from-adapters         # rationale ablation (4 arms @ 30/task)
  python scripts/reproduce.py dataset-stats                # 7,950 trajectories / 53 tasks etc. from the HF dataset (CPU)
  ```

  - Each subcommand prints the expected numbers from the paper next to the reproduced ones.
  - It also accepts `--tasks`/`--episodes-per-task` for a quick smoke-scale run.
  - Cluster users get a `--slurm` flag that submits through the generic templates; everyone else runs locally.

#### Step 5 — Portability and hygiene

- Replace every `/work/umass/...` path (292 tracked files today) with CLI flags or env vars: `ROBOTALK_DATA_ROOT`, `ROBOTALK_OUTPUT_ROOT`, `HF_HOME`, `ROBOCASA_ASSETS`, `VLLM_IMAGE`. Document them in `.env.example`.
- Add a CI guard: `git grep -nE '/work/umass|/scratch|@umass\.edu|gmail\.com|--account=|--partition='` must return nothing outside `scripts/slurm/` placeholders.
- Remove from defaults: the wandb entity and project names, SLURM account, partition and QoS names, and the container paths.
- Make the dataset path default to the HF download (`snapshot_download("DorianAtSchool/RoboTalk")`), not cluster roots.
- Run `gitleaks detect --no-git` on the exported tree. The public repo starts with fresh history, so the old history never ships.

#### Step 6 — Dependencies and installation

- Replace the upstream `setup.py` and `pyproject.toml` metadata with a single `pyproject.toml`:
  - name `robotalk`, our authors, MIT;
  - upstream credit in `NOTICE`;
  - remove upstream-only dependencies we don't import (`tianshou`, `lerobot`, `pynput`, `hidapi`, …).
- Declare the extras:
  - `[gen]`: google-genai, …
  - `[train]`: torch, transformers, peft, accelerate, qwen-vl-utils, flash-attn (pinned)
  - `[eval]`: vllm 0.27.1 (or the container tag)
  - `[analysis]`: matplotlib, …
- Regenerate `uv.lock`.
- Pin robosuite to a commit. Prefer an upstream ARISE commit if our `MasonN808/robosuite@dev` changes are not needed. Otherwise either:
  - fold the needed changes into `robocasa/` as a patch; or
  - mirror the fork under `DorianAtSchool` and pin it there. The `MasonN808` fork may not stay public.
- Test the install from scratch:
  1. a fresh `uv sync`;
  2. the asset download;
  3. replaying one HF trajectory in the sim;
  4. one training step on 2 trajectories;
  5. one eval episode against a released adapter.

#### Step 7 — Hugging Face artifacts

These are inputs for others to build on. Results files are not shipped.

- **Dataset `DorianAtSchool/RoboTalk`:**
  - `license: cc-by-4.0`;
  - a datasheet-style card covering motivation, composition, the generation process (Gemini cascade, FSM validation, acceptance stats from implementation reference §5.5), intended use, limitations, and the Gemini-terms note (B.1a);
  - the arXiv link and a link to the code repo;
  - a revision tag `v1.0` that the code pins.
- **Adapters:** publish the checkpoints behind Figures 6–7, so that `reproduce.py --from-adapters` needs no training.
  - **Already uploaded and verified (2026-09-15, `robotalk_model_release/upload_receipt.json`):** the 5 Instruct and 5 Thinking+rationale scale adapters, `DorianAtSchool/RoboTalk-Qwen3-VL-8B-{Instruct,Thinking-Rationale}-{30,60,90,120,150}traj`.
  - **Still to publish:** the 2 remaining Figure 7 arms at 30 per task (Instruct+rationale and Thinking without rationale), and the license and arXiv fields on all model cards.
  - Group them in one HF collection.
  - Each card needs `base_model`, `license: apache-2.0`, the split, hyperparameters (implementation reference §9.2) and parser flags (the Thinking-without-rationale parser recovery).
  - Reuse `robotalk_model_release/manifest.json` and verify it with `audit_paper_checkpoint_provenance.py` in the archive, before that script is trimmed.
- The DAgger adapters and any non-paper checkpoints are not published, or are removed if already uploaded.

#### Step 8 — Documentation

- **New top-level README**, in this order:
  1. title, authors, paper, dataset and model links, teaser figure (one image, under 1 MB);
  2. Installation;
  3. Quickstart (load the dataset, replay a trajectory, run one eval episode);
  4. **Results**: the paper's numbers for Figs. 5–7 and the Fig. 3 phase counts (values in the Step 2 cross-check), as markdown tables, each followed by its `reproduce.py` command with expected compute (GPU-hours, API cost) and the fast versus full path;
  5. Repository structure;
  6. Generating new data (needs a Gemini API key);
  7. Limitations;
  8. License (the B.1a table, condensed);
  9. Citation;
  10. Acknowledgements (RoboCasa365, robosuite).
- Turn the internal implementation reference into concise public docs (`docs/concurrency_semantics.md`, `docs/data_format.md`, `docs/tool_interface.md`, `docs/evaluation.md`), removing cluster paths, history notes and "do not claim" language.
- Delete the obsolete `training/bc_task_vlm/README.md`, `LIVE_SIM_EVAL.md`, `FIXED_LIVE_SIM.md` and `EXPERIMENT.md`.
- Add `NOTICE` and `docs/FORK_CHANGES.md` covering the upstream RoboCasa365 commit, our changes to `robocasa/` code, and the modified CC BY 4.0 assets.
- Add `CITATION.cff` (`preferred-citation` → arXiv:2609.23997), a BibTeX block, and upstream citations for RoboCasa365 (ICLR 2026), RoboCasa (RSS 2024) and robosuite.

#### Step 9 — Tests and CI

- Split the tests into `tests/unit` (CPU, fast), `tests/sim` (`@pytest.mark.sim`) and `tests/gpu`.
- Delete tests for removed subsystems (DAgger, RL, work partitions, sweeps, HF staging).
- Add `.github/workflows/ci.yml` running ruff, `pytest tests/unit` on Python 3.11, and the path and e-mail guard from Step 5.
- Add one smoke test that loads 2 trajectories from a tiny bundled fixture and builds SFT examples, using no network.
- Add one test that `reproduce.py <x> --dry-run` resolves every config and path.

#### Step 10 — Reproduction verification (acceptance gate)

- From a fresh clone and a fresh environment on a machine *other than* this cluster's usual setup:
  - `reproduce.py dataset-stats` and `fig3` match the paper exactly;
  - `fig7 --from-adapters` (smallest GPU cost) matches the paper within Wilson CI on each cell.
- Spot-check one `fig5` condition and one `fig6` scale point. The full sweeps are optional.
- Have a co-author who was not involved in the cleanup follow the README unaided, and record their friction points.

#### Step 11 — Release

- Create `github.com/DorianAtSchool/robotalk` (empty).
- Export the cleaned tree with no history:
  ```bash
  git archive publication-cleanup | tar -x -C ../robotalk-release
  ```
  Then run `git init` there and make a single initial commit, attributed to the authors.
- Make a final scan of the exported tree: `gitleaks`, the path and e-mail guard, and `du` to confirm no large files.
- Push, tag `v1.0`, and create a GitHub release with no result attachments. Optionally connect Zenodo for a DOI.
- Link the code from the HF dataset and model cards, the arXiv comments field, and the project page.

#### Step 12 — After release: retrain and evaluate on dataset v1.1 (Fig. 6)

Dataset v1.1 replaces 95 of the 7,950 trajectories (see the dataset card), and the released adapters were trained on v1.0.
- 30 of the 95 belong to held-out tasks and were never trained on.
- The other 65 are about 1% of the training trajectories at every scale: 14/1,290 at 30 per task, 25/2,580 at 60, 37/3,870 at 90, 50/5,160 at 120 and 65/6,450 at 150.

This step measures how the SFT scaling results hold up on v1.1, using the public repository exactly as an outside user would.

1. From a fresh clone of the public repo, run `scripts/reproduce.py fig6 --train`. It:
   - downloads dataset revision `v1.1` and rebuilds the training layout;
   - retrains the 10 scaling adapters (Instruct and Thinking+rationale at 30, 60, 90, 120 and 150 trajectories per task);
   - evaluates them on the native 43/10 cohort and regenerates Fig. 6.
2. Compare against the paper per cell, on both error-free and final FSM success with Wilson intervals. Expect agreement within CI rather than identity: about 1% of training trajectories differ, GPU training is not bit-deterministic, and the Thinking models are evaluated with sampling.
3. Report the comparison, for example as a `v1.1` results table in the README.
   - If any cell moves outside its interval, investigate before publishing new numbers.
   - Decide whether to release the v1.1 adapters alongside the paper's v1.0 adapters, under distinct repo names.
4. Not rerun:
   - Fig. 5 uses untuned models and is unaffected by the dataset version.
   - Fig. 7's two remaining arms (Instruct+rationale, Thinking without rationale at 30 per task) are kept at their paper values (author decision, 2026-10-05).

Compute: 10 trainings (one epoch each, on 4 GPUs) plus 10 × 530 live-sim episodes.

#### Step 13 — After release: diversity analysis (Table III / Fig. 4)

Run this from this archive repo, not the public one, once `DorianAtSchool/robotalk` is published.

1. Recover the sampling-comparison trajectories from Delta (Mason) or `~/Projects/robocasa/.../sampling_methods_data_52Tasks_30Trajectories`, or regenerate them for the 53 tasks.
2. Re-run `data_analysis/analyze_sampling_methods.py` (Qwen3-Embedding-4B cosine, normalized Levenshtein) and compare against Table III.
3. Decide whether it is worth adding to the public repo. If yes, add:
   - the four sampling modes;
   - the analysis module;
   - `reproduce.py table3`, scoring trajectories released as an HF dataset config `sampling_comparison`, with `--regenerate` as the Gemini path;
   - a README row stating the task count used.

### B.3 Definition of done

- [ ] Pre-publication snapshot committed, tagged and pushed to the private archive (Step 1).
- [ ] Every file in the release tree is reachable from the paper-to-code map (Step 2). DAgger and other unreported work are gone.
- [ ] Zero cluster paths, e-mails or secrets (CI-enforced).
- [ ] A fresh install and the quickstart work on a clean machine.
- [ ] Every paper number appears in the README with a single `reproduce.py` command, and the Step 10 checks pass.
- [ ] Dataset (CC BY 4.0) and adapters (Apache-2.0) are on the HF Hub with complete cards.
- [ ] LICENSE (MIT), NOTICE (RoboCasa365 and robosuite attribution, modified-asset list) and CITATION.cff are present, and CI is green.
- [ ] Public repo `DorianAtSchool/robotalk` is created from a history-free export and tagged `v1.0`.
