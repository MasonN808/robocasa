# SFT-necessity experiment: multi-agent tool-calling VLMs

Does supervised fine-tuning (SFT) buy anything over a strong out-of-the-box VLM on
RoboCasa two-agent kitchen tasks? This directory holds the full pipeline:
data staging, a model-agnostic eval harness, an LLM judge, and the training
scripts. This doc is the map.

## 1. The question and the design

Each task is a two-robot kitchen scenario (e.g. "garnish the cake"). A
demonstration is a **trajectory**: an interleaved sequence of tool calls by two
agents — physical actions (`navigate_to_fixture`, `pick_up_object`,
`place_on_object`, …) and `communicate` steps where one agent announces its
plan or a hand-off in free text.

Evaluation is **teacher-forced next-step prediction**: given the task, the
ground-truth history of all prior steps, and the acting agent's most recent
camera observation (3 images: top view, room view, map), the model predicts
the **next** tool call as JSON. Every model sees byte-identical prompts.

**Two eval splits** (defined relative to what the SFT model trains on):
- `heldout_trajectories` — 75 unseen trajectories of the **47 training tasks**.
  Measures within-task generalization.
- `heldout_tasks` — 75 trajectories of **5 tasks fully excluded** from SFT.
  Measures transfer to novel tasks.

For out-of-the-box models nothing is truly "held out"; the splits are just two fixed
sample sets scored under identical rules so all models are comparable. Held-out
tasks are intrinsically harder (they cover different tools/fixtures), so only
compare **within a split**.

Held-out tasks: `garnish_cake`, `meat_skewer_assembly`, `hot_dog_setup`,
`cluster_items_for_clearing`, `beverage_organization` (selected so every tool
is covered by ≥2 training tasks — see
`data_analysis/select_held_out_tasks.py` and
`data_analysis/analysis/held_out_task_selection/`).

## 2. Metrics

Per teacher-forced step (parsing strips any `<think>…</think>` block first):

| Metric | Meaning |
|---|---|
| `parse_rate` | output was valid JSON (formatting only) |
| `valid_rate` | call also passes schema/task validation |
| `tool_name_acc` | predicted tool matches expert (args ignored) |
| `exact_call_acc` | tool **and** all args match |
| `action_tool_acc` / `action_exact_acc` | the two above, **physical steps only** |
| `comm_fraction` | share of steps whose ground truth is `communicate` |
| `comm_tool_rate` | on communicate steps, model chose to communicate |
| `comm_judged_acc` | on communicate steps, **LLM judge** rates the message equivalent to the expert's (paraphrase OK) — free text can't exact-match |
| `judged_overall_acc` | action exact-match + judged communicate-match, over all steps — the headline "communication-fair" number |
| `traj_all_steps` | trajectories with **every** step correct (strict sequencing) |
| `traj_prefix` | mean fraction of a trajectory correct before the first error |

Scoring is sequencing-aware: under teacher forcing the "correct" answer at each
step is what the expert did *at that point in the order*. Predicting a sensible
action where the expert first does a `communicate` handshake scores wrong.
Caveat: teacher forcing penalizes valid alternative orderings, so absolute
numbers understate raw competence — but the out-of-the-box-vs-fine-tuned **gap** is fair.

## 3. Pipeline / file map

**Data staging** (`training/scripts/`):
- `stage_experiment_tasks_to_tar.sh` — download an HF sweep dataset and store
  it as ONE tar on `/work` (inode-safe: `/work` is file-count limited).
- `tar_extracted_tasks.sh` — tar an already-extracted task, reclaim inodes.
- `assemble_dataset_root.sh` — build the 52-task dataset root on node-local
  `/tmp` (symlink local tasks, untar staged tasks) so nothing extracted lands
  on `/work`.
- `build_manifests_job.sh` — assemble root → `build_eval_manifest.py` → export
  the ~5 GB eval subset referenced by the manifests.

**Eval harness** (`training/bc_task_vlm/`):
- `eval_standalone.py` — the driver. Backends: `gemini` (Vertex, forced JSON via
  `response_schema`, thinking budget, temperature) and `hf` (local, forced JSON
  via xgrammar, `--enable-thinking`, `--temperature` sampling). Prompt variants:
  `--task-spec-detail`, `--few-shot`. Resume-safe (regenerates failed records,
  refuses to pool mismatched configs, filters+dedupes to the manifest).
- `prompting.py` — prompt construction (shared by train and eval).
- `schema_utils.py` / `tool_calling.py` — response-schema build + validation;
  honor `tool_args` / `optional_tool_args` / `tool_arg_any_of`.
- `evaluation.py` / `metrics.py` — scorer, comm/action split, reasoning
  stripping, trajectory aggregates.
- `judge_communications.py` — LLM-as-judge for communicate steps.
- `export_results_table.py` — collate every run dir → `results_table.{csv,md}`.
- `plot_model_comparison.py` — grouped-bar figure with Wilson CIs.

**Training**:
- `training/scripts/training_qwen36_27B_sft.sh` — 27B LoRA SFT on the cluster
  (4×H200, tool_call format, 512² images, tar-assembled root).
- `training/scripts/local/` — single-GPU LoRA SFT on a trajectory subset
  (stage → rsync → probe → train). See its README.

## 4. Eval matrix (models × configs)

✅ done · ⏳ running/queued · ❌ not run (see notes)

| Model | baseline | +spec | thinking | temp 0.7 | +few-shot | +spec+few-shot | think+spec+few-shot |
|---|---|---|---|---|---|---|---|
| Gemini 3 Flash | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| Gemini 3.5 Flash | ✅ | ✅ | ✅ | ❌ | ✅ | ✅ | ❌ |
| Gemini 3.1 Pro | ✅¹ | ✅¹ | —¹ | ❌ | ❌ | ❌ | ✅ |
| Qwen3.6-27B base | ✅ | ✅ | ⏳ | ⏳ | ✅ | ⏳ | ⏳ |
| Qwen3.6-27B SFT | ⏳² | — | — | — | — | — | — |

¹ Pro's `thinking_budget=0` produces empty responses ~55–66% of the time under
forced JSON (verified, reproducible), so Pro's baseline/+spec runs use default
thinking and it has no thinking-off data point.
² SFT smoke + full run queued on the cluster; a local Qwen3-VL-8B SFT is the
faster alternative (`training/scripts/local/`).

Not-run configs were skipped deliberately: temperature is a no-op at headline
scale (0.505 vs 0.524 greedy), thinking-on-top-of-everything *hurt* on 3 Flash
(0.641 → 0.620, communication collapsed), and Pro is ~6× the cost with zero
spec gain.

## 5. Key findings (out-of-the-box models, corrected judge)

Full numbers in `training/bc_task_vlm/eval_runs/results_table.{csv,md}`.

- **Trajectory-level all-steps-correct is 0.000 in every out-of-the-box config**
  (4 models × 7 prompt variants × 2 splits). No frontier model completes a
  single one of 150 trajectories. Step accuracy improves with aids; sequential
  competence does not.
- **Best out-of-the-box ≈ 0.65 judged-overall** (3.5 Flash / 3 Flash +spec+few-shot,
  held-out trajectories). This is the bar SFT must clear.
- **Task spec helps weak models, not strong ones**: it doubles 3 Flash's
  comm-selection (0.30 → 0.52) but does nothing for Pro (0.25 → 0.24) — a strong
  reasoner already infers the conventions.
- **Reasoning trades communication for action**: given a chance to think, models
  act instead of announcing the coordination handshake — exactly what the task
  penalizes.
- **Model ordering is sane**: Pro ≥ 3.5 Flash ≥ 3 Flash-with-thinking ≫ 3 Flash
  baseline on action accuracy (Pro tops out at 0.908 action-exact).

## 6. Bugs found and fixed (high-effort code review)

Verified findings, all fixed (see git history):

1. **Judge compared empty objects** — read tool args under `args` but records
   store `arguments`; every judged comparison was `{}`≡`{}`. Fixed; all runs
   re-judged. (This inflated the earlier `judged_overall` numbers.)
2. **tool_call scoring rejected valid any-of args** — `tool_call_to_single_step_payload`
   and `build_tool_schemas` only knew `tool_args`, so a perfect SFT prediction
   for placement tools scored invalid and the prompt schema contradicted the
   supervised target. Fixed via shared `declared_tool_arg_names`. (Would have
   sunk the SFT eval; no out-of-the-box run affected — all use plain format.)
3. **Resume treated failed generations as done** — transient API/GPU outages
   permanently deflated scores. Now regenerated on resume.
4. **Judge failures / config mixing** — error verdicts persisted forever;
   resume didn't check flags. Fixed (retry errors, config-mismatch guard,
   manifest filter + dedupe in finalize).
5. **Processor pixel-knob inconsistency + mid-run crash** — min/max_pixels vs
   size.*_edge disagreed; knobless processors crashed in a worker. Now pinned
   consistently; knobless warns and no-ops. (No-op on this dataset — all images
   are 6000×4800, above the cap.)
6. **`rm -rf "$HF_HOME"` trap** in a staging script could delete the shared
   group cache. Now only ever deletes a job-owned `/tmp` dir.

## 7. Plan / status

- [x] **Stage 0** — held-out task selection; publish 3 local-only tasks to HF.
- [x] **Stage 1** — model-agnostic eval harness; both backends verified with
      byte-identical prompts.
- [x] **Stage 2** — tar-staging (inode-safe); manifests + eval subset;
      leakage guards pass.
- [x] **Stage 2.5** — pilot ablations (forced JSON, thinking, temp, few-shot).
- [x] **Stage 3a** — out-of-the-box headline evals: 4 models × up to 7 prompt
      variants (see matrix). LLM-judge on all.
- [x] **Reviews** — physical/communication metric split; LLM judge;
      task-spec-detail prompt; high-effort bug review (6 fixes).
- [x] **Qwen thinking + temperature** wired into the HF backend.
- [x] **Local SFT scripts** (`training/scripts/local/`).
- [ ] **Stage 3b** — finish remaining Qwen variants (thinking, spec+few-shot,
      think+spec+few-shot, temp 0.7) + judges. *(running)*
- [ ] **Stage 4** — SFT the model. Cluster: `training_qwen36_27B_sft.sh`
      (smoke + full, queued). Local: Qwen3-VL-8B LoRA on a subset.
- [ ] **Stage 4b** — evaluate the SFT model on both splits (tool_call format)
      + judge.
- [ ] **Stage 5** — regenerate `results_table` + comparison figure with the
      SFT rows; write up (headline: does SFT move `traj_all_steps` off 0.000).

## 8. Environment notes (NCSA Delta)

- `/work` is **inode-limited** — keep bulk data as tars, extract only to
  node-local `/tmp` inside jobs.
- HF cache: `/work/hdd/bgjs/.cache/huggingface` (shared group cache). Never let
  a cleanup trap target it.
- Vertex Gemini needs `GOOGLE_CLOUD_LOCATION=global`; regional endpoints 404.
- Login-node torch/transformers imports over Lustre take minutes — run as jobs.
- SLURM: CPU → `bgjs-delta-cpu`/`cpu`; GPU → `bgjs-delta-gpu`/`gpuH200x8`.
