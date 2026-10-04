# Communication wait-description correction and reruns

Launched September 14, 2026 as Slurm array **853502**, using `eval_communication_waitfix_v3.sbatch`. The 12 shards use one GPU and four simulator workers each, with at most six shards running concurrently and a three-hour limit per shard.

## Corrected contract

- **Unguided:** communication available; tools describe basic message and matching-release mechanics; no prompt recommendation to communicate or opening protocol.
- **Minimal:** same tool definitions as Unguided; main prompt additionally encourages communication and describes requesting a named release immediately before waiting.
- **Intermediate:** retains detailed canonical communication/wait mechanics and additional handover/own-portion-completion guidance; removes coordination-phase argument and opening-protocol requirement.
- **Full / None:** prompt text and complete tool metadata were compared before/after and are unchanged.

The original bug confused “waits never automatically time out” with “communication is unavailable.” Only communication-disabled None now receives the permanent self-suspension description. Runtime release behavior was not changed.

`prompt_snapshots.json` stores the real PrepareCoffee/traj_000000 initial prompt for all five modes, canonical function schemas, per-mode communication/wait schema replacements, and source SHA256 hashes. Each evaluation shard verifies the reviewed source hashes before starting.

## Jobs and outputs

All paths below are under `training/bc_task_vlm/eval_runs/fixed_live_sim/`.

| Array tasks | Model | Mode | Output run |
|---|---|---|---|
| 853502_0 / _1 | Qwen3-VL-8B-Instruct | Unguided | `ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_unguided_waitfix_v3` |
| 853502_2 / _3 | Qwen3-VL-8B-Instruct | Minimal | `ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_minimal_waitfix_v3` |
| 853502_4 / _5 | Qwen3-VL-8B-Instruct | Intermediate | `ootb_qwen3vl8b_fixed10_promptv9_vllm_comm_intermediate_waitfix_v3` |
| 853502_6 / _7 | Gemini 3 Flash | Unguided | `ootb_gemini3flash_fixed10_promptv9_comm_unguided_waitfix_v3` |
| 853502_8 / _9 | Gemini 3 Flash | Minimal | `ootb_gemini3flash_fixed10_promptv9_comm_minimal_waitfix_v3` |
| 853502_10 / _11 | Gemini 3 Flash | Intermediate | `ootb_gemini3flash_fixed10_promptv9_comm_intermediate_waitfix_v3` |

The first index in each pair runs `train_task_types` (470 episodes); the second runs `heldout_task_types` (60 episodes). These retain the original 47/6 episode population, which can be regrouped to 43/10 for the existing paper artifact. Each model/mode has 530 episodes; total 3,180. Full and None are not rerun.

Settings: 10 episodes/task; evaluation seed 20260817; scene seed 20260819; canonical configuration manifest `fixed_live_sim_configuration_v1/frozen.json`; compatibility cache `new_access_state_generation_preflight/scene_compatibility_cache_v2.json`; private/no-index/consume-once context; uniform durations; FSM success; `report_failed` rejection behavior. Qwen is served without an adapter, with Instruct, thinking disabled, temperature zero, Hermes tool parser, 256 output tokens, and a 16,384-token server limit. Gemini uses `gemini-3-flash-preview` and the existing evaluator defaults.

## Validation

- 77 tests and 9 subtests passed across communication profiles, prompt parity, and concurrent FSM tests.
- Full/None prompt and tool metadata are exactly unchanged.
- All 3,180 selected episode IDs match the six earlier model/mode evaluations across both splits.
- Submission script passes `bash -n`.
- An attempted broader tool-interface test collection hit a MuJoCo illegal-instruction import on the login node. That environment failure was not a test assertion failure; the simulator-dependent tool-interface suite was not claimed as passed.

## Result handoff

Keep historical results until these runs finish. After completion, verify 470+60 unique episodes per model/mode and no missing worker output, summarize the new JSONL files, and update the communication artifact and its source mapping together. Preserve original Full and None rows. Do not replace summary files without updating `build_43_10_eval_artifact.py`'s raw-source mapping, because that builder explicitly checks aggregate and raw counts agree. The old `_intermediate_v1` and `_intermediate_v2` directories are not the corrected-wait evaluations.
# Relaunch after launcher failure

Latest replacement: **853823**. Array 853696 was rejected by the resume
configuration guard because the old output directories recorded the invalid
argument. The launcher now appends `_job${SLURM_ARRAY_JOB_ID}` to both model
run names, so each new submission uses fresh output directories. Prior outputs
are preserved. Monitor: `comm-waitfix-853823-email`, every 30 minutes.

Array 853502 failed before evaluating episodes because the launcher passed
`report_failed` instead of the CLI value `report-failed`. The launcher was
corrected and all twelve shards resubmitted as array **853696**. Reviewed
prompt/runtime source hashes remain unchanged. The 30-minute email monitor
now tracks 853696 in tmux session `comm-waitfix-853696-email`.
