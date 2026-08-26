# Training and Evaluation Job Tracker

Last refreshed: August 25, 2026, 10:08 PM EDT

## Technical summary

The 150-trajectories/task regularization evaluations are complete. Based on their error-free success rates, a new 43/10 scale sweep using learning rate `1e-4` and weight decay `0.01` was launched for 30, 60, 90, 120, and 150 trajectories/task. The 30/60/90/120 non-rationale training runs and the Qwen3-VL-8B-Thinking rationale-supervised 30/task run have completed. The 150/task non-rationale run is still training; its successful completion will release the staged half/full-checkpoint evaluations.

This tracker covers substantive training and evaluation experiments under the current tick-based, no-step-index pipeline. Smoke tests, infrastructure retries, and obsolete prompt versions are summarized separately rather than mixed with usable experiment results.

## Running now

| Slurm task/job | Experiment | Progress | Purpose |
|---|---|---|---|
| `515471` | 43/10 non-rationale, 150/task, LR `1e-4`, WD `0.01` | Running on 4 B200 GPUs; started August 25 at 2:54 PM EDT | Final scale-sweep training dependency before evaluations |

## Directly queued

| Slurm task/job | Experiment | Checkpoint | Split or purpose | Waiting on |
|---|---|---:|---|---|
| `515473` | Regularized scale-sweep stage 3 | — | Launches 30/60/90 half/full-checkpoint evaluations, then stage 4 | Successful completion of 150/task training job `515471` (the other dependencies have completed) |

## Dependency-queued experiment program

The hardened staged launchers have completed the 30/60/90/120 non-rationale training jobs and the requested Thinking+rationale run. Job `515471` is the only remaining training dependency. Every remaining launcher retries transient `QOSMaxGRESPerUser` rejections for up to three hours and submits at most 12 GPUs per stage. Later evaluation job IDs are created only when the preceding stage releases the user's GPU quota.

| Order | Model and supervision | Task split | Trajectories/task | Training | Evaluations queued afterward |
|---:|---|---|---:|---|---|
| 1 | Qwen3-VL-8B-Instruct, tool-call targets only | 43/10 | 30, 60, 90 | LR `1e-4`, WD `0.01`, 1 epoch; checkpoints at 0.5 and 1.0 | Both checkpoints × both splits after all training completes |
| 2 | Qwen3-VL-8B-Instruct, tool-call targets only | 43/10 | 120, 150 | LR `1e-4`, WD `0.01`, 1 epoch; checkpoints at 0.5 and 1.0 | Both checkpoints × both splits after all training completes |
| 3 | Qwen3-VL-8B-Thinking, rationale + tool-call targets | 43/10 | 30 | LR `1e-4`, WD `0.01`, 1 epoch; checkpoints at 0.5 and 1.0 | Both checkpoints × both splits after all training completes |
| 4 | Qwen3-VL-8B-Instruct evaluation | 43/10 | 30, 60, 90 | Fixed prompt-v9 cohort, 10 episodes/task | Three four-task evaluation arrays |
| 5 | Qwen3-VL-8B-Instruct and Thinking evaluation | 43/10 | Instruct 120/150; Thinking+rationale 30 | Fixed prompt-v9 cohort, 10 episodes/task | Three four-task evaluation arrays |

## Completed training runs

### Current prompt-v9 scale and supervision experiments

| Model | Task split | Trajectories/task | Supervision | Training completed | Saved/evaluated checkpoints | Slurm job(s) or run family |
|---|---|---:|---|---|---|---|
| Qwen3-VL-8B-Instruct | 47/6 | 30 | Tool call only | 2 epochs | 0.5, 1.0, 1.5, 2.0 | `441534`; `qwen3vl8b-tick150-scale30-*` |
| Qwen3-VL-8B-Instruct | 47/6 | 60 | Tool call only | 1.5 usable epochs; original 2-epoch allocation timed out | 0.5, 1.0, 1.5 | `441535`; `qwen3vl8b-tick150-scale60-*` |
| Qwen3-VL-8B-Instruct | 47/6 | 90 | Tool call only | 1.5 usable epochs; original 2-epoch allocation timed out | 0.5, 1.0, 1.5 | `441536`; `qwen3vl8b-tick150-scale90-*` |
| Qwen3-VL-8B-Instruct | 47/6 | 120 | Tool call only | 1 epoch | 0.5, 1.0 | `447526`; `qwen3vl8b-tick150-scale120-*` |
| Qwen3-VL-8B-Instruct | 47/6 | 150 | Tool call only | 1 epoch | 0.5, 1.0 | `447517`; `qwen3vl8b-tick150-scale150-*` |
| Qwen3-VL-8B-Instruct | 47/6 | 30 | Rationale + tool call | 1 epoch | 0.5, 1.0 | `463802`; `qwen3vl8b-tick150-scale30-reasoning-*` |
| Qwen3-VL-8B-Instruct | 43/10 | 30 | Tool call only | 1 epoch | 0.5, 1.0 | `463820`; `qwen3vl8b-tick150-s30-noreason-train43-*` |
| Qwen3-VL-8B-Instruct | 43/10 | 30 | Rationale + tool call | 1 epoch | 0.5, 1.0 | `447734`; `qwen3vl8b-tick150-s30-reasoning-train43-*` |

### Completed 150/task regularization training

| Slurm job | Learning rate | Weight decay | LoRA rank/alpha | Epochs | Checkpoints retained |
|---|---:|---:|---|---:|---|
| `470942` | `1e-4` | `0` | 16/32 | 1 | 0.25, 0.5, 0.75, 1.0 |
| `471739` | `1e-4` | `0.01` | 16/32 | 1 | 0.25, 0.5, 0.75, 1.0 |
| `472623` | `5e-5` | `0.01` | 16/32 | 1 | 0.25, 0.5, 0.75, 1.0 |

The 0.5- and 1.0-epoch checkpoints have been evaluated on both task splits. The 0.25- and 0.75-epoch evaluation tasks were intentionally canceled.

## Completed substantive evaluations

### Fine-tuned model evaluations

| Evaluation family | Task split | Completed checkpoints | Cohort coverage |
|---|---|---|---|
| 47/6 non-rationale scale 30 | 47/6 | 0.5, 1.0, 1.5, 2.0 | Both splits, 10 episodes/task |
| 47/6 non-rationale scale 60 | 47/6 | 0.5, 1.0, 1.5 | Both splits, 10 episodes/task |
| 47/6 non-rationale scale 90 | 47/6 | 0.5, 1.0, 1.5 | Both splits, 10 episodes/task |
| 47/6 non-rationale scale 120 | 47/6 | 0.5, 1.0 | Both splits, 10 episodes/task |
| 47/6 non-rationale scale 150 | 47/6 | 0.5, 1.0 | Both splits, 10 episodes/task |
| 47/6 rationale-trained scale 30 | 47/6 | 1.0 | Both splits, 10 episodes/task; current vLLM rerun complete |
| 43/10 non-rationale scale 30 | 43/10 | 1.0 | Both splits, 10 episodes/task |
| 43/10 rationale-trained scale 30 | 43/10 | 1.0 | Both splits, 10 episodes/task; current vLLM rerun complete |
| 47/6 scale-150 regularization sweep | 47/6 | 0.5, 1.0 for three LR/WD arms | Both splits, 10 episodes/task; fixed cohort complete |

### Out-of-the-box and communication evaluations

| Model | Communication modes completed | Task coverage | Current prompt/cohort |
|---|---|---|---|
| Base Qwen3-VL-8B-Instruct | Full, minimal, unguided, none | Both 47 trained and 6 held-out task splits | Prompt v9 fixed cohort |
| Gemini 3 Flash | Full, minimal, unguided, none | Both 47 trained and 6 held-out task splits | Prompt v9 fixed cohort |
| Base Qwen3-VL-8B-Thinking | Full/default | Both 47 trained and 6 held-out task splits | Prompt v9 fixed cohort |

### Earlier completed comparison work

| Experiment family | Status | Role in current analysis |
|---|---|---|
| Corrected no-index 27/3 SFT evaluation | Complete | Historical baseline |
| Current data-derived 27/3 and 90/10 comparisons | Complete | Historical, not the fixed-cohort scale comparison |
| Communication-mode smoke suites through prompt v9 | Complete | Harness/prompt validation; not headline full-evaluation results |
| Multimodal LoRA HF versus vLLM parity probes | Complete | Confirmed the current adapter-serving path |

## Completed preparation for queued runs

| Slurm task | Artifact | Status |
|---|---|---|
| `492919_0` | 43/10 non-rationale, 60/task | Complete |
| `492919_1` | 43/10 non-rationale, 90/task | Complete |
| `492919_2` | 43/10 non-rationale, 120/task | Complete |
| `492919_3` | 43/10 non-rationale, 150/task | Complete |
| `492919_4` | 43/10 rationale, 60/task | Complete |

The 43/10 trajectory selections were checked to ensure `30 ⊂ 60 ⊂ 90 ⊂ 120 ⊂ 150` for every training task. The rationale artifacts are not pretokenized; Qwen Thinking's processor and chat template are applied during training.

## Superseded, canceled, or failed orchestration

| Job(s) | Status | Interpretation |
|---|---|---|
| `470890`, `492384` | Failed immediately with `QOSMaxGRESPerUser` | Superseded launchers; no training was lost or partially used |
| `492403` | Canceled | Replaced by the combined staged launcher `492922` |
| `492922` | Failed immediately with `QOSMaxGRESPerUser` | Created no downstream jobs; superseded for the non-rationale 43/10 scale work by launcher `511809` |
| `511813`, `511911` | Canceled before running | Replaced first to include Thinking+rationale 30/task and then by hardened launcher `511935`; no GPU work was interrupted |
| `492386_0`, `_1`, `_4`, `_5`, `_8`, `_9`, `_12`, `_13`, `_16`, `_17`, `_20`, `_21` | Canceled intentionally | Removed 0.25- and 0.75-epoch regularization evaluations |
| Earlier prompt-v5 through prompt-v8 evaluation retries | Superseded | Retained only as debugging history; prompt-v9 evaluations are the current comparable results |

## How to interpret this tracker

- **Running** means Slurm currently has an allocated node.
- **Directly queued** means a Slurm job ID exists and is waiting for a resource or dependency.
- **Dependency-queued** means the approved experiment is encoded in the staged launcher, but its final GPU job ID will be created only when the preceding stage finishes. This avoids the cluster's per-user GPU submission limit.
- **Completed evaluation** means aggregate metrics exist for the listed split and checkpoint; it does not imply that the model performed well.
- The tracker intentionally separates current prompt-v9/fixed-cohort experiments from legacy or superseded runs so incompatible results are not mistaken for direct comparisons.

## Next status update

Refresh this document after either of these events:

1. Training job `515471` completes or fails.
2. Stage launcher `515473` creates the 30/60/90 evaluation jobs and the stage-4 launcher.
