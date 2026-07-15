# Local LoRA SFT (single-GPU box, e.g. RTX 5090)

Fine-tune a small Qwen3-VL model on a subset of the 47-task experiment without
the cluster or the full ~170 GB corpus. Three steps:

## 1. Stage a subset (on the cluster)

```bash
TRAJ_PER_TASK=30 bash training/scripts/local/stage_subset.sh
```

Writes `training/bc_task_vlm/local_train_subset/<task>/traj_XXXXXX/…`
(~4 GB at 30 traj/task). Trajectories are taken in sorted order, so the subset
is deterministic.

## 2. Pull it to the local box

```bash
rsync -avhP --info=progress2 \
  <cluster-host>:/work/hdd/bgjs/dbenhamougoldfajn/robocasa/training/bc_task_vlm/local_train_subset/ \
  ~/robocasa_local_train_subset/
```

The scripts only read `data_analysis/analysis/held_out_task_selection/held_out_task_selection.json`
(committed) for the task list, so a normal clone of the repo plus the rsync'd
data is enough.

## 3. Probe, then train (on the local box)

Measure throughput first — a fresh box's samples/sec is the one number that
decides how much data fits in your window:

```bash
PROBE=1 DATA_ROOT=~/robocasa_local_train_subset \
  bash training/scripts/local/train_qwen3vl_8b.sh
```

It trains ~40 steps and prints e.g. `for 3 epochs in 9h -> ~30 traj/task`.
Re-stage with that `TRAJ_PER_TASK` if needed, then launch the real run:

```bash
DATA_ROOT=~/robocasa_local_train_subset \
  bash training/scripts/local/train_qwen3vl_8b.sh
```

Defaults: Qwen3-VL-8B, LoRA (r=16), 512² images, bf16, gradient checkpointing,
`sdpa` attention (no flash-attn needed), 3 epochs, effective batch 8
(per-device 2 × grad-accum 4). Override any via env vars (`MODEL_PATH`,
`NUM_EPOCHS`, `IMAGE_RESOLUTION`, `PER_DEVICE_BATCH_SIZE`, `MAX_STEPS`, …).

To cap the wall-clock regardless of epochs, set `MAX_STEPS`; training stops at
whichever of `--num-epochs` / `MAX_STEPS` comes first.

## Evaluate the adapter

Reuse the standalone eval harness against the exported eval subset:

```bash
python -m training.bc_task_vlm.eval_standalone \
  --backend hf --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path training/bc_task_vlm/runs/qwen3vl-8b-local-sft \
  --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_trajectories.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --sft-format tool_call --image-resolution 512 --batch-size 4
```
