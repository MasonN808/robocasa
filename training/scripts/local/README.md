# Local LoRA SFT (single-GPU box, e.g. RTX 5090)

Fine-tune a small Qwen3-VL model on a subset of the 47-task experiment on your
own GPU — no cluster, no rsync. The training tasks all live on HuggingFace, so
the subset is pulled directly.

Scripts (run in order):

| Script | Where | What |
|---|---|---|
| `stage_subset.sh` | local | download N traj/task from HF into a subset root |
| `probe_throughput.sh` | local | measure samples/sec, recommend TRAJ_PER_TASK |
| `train_qwen3vl_8b.sh` | local | the actual LoRA SFT run |
| `_launch_args.sh` | — | shared config sourced by probe + train (not run directly) |

Prereqs on the box: this repo checked out and importable, a Python env with a
CUDA torch build for your GPU plus `transformers`, `peft`, `accelerate`,
`xgrammar`, `huggingface_hub`, `datasets`, `Pillow`. Set `HF_TOKEN` if the
datasets are gated. Point `PYTHON_BIN` at your interpreter if it isn't `python`.

## 1. Stage a subset (~4 GB at 30 traj/task)

```bash
TRAJ_PER_TASK=30 SUBSET_ROOT=~/robocasa_local_train_subset \
  bash training/scripts/local/stage_subset.sh
```

Writes `~/robocasa_local_train_subset/<task>/traj_XXXXXX/…`. Deterministic
(first N episodes per task), resumable.

## 2. Probe throughput (5–10 min)

A fresh box's samples/sec is the one number that decides how much data fits in
your window, so measure it before committing:

```bash
DATA_ROOT=~/robocasa_local_train_subset \
  bash training/scripts/local/probe_throughput.sh
```

Prints e.g. `3 epochs in 9h -> ~30 traj/task`. If it suggests a different
number, re-run step 1 with that `TRAJ_PER_TASK`.

## 3. Train

```bash
DATA_ROOT=~/robocasa_local_train_subset \
  bash training/scripts/local/train_qwen3vl_8b.sh
```

Defaults: Qwen3-VL-8B, LoRA (r=16), 512² images, bf16, gradient checkpointing,
`sdpa` attention, 3 epochs, effective batch 8 (per-device 2 × grad-accum 4).
Override via env vars. Set `MAX_STEPS` to hard-cap wall-clock; training stops at
whichever of `--num-epochs` / `MAX_STEPS` comes first.

## 4. Evaluate the adapter

Reuse the standalone eval harness against the exported eval subset (which you
can also stage locally with `stage_subset.sh` pointed at the eval manifests, or
copy from the cluster):

```bash
python -m training.bc_task_vlm.eval_standalone \
  --backend hf --model-name-or-path Qwen/Qwen3-VL-8B-Instruct \
  --adapter-path training/bc_task_vlm/runs/qwen3vl-8b-local-sft \
  --manifest training/bc_task_vlm/eval_manifests/exp52/eval_manifest_heldout_trajectories.json \
  --dataset-root training/bc_task_vlm/eval_data_subset \
  --sft-format tool_call --image-resolution 512 --batch-size 4

python -m training.bc_task_vlm.judge_communications \
  --run-dir training/bc_task_vlm/eval_runs/<your-run-dir>
```

Then add its rows to the comparison table via `export_results_table.py`.
