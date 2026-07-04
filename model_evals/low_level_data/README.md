# Low-Level VLA Data Collection

This package collects low-level physical-control segments from task-level trajectories and exports them for model fine-tuning.

## Collection Modes

### VLA rollouts

Run a pretrained backend on every selected physical tool call, save raw segments, and optionally filter by predicate success.

```bash
python model_evals/low_level_data/collect_vla.py \
  --backend rldx \
  --trajectory_json data_generation/task_level/data/.../verbalized/deliver_straw/trajectories/traj_000000.json \
  --task DeliverStraw \
  --log_dir model_evals/data/rldx_deliver_straw \
  --tools all_physical \
  --save_policy success_only
```

Batch over a folder:

```bash
python model_evals/low_level_data/collect_vla_folder.py \
  --backend rldx \
  --trajectory_root data_generation/task_level/data/.../verbalized \
  --trajectory_indices 0 1 2 \
  --num_trials 3 \
  --tools all_physical \
  --save_policy success_only \
  --log_dir model_evals/data/rldx_low_level_collection
```

`--save_policy` options:

- `success_only`: only predicate-successful VLA rollouts are marked accepted.
- `all`: every rollout with recorded observations is accepted.
- `manual_review`: rollouts are recorded but not accepted until reviewed.

### Teleop

Teleop collection is operator-gated. Predicate success is logged, but save/discard is decided by the teleoperator.

```bash
python model_evals/low_level_data/collect_teleop.py \
  --trajectory_json data_generation/task_level/data/.../verbalized/deliver_straw/trajectories/traj_000000.json \
  --task DeliverStraw \
  --log_dir model_evals/data/teleop_deliver_straw \
  --tools all_physical
```

The initial teleop implementation is line-mode:

- `wasd/qe`: translate end effector
- `ijkl/uo`: rotate end effector
- `g`: close gripper
- `h`: open gripper
- `raw <12 floats>`: submit an exact RoboCasa 12D action
- `done`: finish segment
- `abort`: discard immediately

After each segment, choose:

- `s`: save
- `d`: discard
- `r`: retry
- `q`: quit

## Raw Segment Format

Collectors write raw model-agnostic segments under `segments/`:

```text
segment_000000/
  metadata.json
  arrays/
    states.npy
    actions_hdf5_order.npy
    actions_lerobot_order.npy
    robot0_agentview_left.npy        # when captured
    robot0_agentview_right.npy       # when captured
    robot0_eye_in_hand.npy           # when captured
  videos/
    robot0_agentview_left.mp4        # when captured
    robot0_agentview_right.mp4       # when captured
    robot0_eye_in_hand.mp4           # when captured
```

Even if the original acting agent is `agent_1`, exported model data is canonicalized to single-robot `robot0_*` keys, because the pretrained RoboCasa VLAs use that schema.

## Review

For `manual_review` or later filtering:

```bash
python model_evals/low_level_data/review_segments.py \
  --segments_dir model_evals/data/rldx_low_level_collection/collection_.../segments \
  --output accepted_segments.jsonl
```

The review tool updates `metadata.json` in place.

## Exporters

### Canonical LeRobot

```bash
python model_evals/low_level_data/export/lerobot.py \
  --segments_dir model_evals/data/.../segments \
  --output_dir model_evals/data/exports/low_level_lerobot
```

This writes RoboCasa PandaOmron LeRobot data with:

- `observation.state`: 16D state
- `action`: 12D LeRobot/PandaOmron action order
- `observation.images.robot0_agentview_left`
- `observation.images.robot0_agentview_right`
- `observation.images.robot0_eye_in_hand`
- `annotation.human.task_description`
- `meta/modality.json`
- `meta/stats.json`

### GR00T N1.5

```bash
python model_evals/low_level_data/export/gr00t.py \
  --segments_dir model_evals/data/.../segments \
  --output_dir model_evals/data/exports/gr00t_low_level
```

The result is the canonical LeRobot export plus a `gr00t_dataset_snippet.json`. Add the path to `DATASET_SOUP_REGISTRY` and train with `data_config=panda_omron`.

### pi0.5 / OpenPI

```bash
python model_evals/low_level_data/export/pi05.py \
  --segments_dir model_evals/data/.../segments \
  --output_dir model_evals/data/exports/pi05_low_level
```

The dataset uses the same LeRobot + PandaOmron modality schema that `external/openpi/src/openpi/groot_utils/groot_openpi_dataset.py` reads. Add it to `DATASET_SOUP_REGISTRY`, compute norm stats, then train with an OpenPI pi0.5 RoboCasa config.

### RLDX-1

```bash
python model_evals/low_level_data/export/rldx.py \
  --segments_dir model_evals/data/.../segments \
  --output_dir model_evals/data/exports/rldx_low_level
```

This writes the canonical LeRobot dataset and a generated `rldx_robocasa_low_level_config.py` that maps RLDX to the canonical RoboCasa camera names.

### GWP

```bash
python model_evals/low_level_data/export/gwp.py \
  --segments_dir model_evals/data/.../segments \
  --output_dir model_evals/data/exports/gwp_low_level
```

The local GWP code is inference-only, so this exporter writes the runtime schema observed by the server and a matching `norm_stats_delta.json`. Validate it against the upstream GWP training code before launching fine-tuning.
