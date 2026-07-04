# Low-Level Trajectory Tool Probing

This package evaluates pretrained VLA backends on individual physical tool calls
inside task-level synthetic trajectories.

The synthetic executor is used only to advance the trajectory after each probe.
It does not provide controller demonstrations. For every selected physical tool
call, the runner:

1. snapshots the live simulator state before the tool,
2. runs the VLA from that state with a low-level prompt,
3. evaluates a physical success predicate,
4. restores the pre-tool state,
5. executes the synthetic tool to continue to the next trajectory step.

The VLA receives live observations from the executor env. Images are rendered on
demand from the current simulator state, matching the existing `get_image` path,
while state keys reuse the same mapping as `RoboCasaGymEnv`.

## Dry Run

Dry run walks the trajectory, builds prompts, evaluates initial predicates, and
uses synthetic tools for progression without contacting a model server.

```bash
python model_evals/low_level/eval_trajectory_tools.py \
  --backend dry \
  --trajectory_json data_generation/task_level/data/diversity_analysis/sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/verbalized/deliver_straw/trajectories/traj_000000.json \
  --task DeliverStraw \
  --max_segments 4 \
  --log_dir /tmp/robocasa_low_level_dry
```

## Model Run

Start a backend server first, then run for example RLDX:

```bash
python model_evals/low_level/eval_trajectory_tools.py \
  --backend rldx \
  --host 127.0.0.1 \
  --port 5555 \
  --trajectory_json path/to/traj_000000.json \
  --task DeliverStraw \
  --max_steps_per_probe 240 \
  --replan_steps 5 \
  --cameras default \
  --log_dir model_evals/backends/rldx/logs/low_level/debug_deliver_straw
```

Use `--tools all_physical`, `--tools primary`, or explicit tool names such as
`--tools pick_up_object place_in_receptacle`.

`--max_steps_per_probe` is the per-tool VLA horizon. The default is `240`,
which is about 12 seconds at the 20 Hz eval control rate. Increase it for
slow subtasks such as navigation or precise placement. Low-level probes run to
this full horizon by default and evaluate success at the final state. If you
want RoboCasa-style early stopping for speed, pass `--stop_on_success`; the
runner still records `first_success_step` in `stats.json` for debugging.

With `--cameras default`, videos are saved from actual simulator cameras, not
from the normalized model-input names. For a two-robot probe, the default set is
one camera for the non-acting robot plus one agent-view and one eye-in-hand
camera for the acting robot. The model request still uses normalized
`video.robot0_*` keys because the backend adapters expect the single-robot
training schema; `robot0` in those request keys means "the currently controlled
robot". Saved video filenames keep the physical simulator identity, so
`robot0_*` is always agent 0 and `robot1_*` is always agent 1.

Each probed step writes `actions.npy`, containing the postprocessed VLA action
sequence sent to the selected robot for that probe. The corresponding
`stats.json` records the controlled `robot_idx`, prompt, horizon,
`actions_shape`, initial/final predicate results, `first_success_step`, and any
synthetic-held objects present before the probe. If a previous synthetic step
picked up an object, the runner now resyncs that held object and briefly closes
the corresponding robot gripper before snapshotting the probe start state.

The `place_in_receptacle` predicate is intentionally stricter than simple XY
proximity. For object receptacles such as cups or bowls, it uses RoboCasa's
contact-based `check_obj_in_receptacle` helper and requires the object to be
released. For fixture receptacles such as drawers or cabinets, it uses
RoboCasa's fixture interior check.

## Scene Sampling

Low-level trajectory eval defaults to official-faithful scene sampling:

```text
--scene_sampling official --split pretrain --scene_seed 7
```

In batch mode, jobs are grouped by task. For each task, the runner creates one
persistent RoboCasa environment with the same seed style as the official eval,
treats every trajectory-trial pair as the next eval episode, and advances the
environment with `reset()` between episodes. For example, 5 trajectory indices
with 3 trials each become 15 official-style task episodes.

The runner records failures per trajectory-trial and continues the batch. A
failed reset, adaptation, initial-state load, probe, or synthetic advancement is
written to the job output directory instead of stopping the whole eval.

For the old trajectory-pinned behavior, use:

```text
--scene_sampling trajectory
```

Explicit `--layout`, `--style`, and `--seed` still override sampling for a
single run. Parallel batch workers preserve the same per-task episode sequence
as a single-worker run by advancing through skipped episode indices before
running their assigned jobs.

## Scene Adaptation Validation

Before running expensive VLA probes with official-style layout/style diversity,
validate that a symbolic trajectory can adapt and initialize across sampled
scenes. This utility does not contact a model server. With `--execute`, it also
runs the full adapted synthetic tool sequence.

```bash
/home/dorian/miniforge3/envs/robocasa/bin/python model_evals/low_level/validate_trajectory_scenes.py \
  --trajectory_json data_generation/task_level/data/diversity_analysis/sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/verbalized/deliver_straw/trajectories/traj_000000.json \
  --task DeliverStraw \
  --scenes 11:34:42 27:34:43 35:34:44 \
  --execute \
  --log_dir /tmp/robocasa_scene_validation
```

Outputs are written under:

```text
<log_dir>/<task>/<trajectory_id>/<timestamp>/
  summary.json
  scenes.json
  input_trajectory.json
  L<layout>_S<style>_sd<seed>/
    adapted_trajectory.json
    initial_state_load.json
```

Use this first when adding layout/style sampling to low-level eval. A scene is
usable only if adaptation, initial-state loading, and synthetic execution all
succeed.

## Folder / Batch Eval

Use `eval_trajectory_folder.py` to run the same low-level probe over a trajectory
root with task, trajectory-index, and trial controls.

Dry-run example over one task and one trajectory index:

```bash
/home/dorian/miniforge3/envs/robocasa/bin/python model_evals/low_level/eval_trajectory_folder.py \
  --backend dry \
  --trajectory_root data_generation/task_level/data/diversity_analysis/sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/verbalized \
  --tasks deliver_straw \
  --trajectory_indices 0 \
  --num_trials 1 \
  --max_segments_per_trajectory 2 \
  --log_dir /tmp/robocasa_low_level_batch_dry
```

RLDX example after starting an RLDX server:

```bash
/home/dorian/miniforge3/envs/robocasa/bin/python model_evals/low_level/eval_trajectory_folder.py \
  --backend rldx \
  --host 127.0.0.1 \
  --port 5555 \
  --trajectory_root data_generation/task_level/data/diversity_analysis/sampling_methods_data_52Tasks_30Trajectories/sampling_methods_data_52Tasks_30Trajectories/verbalized \
  --trajectory_indices 0 1 2 \
  --num_trials 3 \
  --tools all_physical \
  --log_dir model_evals/backends/rldx/logs/low_level/verbalized_idx0_2_3trials \
  --cameras default
```

Useful filters:

```text
--tasks all
--tasks deliver_straw prepare_coffee
--exclude_tasks prepare_coffee
--trajectory_indices 0 1 2
--num_trials 3
--tools all_physical
--tools primary
--tools pick_up_object place_in_receptacle
--skip_tools give_space
--max_segments_per_trajectory 5
--max_tasks 10
--max_jobs 20
--num_workers 4 --worker_id 0
```

Batch output layout:

```text
<log_dir>/batch_<timestamp>_worker<id>/
  jobs.json
  batch_summary.json
  batch_results.json
  batch_segments.jsonl
  <task_dir>/<traj_id>/trial_000/
    summary.json
    adapted_trajectory.json
    step_*/stats.json
    step_*/actions.npy
    step_*/video_*.mp4
```

`batch_summary.json` aggregates success by tool, task, and support tier. The
`batch_segments.jsonl` file has one row per probed physical tool call and is the
right input for plotting later.
