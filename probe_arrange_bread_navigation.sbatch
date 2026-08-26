#!/usr/bin/env bash
#SBATCH --job-name=bread-nav-probe
#SBATCH --partition=rtx-devel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --chdir=/work/umass/shlomo_umass/dbenhamougol_umass/robocasa-integration
#SBATCH --output=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.out
#SBATCH --error=/work/umass/shlomo_umass/dbenhamougol_umass/slurm_logs/%x-%j.err
set -euo pipefail
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PY=/work/umass/shlomo_umass/dbenhamougol_umass/envs/robocasa-v3/bin/python
exec "$PY" - <<'PY'
import json
from pathlib import Path
from training.bc_task_vlm.live_sim_eval import SimSession

root=Path('/work/umass/shlomo_umass/dbenhamougol_umass/tick_render100_from150_partition_none_concurrent_v3/arrange_bread_bowl')
t=json.load(open(root/'traj_000107'/'original_trajectory.json'))
s=SimSession(composite_task='ArrangeBreadBowl', sample_trajectory=t, layout=11,
             style=34, seed=42, gl_backend='egl', render_size=128,
             map_dpi=30, map_renderer='raster')
cases = [
 ('agent0_direct', [], 0),
 ('agent1_direct', [], 1),
 ('agent0_after_agent1_give_space_counter', [('give_space','counter',1)], 0),
 ('agent0_after_agent1_give_space_toaster', [('give_space','toaster_oven',1)], 0),
 ('agent0_after_agent0_give_space_counter', [('give_space','counter',0)], 0),
 ('agent1_after_agent0_give_space_counter', [('give_space','counter',0)], 1),
]
try:
 for label, preactions, robot in cases:
  adapter, adapted = s.start_trajectory(t)
  nav_call = adapter._adapt_step(
      {'agent': f'agent_{robot}', 'tool': 'navigate_to_fixture',
       'args': {'fixture_id': 'toaster_oven'}},
      resolved_initial_state=adapted['initial_state'], output_dir=None)
  before=[s.executor.runner._get_robot_position(i).tolist() for i in (0,1)]
  pre=[]
  for _tool, fixture, idx in preactions:
   pre_call = adapter._adapt_step(
       {'agent': f'agent_{idx}', 'tool': 'give_space',
        'args': {'fixture_id': fixture}},
       resolved_initial_state=adapted['initial_state'], output_dir=None)
   result=s.executor.execute(pre_call['tool'], robot_idx=idx, **pre_call['args'])
   pre.append({'fixture':fixture,'robot':idx,'success':result.success,
               'positions':[s.executor.runner._get_robot_position(i).tolist() for i in (0,1)]})
  result=s.executor.execute(nav_call['tool'], robot_idx=robot, **nav_call['args'])
  print(json.dumps({'case':label,'before':before,'pre':pre,'nav_robot':robot,
                    'nav_success':result.success,'details':result.details,
                    'after':[s.executor.runner._get_robot_position(i).tolist() for i in (0,1)]}))
finally:
 s.close()
PY
