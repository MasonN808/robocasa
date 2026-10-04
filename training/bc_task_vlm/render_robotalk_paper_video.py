"""Replay a RoboTalk tick trajectory using the August communication-video template."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from training.bc_task_vlm.render_communication_ablation_examples import (
    ROOT, SimSession, imageio, np, render_card,
)
from data_generation.task_level.scene_sampling import load_compatibility_cache, sample_compatible_scene
from data_generation.task_level.tasks.specs import load_verified_task_specs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    trajectory = json.loads(args.trajectory.read_text())
    spec = next(s for s in load_verified_task_specs() if s.composite_task == trajectory['composite_task'])
    trajectory['task'] = spec.task_goal
    cache = load_compatibility_cache(ROOT / 'training/bc_task_vlm/reports/new_access_state_generation_preflight/scene_compatibility_cache_v2.json')
    scene = sample_compatible_scene(cache, task=trajectory['composite_task'],
        physical_configuration_signature=trajectory['physical_configuration_signature'],
        sampling_seed=20260819, sample_index=int(trajectory['trajectory_id'].split('_')[-1]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    session = SimSession(composite_task=trajectory['composite_task'], sample_trajectory=trajectory,
        layout=scene['layout'], style=scene['style'], seed=scene['seed'],
        gl_backend='egl', render_size=700, map_dpi=60, map_renderer='raster')
    receipt = {'source': str(args.trajectory), 'scene': scene, 'steps': [],
               'format': 'initial frame and post-timestep frames; fixed expert replay, not policy rollout'}
    selection = {'title': 'RoboTalk | PlateStoreDinner | Trajectory 007',
                 'task_goal': spec.task_goal, 'fit_full_messages': True}
    record = {'task_name': 'plate_store_dinner', 'communication_mode': 'full'}
    panels = {'agent_0': None, 'agent_1': None}
    fps = 4
    try:
        adapter, adapted = session.start_trajectory(trajectory)
        with imageio.get_writer(args.output_dir / 'plate_store_dinner_007.mp4', fps=fps,
                               codec='libx264', quality=8, macro_block_size=None) as writer:
            def capture(index, representative, seconds):
                card = render_card(session.executor.render(), selection, record, representative,
                                   index, len(trajectory['tick_rows']), panels)
                card.save(args.output_dir / f'frame_{index:03}.jpg')
                for _ in range(fps * seconds):
                    writer.append_data(np.asarray(card))
            capture(0, None, 2)
            for ordinal, tick in enumerate(trajectory['tick_rows'], 1):
                current = []
                for agent in panels:
                    entry = tick.get(agent)
                    if not entry or entry.get('state') == 'blocked':
                        previous = panels[agent]
                        if previous is None or previous['proposal']['tool'] != 'wait_for_signal':
                            raise RuntimeError(f'Missing unblocked agent at timestep {tick["tick"]}: {agent}')
                        continue
                    proposal = {'agent': agent, 'tool': entry['tool'], 'args': entry.get('args', {})}
                    step = {'proposal': proposal, 'sim_time': tick['tick'], 'legal': True}
                    if entry['tool'] not in {'communicate', 'wait_for_signal', 'get_image'}:
                        call = adapter._adapt_step(proposal, resolved_initial_state=adapted['initial_state'], output_dir=None)
                        result = session.executor.execute(call['tool'], robot_idx=call['robot_idx'], **call['args'])
                        step['sim_success'] = result.success
                        receipt['steps'].append({'tick': tick['tick'], **proposal, 'result': asdict(result)})
                        if not result.success:
                            raise RuntimeError(f'Replay failed: {result}')
                    panels[agent] = step
                    current.append(step)
                capture(ordinal, current[0], 3 if any(s['proposal']['tool'] == 'communicate' for s in current) else 2)
                print(f'Rendered timestep {tick["tick"]}', flush=True)
        receipt['native_success'] = bool(session.executor.env._check_success())
        receipt['completed'] = True
    finally:
        (args.output_dir / 'replay_receipt.json').write_text(json.dumps(receipt, indent=2))
        session.close()


if __name__ == '__main__':
    main()
