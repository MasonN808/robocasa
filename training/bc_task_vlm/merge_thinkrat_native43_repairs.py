"""Replace four resampled tasks without overwriting either original evaluation."""
import collections
import json
from pathlib import Path

from training.bc_task_vlm.live_sim_eval import _write_metrics

BASE = Path(__file__).resolve().parent
EVAL = BASE / 'eval_runs/fixed_live_sim'
TASKS = {'arrange_bread_bowl', 'distribute_chicken', 'prepare_cheese_station', 'serve_meal_juice'}


def load(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(rows) == len({r['episode_id'] for r in rows}), path
    return rows


def main():
    suffix = Path('heldout_task_types/aggregate/live_sim_trajectories.jsonl')
    reference = {r['episode_id']:r for r in load(EVAL / 'sft_s120_noreason_train43_held10_lr1e4_wd001_ep1p0_fixed10_promptv9' / suffix)}
    for scale in (60, 90, 120):
        name = f'thinking_rationale_s{scale}_train43_held10_lr1e4_wd001_ep1p0_fixed10_promptv9'
        source = EVAL / name / suffix
        replacement = EVAL / f'thinking_rationale_s{scale}_native43_four_tasks_ep1p0_job862763' / suffix
        original, new = load(source), load(replacement)
        assert collections.Counter(r['task_name'] for r in new) == dict.fromkeys(TASKS, 10)
        retained = [r for r in original if r['task_name'] not in TASKS]
        assert len(retained) == 60
        merged = sorted(retained + new, key=lambda r:(r['task_name'],r['episode_rank']))
        assert len(merged) == 100 and len({r['episode_id'] for r in merged}) == 100
        assert set(reference) == {r['episode_id'] for r in merged}
        for row in merged:
            for key in ('configuration_signature','coordinator_id','scene'):
                assert row[key] == reference[row['episode_id']][key], (scale,row['episode_id'],key)
        target = EVAL / (name + '_native43matched') / suffix
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(''.join(json.dumps(r)+'\n' for r in merged))
        _write_metrics(target,target.parent)
        receipt = dict(original=str(source),replacement=str(replacement),reference=str(EVAL / 'sft_s120_noreason_train43_held10_lr1e4_wd001_ep1p0_fixed10_promptv9' / suffix),retained_episodes=60,replaced_episodes=40,exact_native43_configurations_verified=True)
        (target.parent/'replacement_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(scale, sum(bool(r['fsm_goal_satisfied']) and r['rejected_total']==0 for r in merged), '/100',flush=True)


if __name__ == '__main__':
    main()
