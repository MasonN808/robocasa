"""Read-only source audit; writes only a separate audit JSON report."""
import collections
import json
import struct
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
OUT = BASE / 'reports/paper_checkpoint_provenance_audit.json'


def read(path):
    return json.loads(Path(path).read_text())


def records(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main():
    artifact = read(BASE / 'eval_runs/fixed_live_sim_43_10_results_artifact/artifact.json')
    reference = None
    results = []
    for row in artifact['snapshot']['datasets']['native_results']:
        metrics_path = ROOT / row['source_path']
        split_dir = metrics_path.parent.parent
        run_dir = split_dir.parent
        source_run = run_dir
        receipt = run_dir / 'regroup_receipt.json'
        if receipt.exists():
            source_run = Path(read(receipt)['source_run'])
        parallel = read(source_run / split_dir.name / 'parallel_run.json')
        args = parallel['identity']['live_args_template']
        ck = Path(args[args.index('--adapter-path') + 1])
        state = read(ck / 'trainer_state.json')
        cfg = read(ck.parent / 'run_config.json')
        adapter = read(ck / 'adapter_config.json')
        with (ck / 'adapter_model.safetensors').open('rb') as f:
            header_length = struct.unpack('<Q', f.read(8))[0]
            header = json.loads(f.read(header_length))
        tensors = {k:v for k,v in header.items() if k != '__metadata__'}
        size = (ck / 'adapter_model.safetensors').stat().st_size
        header_valid = max(v['data_offsets'][1] for v in tensors.values()) == size - 8 - header_length
        rows = records(metrics_path.with_name('live_sim_trajectories.jsonl'))
        latest = {r['episode_id']:r for r in rows}
        metrics = read(metrics_path)
        successes = sum(bool(r['fsm_goal_satisfied']) and r['rejected_total'] == 0 for r in latest.values())
        settings = {k:cfg.get(k) for k in (
            'model_name_or_path','num_epochs','learning_rate','weight_decay','lora_r',
            'lora_alpha','lora_dropout','max_length','per_device_batch_size','grad_accum',
            'seed','train_reasoning','partial_step_index_mode','sft_format',
            'preprocessed_data_dir','init_adapter_path','processor_name_or_path')}
        if reference is None:
            reference = settings
        per_task = collections.defaultdict(lambda: {'episodes':0,'error_free':0,'fsm_success':0,'rejected':0})
        for r in latest.values():
            t = per_task[r['task_name']]
            t['episodes'] += 1
            t['error_free'] += int(bool(r['fsm_goal_satisfied']) and r['rejected_total'] == 0)
            t['fsm_success'] += int(bool(r['fsm_goal_satisfied']))
            t['rejected'] += int(r['rejected_total'] > 0)
        identities = [{k:r.get(k) for k in ('task_name','episode_rank','configuration_signature','coordinator_id','scene','expert_steps')} for r in latest.values()]
        source_rows = []
        for source_split in ('train_task_types','heldout_task_types'):
            source_path = source_run / source_split / 'aggregate/live_sim_trajectories.jsonl'
            if source_path.exists():
                source_rows.extend(records(source_path))
        source_index = {r['episode_id']:r for r in source_rows}
        result = dict(model=row['model'],label=row['checkpoint'],split=row['split'],
            run=str(run_dir),source_run=str(source_run),checkpoint=str(ck),
            actual_epoch=state['epoch'],global_step=state['global_step'],
            epoch_label_matches=abs(float(row['checkpoint'].split()[-1])-state['epoch']) < .01,
            episodes=len(latest),raw_rows=len(rows),successes=successes,
            aggregate_matches=successes==metrics['num_fsm_error_free_successes'],
            artifact_matches=abs(row['error_free_success_rate']-successes/len(latest)) < 1e-10,
            source_rows_match=all(
                {field:value for field,value in source_index.get(k,{}).items() if field != 'cohort_split'}
                == {field:value for field,value in v.items() if field != 'cohort_split'}
                for k,v in latest.items()),
            complete_task_counts=all(t['episodes']==10 for t in per_task.values()),
            num_tasks=len(per_task),tensor_count=len(tensors),header_valid=header_valid,
            adapter_base_matches=adapter['base_model_name_or_path']==cfg['model_name_or_path'],
            train_tasks=cfg.get('train_tasks'),settings=settings,
            eval_args=args,invocations=parallel.get('invocations'),
            per_task=per_task,episode_identities=identities)
        results.append(result)
        print(row['model'],row['checkpoint'],split_dir.name,'ACTUAL',round(state['epoch'],4),
              'success',successes,'/',len(latest),'epoch_ok',result['epoch_label_matches'],
              'source_ok',result['source_rows_match'],flush=True)
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(json.dumps(results,indent=2)+'\n')
    print(OUT)


if __name__ == '__main__':
    main()
