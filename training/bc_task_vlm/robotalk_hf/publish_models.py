"""Prepare and publish the ten verified one-epoch RoboTalk scaling adapters."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from huggingface_hub import HfApi

BASE = Path(__file__).resolve().parents[1]
OUTPUT = BASE / 'reports/robotalk_model_release'
SCALES = (30, 60, 90, 120, 150)
STEPS = (648, 1298, 1945, 2593, 3240)
CONFIG_KEYS = (
    'model_name_or_path','processor_name_or_path','num_epochs','learning_rate',
    'weight_decay','lora_r','lora_alpha','lora_dropout','lora_target_modules',
    'lr_scheduler_type','warmup_ratio','per_device_batch_size','grad_accum',
    'max_length','max_tokens','max_images_per_sample','image_resolution',
    'seed','train_reasoning','sft_format','partial_history',
    'partial_observation_mode','partial_step_index_mode','train_get_image',
    'supervise_last_assistant_turn_only','bf16','gradient_checkpointing',
)


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def card(repo, model, scale, thinking, results):
    target = 'a rationale trace followed by a tool call' if thinking else 'the next tool call'
    table = '\n'.join(
        f"| {label} | {r['error_free_successes']}/{r['episodes']} | {100*r['error_free_success_rate']:.2f}% | {100*r['ci_low']:.2f}–{100*r['ci_high']:.2f}% |"
        for label,r in results.items()
    )
    return f'''---
base_model: {model}
library_name: peft
pipeline_tag: image-text-to-text
datasets:
- DorianAtSchool/RoboTalk
language:
- en
tags:
- lora
- robotics
- multi-agent
- vision-language
- base_model:adapter:{model}
---

# {repo.split('/')[-1]}

One-epoch LoRA adapter trained on **{scale} RoboTalk trajectories per training task**
across 43 tasks ({scale*43:,} training trajectories). Ten additional tasks are held
out. The prediction target is **{target}**. One shared policy controls two agents
with separate partially observable contexts and a coordinator–follower protocol.

## Load the adapter

This repository is an adapter, not a standalone model. It requires the matching
Qwen model and processor. Use Transformers with Qwen3-VL support and PEFT.

```python
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

base_id = "{model}"
processor = AutoProcessor.from_pretrained(base_id)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, device_map="auto"
)
model = PeftModel.from_pretrained(model, "{repo}")
model.eval()
```

Use the model's chat template for tool calling. Reproducing the scores requires
the RoboTalk task goal, initial state, global tool definitions, per-agent visual
observations and private histories, and full communication protocol—not a generic
chat prompt. Histories do not include step indices. See the
[dataset](https://huggingface.co/datasets/DorianAtSchool/RoboTalk) for trajectories
and prompt metadata.

## Training

- Base model: `{model}`.
- One epoch, learning rate 1e-4, weight decay 0.01, cosine schedule, 3% warmup.
- LoRA rank 16, alpha 32, dropout 0.05.
- Effective batch size 64 across four GPUs; sequence limit 8,192 tokens including visual tokens.
- Seed 42; the released adapter contains language-model LoRA weights, not a separately fine-tuned vision backbone.
- Larger scale datasets contain the smaller-scale selections.

Full sanitized settings are in `training_config.json`; task membership is in
`task_split.json`. Optimizer states and training-machine paths are not included.

## Closed-loop evaluation

Full communication, ten episodes per task, native 43/10 fixed-cohort sampling.
Error-free FSM success means reaching the symbolic goal with no rejected calls.
The intervals are 95% Wilson intervals over episodes, not over newly sampled tasks.

| Split | Error-free successes / episodes | Success rate | 95% interval |
|---|---:|---:|---:|
{table}

Exact counts and evaluation settings are in `evaluation_results.json`.
These results use the actual one-epoch checkpoint and corrected cohort selections.
Instruct evaluation uses temperature 0 and a 256-token output budget; Thinking
evaluation uses temperature 0.6, top-p 0.95, top-k 20 and a 2,048-token output budget.
These inference settings differ between model variants.

## Scope and limitations

Intended for research on high-level coordination in a household simulator, not
direct hardware control or safety-critical deployment. FSM success is a symbolic
task metric and need not imply satisfaction of every native physical criterion.
The underlying Qwen model is distributed under Apache-2.0; consult its model
repository for its terms. This card does not assign a separate adapter license.
'''


def prepare():
    artifact = read(BASE/'eval_runs/fixed_live_sim_43_10_results_artifact/artifact.json')
    rows = artifact['snapshot']['datasets']['scale_results']
    split = read(BASE/'scale_experiments/tick150_reasoning_43train_10held_seed20260821/43_train_10_heldout.json')
    membership = {k:split[k] for k in ('train_tasks','held_out_tasks')}
    assert len(membership['train_tasks'])==43 and len(membership['held_out_tasks'])==10
    entries = []
    for thinking in (False,True):
        for scale, step in zip(SCALES,STEPS):
            variant = 'Thinking-Rationale' if thinking else 'Instruct'
            repo = f'DorianAtSchool/RoboTalk-Qwen3-VL-8B-{variant}-{scale}traj'
            name = f"qwen3vl8b{'-thinking' if thinking else ''}-tick150-s{scale}-{'rationale' if thinking else 'noreason'}-train43-held10-ep1-halfckpt-lr1e4-wd001-noindex-promptv9"
            run = BASE/'runs'/name
            ck = run/f'checkpoint-{step}'
            cfg, state, adapter = read(run/'run_config.json'),read(ck/'trainer_state.json'),read(ck/'adapter_config.json')
            assert state['epoch']==1.0 and state['global_step']==step
            assert cfg['train_reasoning']==thinking
            assert set(cfg['train_tasks'])==set(membership['train_tasks'])
            assert adapter['base_model_name_or_path']==cfg['model_name_or_path']
            title = f"{'Thinking + rationale SFT' if thinking else 'SFT'} {scale}/task"
            selected = [r for r in rows if r['model']==title and r['checkpoint']=='epoch 1.0']
            assert len(selected)==2
            results={}
            for r in selected:
                label='held_out_tasks' if r['split'].startswith('Held') else 'in_training_tasks'
                results[label]={k:r[k] for k in ('error_free_successes','episodes','error_free_success_rate','ci_low','ci_high')}
            expected=([71,72,66,77,73] if thinking else [45,64,55,48,65])[SCALES.index(scale)]
            assert results['held_out_tasks']['error_free_successes']==expected
            out=OUTPUT/'upload'/repo.split('/')[-1]
            out.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ck/'adapter_model.safetensors',out/'adapter_model.safetensors')
            write(out/'adapter_config.json',adapter)
            settings={k:cfg.get(k) for k in CONFIG_KEYS}
            settings.update(trajectories_per_task=scale,training_trajectories=43*scale,num_training_tasks=43,num_held_out_tasks=10,effective_batch_size=64,num_gpus=4,checkpoint_global_step=step,checkpoint_epoch=state['epoch'],prompt_contract_version='state_grounded_global_tools_no_index_v9_unambiguous_garnish_cake_goal')
            write(out/'training_config.json',settings)
            write(out/'task_split.json',membership)
            evaluation=dict(metric='error_free_fsm_success',episodes_per_task=10,communication_mode='full',evaluation_seed=20260817,scene_sampling_seed=20260819,cohort='native_43_train_10_heldout',max_new_tokens=2048 if thinking else 256,temperature=0.6 if thinking else 0.0,top_p=0.95 if thinking else None,top_k=20 if thinking else None,results=results)
            write(out/'evaluation_results.json',evaluation)
            (out/'README.md').write_text(card(repo,cfg['model_name_or_path'],scale,thinking,results))
            for path in out.iterdir():
                if path.suffix in ('.json','.md'):
                    content=path.read_text()
                    assert '/work/' not in content and 'dbenhamou' not in content and 'umass' not in content
            hashes={p.name:sha(p) for p in out.iterdir() if p.is_file()}
            entries.append(dict(repo=repo,source_checkpoint=str(ck),upload_dir=str(out),hashes=hashes))
            print('Prepared',repo,flush=True)
    write(OUTPUT/'manifest.json',entries)


def upload():
    api=HfApi()
    receipt_path=OUTPUT/'upload_receipt.json'
    receipts=read(receipt_path) if receipt_path.exists() else {}
    for entry in read(OUTPUT/'manifest.json'):
        repo=entry['repo']; folder=Path(entry['upload_dir'])
        for name,digest in entry['hashes'].items():
            assert sha(folder/name)==digest
        if repo not in receipts:
            api.create_repo(repo,repo_type='model',private=False,exist_ok=False)
            receipts[repo]={'created':True}
            write(receipt_path,receipts)
        info=api.model_info(repo,files_metadata=True)
        assert not info.private
        if receipts[repo].get('verified'):
            assert info.sha==receipts[repo]['commit'], 'Repository changed externally'
        else:
            commit=api.upload_folder(repo_id=repo,repo_type='model',folder_path=folder,commit_message='Release verified one-epoch RoboTalk LoRA adapter')
            receipts[repo]['commit']=commit.oid
            write(receipt_path,receipts)
            info=api.model_info(repo,files_metadata=True)
        remote={s.rfilename:s for s in info.siblings}
        assert set(remote)==set(entry['hashes'])|{'.gitattributes'}
        for name,digest in entry['hashes'].items():
            item=remote[name]
            if item.lfs:
                assert item.lfs.sha256==digest
            else:
                data=(folder/name).read_bytes()
                assert item.blob_id==hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
        receipts[repo]['verified']=True
        write(receipt_path,receipts)
        print('VERIFIED PUBLIC https://huggingface.co/'+repo,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('prepare','upload'))
    args=parser.parse_args()
    (prepare if args.stage=='prepare' else upload)()
