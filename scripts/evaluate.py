"""lm-evaluation-harness runner for B0/L/R/G; no independent scoring implementation."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import torch
from transformers import AutoTokenizer
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from graft.data import load_keys
from graft.harness import GraftHFLM
from graft.model import load_model, restore_trainable
from scripts.prepare_harness_tasks import prepare
from scripts.train import file_hash


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--tasks',default='ceval-valid',help='Comma-separated ceval-valid,ceval-test,cmmlu or leaf task names')
    p.add_argument('--num-fewshot',type=int,default=0)
    p.add_argument('--batch-size',type=int,default=1)
    p.add_argument('--device',default='cuda')
    p.add_argument('--limit',type=int,help='Smoke only: maximum examples PER SUBJECT')
    p.add_argument('--memory-mode',choices=['normal','zero'],default='normal')
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.batch_size != 1:
        raise ValueError('This memory-aware harness adapter currently requires --batch-size 1 because HFLM does not pass causal padding masks to _model_call')
    torch.set_num_threads(8)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False) if a.checkpoint else None
    group=saved['manifest']['group'] if saved else 'B0'
    memory_dir=a.root/((saved['manifest'].get('memory_dir','memory')) if saved else 'memory')
    if saved:
        data_dir=a.root/saved['manifest'].get('data_dir','data/processed')
        for key,path in [('assets_hash',a.root/'assets.lock.json'),('keys_hash',data_dir/'keys.json')]:
            if saved['manifest'][key]!=file_hash(path):
                raise ValueError(f'Checkpoint {key} mismatch')
        if group in ('R','G','S') and saved['manifest']['memory_hash']!=file_hash(memory_dir/'manifest.json'):
            raise ValueError('Checkpoint memory manifest mismatch')
    model=load_model(a.root/'models/student',group,memory_dir/'table.pt',a.device,False,
                     layer=saved['manifest'].get('student_block',3)-1 if saved else 2,
                     engram_buckets=saved['manifest'].get('engram_buckets',0) if saved else 0,
                     shortconv_kernel=saved['manifest'].get('shortconv_kernel',0) if saved else 0,
                     teacher_memory=not saved['manifest'].get('disable_teacher_memory',False) if saved else True,
                     aligned_init=saved['manifest'].get('aligned_init','none') if saved else 'none')
    if saved:
        restore_trainable(model,saved['model'])
    tok=AutoTokenizer.from_pretrained(a.root/'models/student',local_files_only=True)
    key_path=(a.root/saved['manifest'].get('data_dir','data/processed')/'keys.json') if saved else a.root/'data/processed/keys.json'
    keys=load_keys(key_path) if key_path.exists() else {}
    if group in ('R','G','S') and not keys:
        raise ValueError('Memory models require prepared keys.json')
    lm=GraftHFLM(model,tok,keys,a.batch_size,a.memory_mode)
    task_dir=prepare(a.root)
    manager=TaskManager(include_path=str(task_dir),include_defaults=True)
    tasks=[]
    for task in a.tasks.split(','):
        task=task.strip()
        tasks.append(task if task.startswith('mmlu_') else 'graft_'+task.removeprefix('graft_'))
    result=evaluator.simple_evaluate(model=lm,tasks=tasks,task_manager=manager,
            num_fewshot=a.num_fewshot,batch_size=a.batch_size,limit=a.limit,
            log_samples=True,apply_chat_template=False,bootstrap_iters=1000,
            random_seed=0,numpy_random_seed=1234,torch_random_seed=1234,fewshot_random_seed=1234)
    if result is None:
        raise RuntimeError('Harness did not produce a result')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    samples=result.pop('samples',{})
    for task,records in samples.items():
        sample_path=a.output.parent/f'{a.output.stem}.{task}.samples.jsonl'
        sample_path.write_text('\n'.join(json.dumps(r,ensure_ascii=False,default=str) for r in records)+'\n')
    result['experiment']=dict(group=group,checkpoint=str(a.checkpoint),memory_mode=a.memory_mode,
        student_block=saved['manifest'].get('student_block',3) if saved else None,
        memory_dir=str(memory_dir),
        partial=a.limit is not None,lm_eval_version=importlib.metadata.version('lm_eval'),
        task_snapshot=json.loads((task_dir/'manifest.json').read_text()),
        source_assets_sha256=file_hash(a.root/'assets.lock.json'))
    macro={}
    for prefix in ('graft_ceval-valid_','graft_ceval-test_','graft_cmmlu_'):
        values=[v['acc,none'] for k,v in result['results'].items() if k.startswith(prefix) and 'acc,none' in v]
        if values:
            macro[prefix.rstrip('_')]=sum(values)/len(values)
    result['subject_macro_accuracy']=macro
    a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,default=str))
    print(json.dumps({'results':result['results'],'subject_macro_accuracy':macro},ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
