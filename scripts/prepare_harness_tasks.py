"""Snapshot official harness task YAML; replace dataset transport with local files."""
import argparse
import collections
import hashlib
import importlib.metadata
import json
from pathlib import Path
import lm_eval
import yaml


def load_config(path):
    config=yaml.safe_load(path.read_text())
    include=config.pop('include',None)
    if include is None:
        return config
    if not isinstance(include,str):
        raise ValueError('Unexpected upstream include schema; inspect before changing task semantics')
    return load_config(path.parent/include) | config


def prepare(root):
    root=Path(root).resolve()
    folder=root/'data/harness_tasks'
    folder.mkdir(parents=True,exist_ok=True)
    datafolder=root/'data/harness_data'
    datafolder.mkdir(exist_ok=True)
    rows=json.loads((root/'data/processed/benchmarks.json').read_text())
    by_subject=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in rows:
        capital=row['dataset']=='cmmlu'
        value={'Question' if capital else 'question':row['question'],
               'Answer' if capital else 'answer':row['answer'], **dict(zip('ABCD',row['choices']))}
        by_subject[(row['dataset'],row['subject'])][row['split']].append(value)
    package=Path(lm_eval.__file__).parent/'tasks'
    source_hashes={}
    for dataset,group in [('ceval','ceval-valid'),('cmmlu','cmmlu')]:
        source=package/dataset
        group_file=next(p for p in source.glob('*.yaml') if yaml.safe_load(p.read_text()).get('group')==group)
        group_cfg=yaml.safe_load(group_file.read_text())
        leaves=[]
        test_leaves=[]
        for subjectfile in sorted(source.glob(f'{group}_*.yaml')):
            config=load_config(subjectfile)
            subject=config['dataset_name']
            splits=by_subject[(dataset,subject)]
            if not splits:
                raise ValueError(f'Missing benchmark subject: {dataset}/{subject}')
            data_files={}
            for split,records in splits.items():
                path=datafolder/f'{dataset}_{subject}_{split}.jsonl'
                path.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n')
                data_files[split]=str(path)
            name='graft_'+config['task']
            config.update(task=name,dataset_path='json',dataset_name=None,dataset_kwargs={'data_files':data_files})
            if dataset=='ceval':
                config.pop('test_split',None)
            (folder/f'{name}.yaml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False))
            leaves.append(name)
            if dataset=='ceval':
                if any(r['answer'] not in list('ABCD') for r in splits['test']):
                    raise ValueError('C-Eval test labels unavailable in this snapshot')
                test_name=name.replace('ceval-valid_','ceval-test_')
                test_config=config | {'task':test_name,'validation_split':None,'test_split':'test'}
                (folder/f'{test_name}.yaml').write_text(yaml.safe_dump(test_config,allow_unicode=True,sort_keys=False))
                test_leaves.append(test_name)
        if not leaves:
            raise RuntimeError('Harness task layout changed; inspect installed package')
        group_cfg.update(group='graft_'+group,task=leaves)
        (folder/f'_{group}.yaml').write_text(yaml.safe_dump(group_cfg,allow_unicode=True,sort_keys=False))
        if test_leaves:
            test_group=group_cfg | {'group':'graft_ceval-test','task':test_leaves}
            (folder/'_ceval-test.yaml').write_text(yaml.safe_dump(test_group,allow_unicode=True,sort_keys=False))
        for p in source.iterdir():
            if p.is_file():
                source_hashes[f'{dataset}/{p.name}']=hashlib.sha256(p.read_bytes()).hexdigest()
    manifest=dict(lm_eval_version=importlib.metadata.version('lm_eval'),upstream_task_files=source_hashes,
                  source_benchmarks_sha256=hashlib.sha256((root/'data/processed/benchmarks.json').read_bytes()).hexdigest(),
                  modifications='Local JSON transport and graft_ names; upstream prompts/metrics/description retained. Explicit ceval-test variant changes evaluation split from val to test.',
                  task_groups=['graft_ceval-valid','graft_ceval-test','graft_cmmlu'])
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
    return folder


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    print(prepare(p.parse_args().root))
