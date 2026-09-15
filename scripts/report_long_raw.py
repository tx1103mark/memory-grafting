"""Print compact machine-readable measurements from a long-study folder."""
import argparse
import json
import re
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('.'))
p.add_argument('--study',choices=['long_study','long_source_study'],required=True)
a=p.parse_args()
folder=a.root/'runs'/a.study
pattern=re.compile(r'(?P<name>L|[GR]_T\d+_S\d+)_(?P<budget>2|5|10)M\.json')
out={'study':a.study,'complete':(folder/'complete').exists(),'reports':[],'diagnostics':{}}
for path in sorted(folder.glob('*.json')):
    match=pattern.fullmatch(path.name)
    if not match:continue
    result=json.loads(path.read_text())
    out['reports'].append({'name':match['name'],'budget_million':int(match['budget']),
        'accuracy':result['groups']['graft_ceval-valid']['acc,none'],
        'macro':result['subject_macro_accuracy']['graft_ceval-valid']})
for path in sorted(folder.glob('*.diagnostic.json')):
    out['diagnostics'][path.name.removesuffix('.diagnostic.json')]=json.loads(path.read_text())
for path in sorted(folder.glob('*.off.json')):
    result=json.loads(path.read_text())
    out.setdefault('memory_off',{})[path.name.removesuffix('.off.json')]={
        'accuracy':result['groups']['graft_ceval-valid']['acc,none'],
        'macro':result['subject_macro_accuracy']['graft_ceval-valid']}
for run in sorted((a.root/'runs').glob('long_*')):
    path=run/'metrics.jsonl'
    if not path.exists():continue
    final=None
    for line in reversed(path.read_text().splitlines()):
        row=json.loads(line)
        if 'validation_loss' in row:
            final={key:row[key] for key in ('tokens','validation_loss','alpha','gate_bias','gate_mean',
                  'effective_delta_ratio_mean') if key in row}
            break
    if final is not None:out.setdefault('final_training',{})[run.name.removeprefix('long_')]=final
print(json.dumps(out,ensure_ascii=False))
