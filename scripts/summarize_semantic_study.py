"""Aggregate merged semantic-study reports across three training seeds."""
import argparse
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('--folder',type=Path,default=Path('runs/semantic_study'))
p.add_argument('--output',type=Path,default=Path('runs/semantic_study/summary.json'))
a=p.parse_args()
rows=[]
for teacher,student in ((12,1),(8,12)):
    item={'teacher_block':teacher,'student_block':student,'tasks':{}}
    for task,group_key in (('ceval-valid','graft_ceval-valid'),('cmmlu','graft_cmmlu')):
        scores={group:[] for group in ('G','S','R','L')}
        for seed in (42,43,44):
            for group in scores:
                tag=f'L_seed{seed}' if group=='L' else f'{group}_T{teacher}_S{student}_seed{seed}'
                path=a.folder/f'{tag}.{task}.json'
                result=json.loads(path.read_text(encoding='utf-8'))
                scores[group].append(100*result['subject_macro_accuracy'][group_key])
        report={'groups':{g:{'seeds':v,'mean':float(np.mean(v)),'sample_std':float(np.std(v,ddof=1))}
                          for g,v in scores.items()}}
        report['contrasts']={}
        for baseline in ('S','R','L'):
            delta=np.array(scores['G'])-np.array(scores[baseline])
            report['contrasts'][f'G_minus_{baseline}']={'seed_deltas':delta.tolist(),
                'mean':float(delta.mean()),'sample_std':float(delta.std(ddof=1)),
                'mean_95ci_t_df2':(np.array([-1,1])*4.302652729911275*delta.std(ddof=1)/np.sqrt(3)+delta.mean()).tolist()}
        item['tasks'][task]=report
    rows.append(item)
a.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
print(json.dumps(rows,indent=2))
