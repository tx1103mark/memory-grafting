"""Paired subject-stratified bootstrap of harness per-example accuracies."""
import argparse
import collections
import json
from pathlib import Path
import numpy as np


def samples(report):
    rows={}
    result=json.loads(report.read_text())
    tasks=sorted(name for name,value in result.get('results',{}).items() if isinstance(value,dict) and 'acc,none' in value)
    paths=[]
    for task in tasks:
        path=report.parent/f'{report.stem}.{task}.samples.jsonl'
        if path.exists():paths.append((task,path))
    for task,path in paths:
        for line in path.read_text().splitlines():
            r=json.loads(line)
            key=(task,str(r.get('doc_hash',r['doc_id'])))
            if key in rows:
                raise ValueError('Duplicate sample ID')
            rows[key]=float(r['acc'])
    if not rows:
        raise ValueError('No harness samples next to report')
    return rows


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--grafted',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--replicates',type=int,default=10000)
    a=p.parse_args()
    baseline,grafted=samples(a.baseline),samples(a.grafted)
    if baseline.keys()!=grafted.keys():
        raise ValueError('Paired comparison requires exactly the same task/example IDs')
    by_subject=collections.defaultdict(list)
    corrected=regressed=was_wrong=was_right=0
    for key in sorted(baseline):
        b=baseline[key]
        g=grafted[key]
        by_subject[key[0]].append(g-b)
        corrected+=b==0 and g==1
        regressed+=b==1 and g==0
        was_wrong+=b==0
        was_right+=b==1
    rng=np.random.default_rng(42)
    macro=np.zeros(a.replicates)
    for values in by_subject.values():
        values=np.array(values)
        for start in range(0,a.replicates,100):
            size=min(100,a.replicates-start)
            macro[start:start+size]+=values[rng.integers(0,len(values),(size,len(values)))].mean(1)/len(by_subject)
    result=dict(subject_macro_delta_pp=100*np.mean([np.mean(v) for v in by_subject.values()]),
                subject_macro_delta_95ci_pp=(100*np.quantile(macro,[.025,.975])).tolist(),
                corrected=corrected,regressed=regressed,correction_rate=corrected/was_wrong if was_wrong else None,
                regression_rate=regressed/was_right if was_right else None,examples=len(baseline),
                bootstrap_replicates=a.replicates,seed=42,
                caveat='Single-run paired example bootstrap does not measure training-seed variance.')
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
