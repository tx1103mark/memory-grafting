"""Summarize preregistered three-seed confirmation and paired ablations."""
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
CONF=ROOT/'remote_results/confirmation_study'
MECH=ROOT/'remote_results/mechanism_study'
T95_DF2=4.3026527297

def score(path):
    obj=json.loads(path.read_text(encoding='utf-8'))
    return 100*next(iter(obj['subject_macro_accuracy'].values()))

def metric(tag,seed,task):
    if tag in ('full_G','full_S','full_L') and seed==42:
        stem={'full_G':'G_T12_S1','full_S':'S_T12_S1','full_L':'L'}[tag]
        path=MECH/f'{stem}_5M.{task}.json'
    else:
        path=CONF/f'{tag}_seed{seed}.{task}.json'
    return score(path)

def stats(values):
    mean=statistics.mean(values); sd=statistics.stdev(values)
    half=T95_DF2*sd/math.sqrt(len(values))
    return {'values':values,'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}

def paired(a,b): return stats([x-y for x,y in zip(a,b)])

def main():
    seeds=(42,43,44)
    tags=('full_G','full_S','full_L','nofallback_G','noshortconv_G','fallback_only_G')
    out={'protocol':{'seeds':list(seeds),'primary':'cmmlu','secondary':'ceval-valid'},'tasks':{}}
    for task in ('cmmlu','ceval-valid'):
        values={tag:[metric(tag,s,task) for s in seeds] for tag in tags}
        out['tasks'][task]={'groups':{k:stats(v) for k,v in values.items()},'paired':{
            'full_G-full_S':paired(values['full_G'],values['full_S']),
            'full_G-full_L':paired(values['full_G'],values['full_L']),
            'full_G-nofallback_G':paired(values['full_G'],values['nofallback_G']),
            'full_G-noshortconv_G':paired(values['full_G'],values['noshortconv_G']),
            'full_G-fallback_only_G':paired(values['full_G'],values['fallback_only_G']),
        }}
    target=CONF/'summary.json'; target.write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))

if __name__=='__main__': main()
