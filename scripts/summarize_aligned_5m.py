import argparse,json,math,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
T=4.3026527297

def read(source,tag,seed,task):
    row=json.loads((source/f'{tag}_seed{seed}_5M.{task}.json').read_text(encoding='utf-8'))
    return 100*next(iter(row['subject_macro_accuracy'].values()))

def stats(values):
    mean=statistics.mean(values);sd=statistics.stdev(values);half=T*sd/math.sqrt(3)
    return {'values':values,'mean':mean,'sd':sd,'ci95':[mean-half,mean+half]}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source',type=Path,default=ROOT/'remote_results/aligned_5m')
    args=parser.parse_args();source=args.source
    out={}
    for task in ('cmmlu','ceval-valid'):
        groups={tag:[read(source,tag,seed,task) for seed in (42,43,44)]
                for tag in ('low-lr_G','low-lr_S','L')}
        paired={}
        for left,right in (('low-lr_G','low-lr_S'),('low-lr_G','L'),('low-lr_S','L')):
            paired[f'{left}-{right}']=stats([a-b for a,b in zip(groups[left],groups[right])])
        out[task]={'groups':{tag:stats(values) for tag,values in groups.items()},'paired':paired}
    target=source/'summary.json';target.write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))

if __name__=='__main__':main()
