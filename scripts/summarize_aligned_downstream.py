import json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'runs/aligned_downstream';T=4.3026527297
def read(tag,seed,budget,task):
    x=json.loads((SOURCE/f'{tag}_seed{seed}_{budget:g}M.{task}.json').read_text())
    return 100*next(iter(x['subject_macro_accuracy'].values()))
def stats(x):
    m=statistics.mean(x);s=statistics.stdev(x);h=T*s/math.sqrt(3)
    return {'values':x,'mean':m,'sd':s,'ci95':[m-h,m+h]}
def main():
    tags=('frozen_G','frozen_S','low-lr_G','low-lr_S','random_G','L');out={}
    for task in ('cmmlu','ceval-valid'):
      out[task]={}
      for budget in (.5,2):
        v={tag:[read(tag,s,budget,task) for s in (42,43,44)] for tag in tags}
        out[task][f'{budget:g}M']={'groups':{k:stats(x) for k,x in v.items()},'paired':{}}
        pairs=(('frozen_G','frozen_S'),('frozen_G','L'),('frozen_G','random_G'),
               ('low-lr_G','low-lr_S'),('low-lr_G','L'),('frozen_G','low-lr_G'))
        for a,b in pairs:out[task][f'{budget:g}M']['paired'][f'{a}-{b}']=stats([x-y for x,y in zip(v[a],v[b])])
    target=SOURCE/'summary.json';target.write_text(json.dumps(out,indent=2));print(json.dumps(out))
if __name__=='__main__':main()
