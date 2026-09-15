"""Complementary 10M source-layer ablations for the second H200 host."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.run_layer_study import call

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/long_source_study'
LOG=ROOT/'logs/long_source_study'
PAIRS=((4,1),(8,1),(12,1),(6,6))

def wait_free(gpu):
    while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip())<20000:
        time.sleep(30)

def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (ROOT/'memory/T6/manifest.json').exists():
        wait_free(7)
        call('scripts.build_memory',['--teacher-block',6,'--output-dir','memory/T6'],7,LOG/'build_T6.log')
    jobs=[(g,t,s) for t,s in PAIRS for g in ('G','R')]
    def worker(item):
        gpu,(g,t,s)=item
        name=f'{g}_T{t}_S{s}';run=ROOT/f'runs/long_{name}'
        try:
            wait_free(gpu)
            print(f'START {name} GPU={gpu}',flush=True)
            if not (run/'complete.json').exists():
                memory_dir='memory' if t==12 else f'memory/T{t}'
                args=['--group',g,'--tokens',10000000,'--seed',42,'--micro-batch',2,'--accumulation',2,
                      '--student-block',s,'--memory-dir',memory_dir,'--run-name',run.name,
                      '--snapshot-tokens',2000000,5000000]
                if (run/'last.pt').exists():args+=['--resume',run/'last.pt']
                call('scripts.train',args,gpu,LOG/f'{name}.train.log')
            for budget,checkpoint in ((2,'tokens-2000000.pt'),(5,'tokens-5000000.pt'),(10,'last.pt')):
                output=OUT/f'{name}_{budget}M.json'
                if not output.exists():
                    call('scripts.evaluate',['--checkpoint',run/checkpoint,'--tasks','ceval-valid','--output',output],
                         gpu,LOG/f'{name}_{budget}M.eval.log')
            if g=='G':
                output=OUT/f'{name}_10M.off.json'
                if not output.exists():
                    call('scripts.evaluate',['--checkpoint',run/'last.pt','--tasks','ceval-valid',
                         '--memory-mode','zero','--output',output],gpu,LOG/f'{name}.off.log')
            output=OUT/f'{name}.diagnostic.json'
            if not output.exists():
                call('scripts.diagnose_memory',['--checkpoint',run/'last.pt','--output',output],gpu,LOG/f'{name}.diagnostic.log')
            print(f'DONE {name}',flush=True);return []
        except Exception as e:
            print(f'FAILED {name}: {e}',flush=True);return [dict(name=name,error=str(e))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        errors=sum(pool.map(worker,enumerate(jobs)),[])
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Inspect errors.json; rerun to resume')
    rows=[]
    for budget in (2,5,10):
        for t,s in PAIRS:
            name=f'G_T{t}_S{s}'
            row=dict(budget_million=budget,teacher_block=t,student_block=s)
            for g in ('G','R'):
                result=json.loads((OUT/f'{g}_T{t}_S{s}_{budget}M.json').read_text())
                row[g]=dict(accuracy=result['groups']['graft_ceval-valid']['acc,none'],
                            macro=result['subject_macro_accuracy']['graft_ceval-valid'])
            output=OUT/f'{name}_{budget}M_vs_R.comparison.json'
            call('scripts.compare_results',['--baseline',OUT/f'R_T{t}_S{s}_{budget}M.json',
                 '--grafted',OUT/f'{name}_{budget}M.json','--output',output],0,LOG/'summary.log')
            row['vs_R']=json.loads(output.read_text());rows.append(row)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2))
    (OUT/'complete').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':main()
