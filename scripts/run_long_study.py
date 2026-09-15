"""10M-token source-layer follow-up, with matched random-memory controls."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time
from scripts.run_layer_study import call

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/long_study'
LOG=ROOT/'logs/long_study'
PAIRS=((6,1),(4,6),(8,12))

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
        wait_free(3)
        call('scripts.build_memory',['--teacher-block',6,'--output-dir','memory/T6'],3,LOG/'build_T6.log')
    jobs=queue.Queue()
    for job in [('L',None,3)]+[(g,t,s) for t,s in PAIRS for g in ('G','R')]:jobs.put(job)
    def worker(gpu):
        errors=[]
        while True:
            try:g,t,s=jobs.get_nowait()
            except queue.Empty:return errors
            name='L' if t is None else f'{g}_T{t}_S{s}'
            run=ROOT/f'runs/long_{name}'
            try:
                wait_free(gpu)
                print(f'START {name} GPU={gpu}',flush=True)
                if not (run/'complete.json').exists():
                    args=['--group',g,'--tokens',10000000,'--seed',42,'--micro-batch',2,'--accumulation',2,
                          '--student-block',s,'--run-name',run.name,'--snapshot-tokens',2000000,5000000]
                    if t is not None:args+=['--memory-dir',f'memory/T{t}']
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
                if g in ('G','R'):
                    output=OUT/f'{name}.diagnostic.json'
                    if not output.exists():
                        call('scripts.diagnose_memory',['--checkpoint',run/'last.pt','--output',output],gpu,LOG/f'{name}.diagnostic.log')
                print(f'DONE {name}',flush=True)
            except Exception as e:
                print(f'FAILED {name}: {e}',flush=True)
                errors.append(dict(name=name,error=str(e)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        errors=sum(pool.map(worker,(0,1,2,3)),[])
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Inspect errors.json; rerun to resume')
    rows=[]
    for budget in (2,5,10):
        for t,s in PAIRS:
            name=f'G_T{t}_S{s}'
            row=dict(budget_million=budget,teacher_block=t,student_block=s)
            for g in ('G','R','L'):
                report=OUT/(f'L_{budget}M.json' if g=='L' else f'{g}_T{t}_S{s}_{budget}M.json')
                result=json.loads(report.read_text())
                row[g]=dict(accuracy=result['groups']['graft_ceval-valid']['acc,none'],
                            macro=result['subject_macro_accuracy']['graft_ceval-valid'])
            bases={'L':f'L_{budget}M','R':f'R_T{t}_S{s}_{budget}M'}
            if budget==10:bases['off']=f'{name}_10M.off'
            for label,base in bases.items():
                output=OUT/f'{name}_{budget}M_vs_{label}.comparison.json'
                call('scripts.compare_results',['--baseline',OUT/f'{base}.json','--grafted',OUT/f'{name}_{budget}M.json',
                     '--output',output],0,LOG/'summary.log')
                row[f'vs_{label}']=json.loads(output.read_text())
            rows.append(row)
    (OUT/'summary.json').write_text(json.dumps(rows,indent=2))
    (OUT/'complete').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':main()
