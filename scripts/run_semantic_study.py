"""Three-seed G/shuffled/random/LoRA study split deterministically over two hosts."""
import argparse
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import time
from scripts.run_layer_study import call

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/semantic_study'
LOG=ROOT/'logs/semantic_study'

def wait_free(gpu):
    while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip())<20000:
        time.sleep(30)

def existing(group,teacher,student):
    if group=='L':return ROOT/'runs/long_L/last.pt'
    return ROOT/f'runs/long_{group}_T{teacher}_S{student}/last.pt'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--host',choices=['203','238'],required=True)
    a=p.parse_args()
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    lock=(OUT/f'runner-{a.host}.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if a.host=='203':
        jobs=[(g,t,s,43) for g,t,s in [('L',None,3),('G',12,1),('S',12,1),('R',12,1),
              ('G',8,12),('S',8,12),('R',8,12)]]+[(g,t,s,42) for g,t,s in [('S',8,12),('G',8,12),('R',8,12),('L',None,3)]]
        gpus=range(4)
    else:
        jobs=[(g,t,s,44) for g,t,s in [('L',None,3),('G',12,1),('S',12,1),('R',12,1),
              ('G',8,12),('S',8,12),('R',8,12)]]+[(g,t,s,42) for g,t,s in [('S',12,1),('G',12,1),('R',12,1)]]
        gpus=range(8)
    pending=queue.Queue()
    for job in jobs:pending.put(job)
    def worker(gpu):
        errors=[]
        while True:
            try:group,teacher,student,seed=pending.get_nowait()
            except queue.Empty:return errors
            tag=f'{group}_seed{seed}' if group=='L' else f'{group}_T{teacher}_S{student}_seed{seed}'
            run=ROOT/f'runs/semantic_{tag}'
            try:
                checkpoint=existing(group,teacher,student) if seed==42 and group!='S' else run/'last.pt'
                if not checkpoint.exists():
                    wait_free(gpu)
                    memory='memory' if teacher==12 else f'memory/T{teacher}'
                    args=['--group',group,'--tokens',10000000,'--seed',seed,'--micro-batch',2,'--accumulation',2,
                          '--student-block',student,'--run-name',run.name,'--snapshot-tokens',2000000,5000000]
                    if group!='L':args+=['--memory-dir',memory]
                    if (run/'last.pt').exists():args+=['--resume',run/'last.pt']
                    call('scripts.train',args,gpu,LOG/f'{tag}.train.log')
                for task in ('ceval-valid','cmmlu'):
                    output=OUT/f'{tag}.{task}.json'
                    if not output.exists():
                        wait_free(gpu)
                        call('scripts.evaluate',['--checkpoint',checkpoint,'--tasks',task,'--output',output],gpu,
                             LOG/f'{tag}.{task}.eval.log')
                if group in ('G','S','R'):
                    output=OUT/f'{tag}.diagnostic.json'
                    if not output.exists():
                        wait_free(gpu)
                        call('scripts.diagnose_memory',['--checkpoint',checkpoint,'--output',output],gpu,
                             LOG/f'{tag}.diagnostic.log')
                print('DONE',tag,flush=True)
            except Exception as e:
                errors.append({'tag':tag,'error':str(e)});print('FAILED',tag,e,flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(tuple(gpus))) as pool:
        errors=sum(pool.map(worker,gpus),[])
    (OUT/f'errors-{a.host}.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Rerun after inspecting errors file')
    (OUT/f'complete-{a.host}').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':main()
