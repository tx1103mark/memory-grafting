"""Build 500k memories then test full fallback+ShortConv grafting against shuffled rows."""
import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time
from scripts.run_layer_study import call

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/mechanism_study';LOG=ROOT/'logs/mechanism_study'

def wait_file(path):
    while not path.exists():time.sleep(30)

def wait_free(gpu,minimum=20000):
    while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip())<minimum:time.sleep(30)

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    wait_file(ROOT/'data/processed_500k/manifest.json')
    def build(item):
        gpu,teacher=item;folder=ROOT/f'memory_500k/T{teacher}'
        if not (folder/'manifest.json').exists():
            wait_free(gpu,30000)
            call('scripts.build_memory',['--teacher-block',teacher,'--output-dir',folder,
                 '--data-dir','data/processed_500k','--no-random','--batch-size',256],gpu,LOG/f'build_T{teacher}.log')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:list(pool.map(build,((0,8),(1,12))))
    jobs=[('L',None,3),('G',8,12),('S',8,12),('G',12,1),('S',12,1)]
    def train(item):
        gpu,(group,teacher,student)=item
        tag='L' if group=='L' else f'{group}_T{teacher}_S{student}'
        run=ROOT/f'runs/mechanism_{tag}'
        try:
            wait_free(gpu,30000)
            if not (run/'complete.json').exists():
                args=['--group',group,'--tokens',5000000,'--seed',42,'--micro-batch',2,'--accumulation',2,
                      '--student-block',student,'--data-dir','data/processed_500k','--run-name',run.name,
                      '--snapshot-tokens',500000,1000000,2000000]
                if group!='L':args+=['--memory-dir',f'memory_500k/T{teacher}','--engram-buckets',65536,'--shortconv-kernel',4]
                if (run/'last.pt').exists():args+=['--resume',run/'last.pt']
                call('scripts.train',args,gpu,LOG/f'{tag}.train.log')
            checkpoints=((.5,'tokens-500000.pt'),(1,'tokens-1000000.pt'),(2,'tokens-2000000.pt'),(5,'last.pt'))
            for budget,checkpoint in checkpoints:
                output=OUT/f'{tag}_{budget:g}M.ceval-valid.json'
                if not output.exists():call('scripts.evaluate',['--checkpoint',run/checkpoint,'--tasks','ceval-valid','--output',output],gpu,LOG/f'{tag}_{budget:g}M.ceval.log')
            output=OUT/f'{tag}_5M.cmmlu.json'
            if not output.exists():call('scripts.evaluate',['--checkpoint',run/'last.pt','--tasks','cmmlu','--output',output],gpu,LOG/f'{tag}_5M.cmmlu.log')
            if group!='L':
                output=OUT/f'{tag}.diagnostic.json'
                if not output.exists():call('scripts.diagnose_memory',['--checkpoint',run/'last.pt','--output',output],gpu,LOG/f'{tag}.diagnostic.log')
            print('DONE',tag,flush=True);return []
        except Exception as e:
            print('FAILED',tag,e,flush=True);return [{'tag':tag,'error':str(e)}]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:errors=sum(pool.map(train,enumerate(jobs)),[])
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Inspect errors.json and rerun')
    (OUT/'complete').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':main()
