"""Resumable bounded GPU workers. Never stop processes belonging to other jobs."""
import concurrent.futures
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/layer_study'
LOG=ROOT/'logs/layer_study'
PY=sys.executable

def call(module,args,gpu,log):
    env=os.environ|{'CUDA_VISIBLE_DEVICES':str(gpu),'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
    with log.open('a') as f:
        subprocess.run([PY,'-u','-m',module,*map(str,args)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)

def reuse(source,name):
    if not source.exists(): raise FileNotFoundError(source)
    shutil.copy2(source,OUT/f'{name}.json')
    for file in source.parent.glob(source.stem+'.*.samples.jsonl'):
        shutil.copy2(file,OUT/(name+file.name[len(source.stem):]))

def main():
    os.chdir(ROOT)
    OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    jobs=queue.Queue()
    for teacher in (4,8):
        folder=ROOT/f'memory/T{teacher}'
        if not (folder/'manifest.json').exists():
            call('scripts.build_memory',['--teacher-block',teacher,'--output-dir',folder],3,LOG/f'build_T{teacher}.log')
    reuse(ROOT/'runs/gate_study/gate_L.json','L')
    for group in ('G','R'):
        reuse(ROOT/f'runs/gate_study/gate_{group}_a001.json',f'{group}_T12_S3')
    for teacher in (4,8,12):
        for student in (1,3,6,12):
            if (teacher,student)==(12,3):continue
            for group in ('G','R'):jobs.put((teacher,student,group))
    def worker(gpu):
        errors=[]
        while True:
            try:t,s,g=jobs.get_nowait()
            except queue.Empty:return errors
            name=f'{g}_T{t}_S{s}';run=ROOT/f'runs/layer_{name}'
            try:
                # Sharing is authorized; a worker still leaves room for existing allocations.
                while True:
                    free=int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).strip())
                    if free>=20000:break
                    print(f'{name} waiting for 20GB free on GPU {gpu}',flush=True);time.sleep(30)
                print(f'START {name} GPU={gpu}',flush=True)
                if not (run/'complete.json').exists():
                    args=['--group',g,'--tokens',2000000,'--seed',42,'--micro-batch',2,'--accumulation',2,
                          '--alpha-init',.001,'--student-block',s,'--memory-dir','memory' if t==12 else f'memory/T{t}',
                          '--run-name',run.name]
                    if (run/'last.pt').exists():args+=['--resume',run/'last.pt']
                    call('scripts.train',args,gpu,LOG/f'{name}.train.log')
                if not (OUT/f'{name}.json').exists():
                    call('scripts.evaluate',['--checkpoint',run/'last.pt','--tasks','ceval-valid','--output',OUT/f'{name}.json'],gpu,LOG/f'{name}.eval.log')
                print(f'DONE {name}',flush=True)
            except Exception as e:
                errors.append(dict(name=name,error=str(e)));print(f'FAILED {name}: {e}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        errors=sum(list(pool.map(worker,range(8))),[])
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError(f'{len(errors)} jobs failed; inspect errors.json and rerun to resume')
    call('scripts.summarize_layer_study',[],0,LOG/'summary.log')
    (OUT/'complete').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':main()
