"""Three-seed T12->S1 confirmation and paired component ablations on free GPUs 0-3."""
import fcntl
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from scripts.run_layer_study import call

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/confirmation_study'; LOG=ROOT/'logs/confirmation_study'
MEMORY='memory_500k/T12'; DATA='data/processed_500k'

def wait_free(gpu,minimum=30000):
    while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip()) < minimum:
        time.sleep(30)

def specs():
    result=[]
    # Seed 42 full runs already exist in runs/mechanism_* and are reused by the summary.
    for seed in (43,44):
        for group in ('G','S','L'):
            result.append(dict(tag=f'full_{group}_seed{seed}',group=group,seed=seed,variant='full'))
    for seed in (42,43,44):
        result += [
            dict(tag=f'nofallback_G_seed{seed}',group='G',seed=seed,variant='nofallback'),
            dict(tag=f'noshortconv_G_seed{seed}',group='G',seed=seed,variant='noshortconv'),
            dict(tag=f'fallback_only_G_seed{seed}',group='G',seed=seed,variant='fallback_only'),
        ]
    return result

def train_args(job,run):
    args=['--group',job['group'],'--tokens',5_000_000,'--seed',job['seed'],'--micro-batch',2,
          '--accumulation',2,'--student-block',1,'--data-dir',DATA,'--run-name',run.name]
    if job['group']!='L':
        args += ['--memory-dir',MEMORY]
        if job['variant']!='nofallback': args += ['--engram-buckets',65536]
        if job['variant']!='noshortconv': args += ['--shortconv-kernel',4]
        if job['variant']=='fallback_only': args += ['--disable-teacher-memory']
    return args

def main():
    os.chdir(ROOT); OUT.mkdir(parents=True,exist_ok=True); LOG.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    required=[ROOT/'models/student',ROOT/DATA/'manifest.json',ROOT/MEMORY/'manifest.json']
    if missing:=[str(x) for x in required if not x.exists()]: raise FileNotFoundError(missing)
    work=queue.Queue()
    for job in specs(): work.put(job)
    errors=[]; guard=threading.Lock()
    def worker(gpu):
        local=[]
        while True:
            try: job=work.get_nowait()
            except queue.Empty: return local
            tag=job['tag']; run=ROOT/'runs'/f'confirm_{tag}'
            try:
                wait_free(gpu)
                if not (run/'complete.json').exists():
                    args=train_args(job,run)
                    if (run/'last.pt').exists(): args += ['--resume',run/'last.pt']
                    call('scripts.train',args,gpu,LOG/f'{tag}.train.log')
                # CMMLU is deliberately evaluated first and is the preregistered primary metric.
                for task in ('cmmlu','ceval-valid'):
                    output=OUT/f'{tag}.{task}.json'
                    if not output.exists():
                        call('scripts.evaluate',['--checkpoint',run/'last.pt','--tasks',task,'--output',output],
                             gpu,LOG/f'{tag}.{task}.log')
                if job['group']!='L':
                    output=OUT/f'{tag}.diagnostic.json'
                    if not output.exists():
                        call('scripts.diagnose_memory',['--checkpoint',run/'last.pt','--output',output],
                             gpu,LOG/f'{tag}.diagnostic.log')
                print('DONE',tag,flush=True)
            except Exception as exc:
                local.append({'tag':tag,'error':repr(exc)}); print('FAILED',tag,repr(exc),flush=True)
        return local
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for batch in pool.map(worker,(0,1,2,3)): errors.extend(batch)
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors: raise RuntimeError('Inspect errors.json and rerun; completed jobs are reused')
    (OUT/'complete').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__': main()
