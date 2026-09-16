"""Paired 3-seed downstream gate for the held-out-selected aligned table."""
import concurrent.futures,fcntl,json,os,queue,subprocess,time
from pathlib import Path
from scripts.run_layer_study import call
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/aligned_downstream';LOG=ROOT/'logs/aligned_downstream'

def wait_free(gpu):
    while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip())<30000:time.sleep(30)

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (ROOT/'memory_aligned/T12/manifest.json').exists():call('scripts.install_aligned_memory',[],0,LOG/'install.log')
    jobs=queue.Queue()
    for seed in (42,43,44):
        jobs.put(dict(tag=f'L_seed{seed}',group='L',seed=seed,mode='none'))
        for mode in ('frozen','low-lr'):
            for group in ('G','S'):jobs.put(dict(tag=f'{mode}_{group}_seed{seed}',group=group,seed=seed,mode=mode))
        jobs.put(dict(tag=f'random_G_seed{seed}',group='G',seed=seed,mode='none'))
    def worker(gpu):
        errors=[]
        while True:
            try:j=jobs.get_nowait()
            except queue.Empty:return errors
            tag=j['tag'];run=ROOT/'runs'/f'aligned_{tag}'
            try:
                wait_free(gpu)
                if not (run/'complete.json').exists():
                    args=['--group',j['group'],'--tokens',2_000_000,'--seed',j['seed'],'--micro-batch',2,
                          '--accumulation',2,'--student-block',1,'--data-dir','data/processed_500k',
                          '--run-name',run.name,'--snapshot-tokens',500_000]
                    if j['group']!='L':args+=['--memory-dir','memory_aligned/T12','--aligned-init',j['mode']]
                    if (run/'last.pt').exists():args+=['--resume',run/'last.pt']
                    call('scripts.train',args,gpu,LOG/f'{tag}.train.log')
                sequence=[(2,'cmmlu','last.pt'),(.5,'cmmlu','tokens-500000.pt'),
                          (2,'ceval-valid','last.pt'),(.5,'ceval-valid','tokens-500000.pt')]
                for budget,task,checkpoint in sequence:
                    output=OUT/f'{tag}_{budget:g}M.{task}.json'
                    if not output.exists():call('scripts.evaluate',['--checkpoint',run/checkpoint,'--tasks',task,
                        '--output',output],gpu,LOG/f'{tag}_{budget:g}M.{task}.log')
                if j['group']!='L':
                    output=OUT/f'{tag}.diagnostic.json'
                    if not output.exists():call('scripts.diagnose_memory',['--checkpoint',run/'last.pt','--output',output],
                                                gpu,LOG/f'{tag}.diagnostic.log')
                print('DONE',tag,flush=True)
            except Exception as e:errors.append({'tag':tag,'error':repr(e)});print('FAILED',tag,repr(e),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:errors=sum(pool.map(worker,(0,1,2,3)),[])
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Inspect errors.json and rerun')
    (OUT/'complete').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))
if __name__=='__main__':main()
