"""Biomedical domain pilot inspired by TinyEngram, with leakage-safe MMLU evaluation."""
import concurrent.futures,fcntl,json,os,queue,subprocess,time
from pathlib import Path
from datasets import load_dataset
from scripts.run_layer_study import call

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'runs/biomedical_pilot';LOG=ROOT/'logs/biomedical_pilot'
TASKS='mmlu_clinical_knowledge,mmlu_professional_medicine,mmlu_medical_genetics,mmlu_anatomy,mmlu_high_school_world_history'
SUBJECTS=tuple(x.removeprefix('mmlu_') for x in TASKS.split(','))

def wait_free(gpu):
    while int(subprocess.check_output(['nvidia-smi',f'--id={gpu}','--query-gpu=memory.free',
                                      '--format=csv,noheader,nounits'],text=True).strip())<30000:time.sleep(30)

def main():
    os.chdir(ROOT);OUT.mkdir(parents=True,exist_ok=True);LOG.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'runner.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not (ROOT/'data/biomedical/manifest.json').exists():
        subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','scripts.prepare_biomedical'],cwd=ROOT,
                       stdout=(LOG/'prepare.log').open('a'),stderr=subprocess.STDOUT,check=True)
    raw=ROOT/'memory_biomedical/T12'
    if not (raw/'manifest.json').exists():
        wait_free(2);call('scripts.build_memory',['--teacher-block',12,'--data-dir','data/biomedical',
             '--output-dir','memory_biomedical/T12','--no-random'],2,LOG/'build_memory.log')
    alignment=ROOT/'runs/biomedical_alignment'
    if not (alignment/'aligned_table.pt').exists():
        wait_free(2);call('scripts.run_alignment_study',['--data-dir','data/biomedical','--memory-dir','memory_biomedical/T12',
             '--output-dir','runs/biomedical_alignment','--samples',60000,'--student-block',1,'--teacher-block',12],2,LOG/'alignment.log')
    memory=ROOT/'memory_biomedical_aligned/T12'
    if not (memory/'manifest.json').exists():
        call('scripts.install_aligned_memory',['--source','runs/biomedical_alignment','--base-memory','memory_biomedical/T12',
             '--dest','memory_biomedical_aligned/T12'],2,LOG/'install.log')
    # Evaluation subprocesses run offline for reproducibility; populate every
    # requested official MMLU configuration before entering that environment.
    for subject in SUBJECTS:
        for split in ('dev','validation','test'):
            load_dataset('cais/mmlu',subject,split=split)
    if not (OUT/'B0.json').exists():
        wait_free(2);call('scripts.evaluate',['--tasks',TASKS,'--output',OUT/'B0.json'],2,LOG/'B0.eval.log')

    jobs=queue.Queue()
    for group in ('L','G','S'):jobs.put(group)
    def worker(gpu):
        errors=[]
        while True:
            try:group=jobs.get_nowait()
            except queue.Empty:return errors
            run=ROOT/'runs'/f'biomedical_{group}_seed42'
            try:
                wait_free(gpu)
                if not (run/'complete.json').exists():
                    args=['--group',group,'--tokens',2_000_000,'--seed',42,'--micro-batch',2,'--accumulation',2,
                          '--student-block',1,'--data-dir','data/biomedical','--run-name',run.name,'--snapshot-tokens',500_000,1_000_000]
                    if group!='L':args+=['--memory-dir','memory_biomedical_aligned/T12','--aligned-init','low-lr']
                    if (run/'last.pt').exists():args+=['--resume',run/'last.pt']
                    call('scripts.train',args,gpu,LOG/f'{group}.train.log')
                output=OUT/f'{group}.json'
                if not output.exists():call('scripts.evaluate',['--checkpoint',run/'last.pt','--tasks',TASKS,
                    '--output',output],gpu,LOG/f'{group}.eval.log')
                if group!='L':
                    output=OUT/f'{group}.diagnostic.json'
                    if not output.exists():call('scripts.diagnose_memory',['--checkpoint',run/'last.pt','--output',output],gpu,LOG/f'{group}.diagnostic.log')
                print('DONE',group,flush=True)
            except Exception as e:errors.append({'group':group,'error':repr(e)});print('FAILED',group,repr(e),flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:errors=sum(pool.map(worker,(2,3)),[])
    (OUT/'errors.json').write_text(json.dumps(errors,indent=2))
    if errors:raise RuntimeError('Inspect errors.json and rerun')
    call('scripts.summarize_biomedical_pilot',[],2,LOG/'summary.log')
    (OUT/'complete').write_text(time.strftime('%Y-%m-%d %H:%M:%S'))

if __name__=='__main__':main()
