import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'runs/biomedical_pilot'
TASKS=('mmlu_clinical_knowledge','mmlu_professional_medicine','mmlu_medical_genetics','mmlu_anatomy','mmlu_high_school_world_history')

def main():
    out={'groups':{},'deltas':{},'primary':'mmlu_clinical_knowledge'}
    for group in ('B0','L','G','S'):
        row=json.loads((SOURCE/f'{group}.json').read_text(encoding='utf-8'))
        out['groups'][group]={task:100*row['results'][task]['acc,none'] for task in TASKS}
    for left,right in (('G','S'),('G','L'),('G','B0'),('L','B0')):
        out['deltas'][f'{left}-{right}']={task:out['groups'][left][task]-out['groups'][right][task] for task in TASKS}
    (SOURCE/'summary.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out))

if __name__=='__main__':main()
