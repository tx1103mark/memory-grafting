"""Add the shared LoRA baseline and memory-off contrasts to source-layer results."""
import json
from pathlib import Path
import subprocess
import sys

folder=Path('runs/long_source_study')
pairs=((4,1),(8,1),(12,1),(6,6))
rows=[]
for budget in (2,5,10):
    baseline=json.loads((folder/f'L_{budget}M.json').read_text())
    for teacher,student in pairs:
        name=f'G_T{teacher}_S{student}'
        grafted=json.loads((folder/f'{name}_{budget}M.json').read_text())
        row={'budget_million':budget,'teacher_block':teacher,'student_block':student,
             'G':{'accuracy':grafted['groups']['graft_ceval-valid']['acc,none'],
                  'macro':grafted['subject_macro_accuracy']['graft_ceval-valid']},
             'L':{'accuracy':baseline['groups']['graft_ceval-valid']['acc,none'],
                  'macro':baseline['subject_macro_accuracy']['graft_ceval-valid']}}
        comparisons={'L':folder/f'L_{budget}M.json'}
        if budget==10:comparisons['off']=folder/f'{name}_10M.off.json'
        for label,base in comparisons.items():
            output=folder/f'{name}_{budget}M_vs_{label}.comparison.json'
            subprocess.run([sys.executable,'-m','scripts.compare_results','--baseline',str(base),
                            '--grafted',str(folder/f'{name}_{budget}M.json'),'--output',str(output)],check=True,
                           capture_output=True,text=True)
            row[f'vs_{label}']=json.loads(output.read_text())
        rows.append(row)
(folder/'combined_comparisons.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
