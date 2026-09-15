import json
from pathlib import Path
import subprocess
import sys

def main():
    folder=Path('runs/layer_study')
    rows=[]
    for t in (4,8,12):
        for s in (1,3,6,12):
            row=dict(teacher_block=t,student_block=s)
            for g in ('G','R'):
                result=json.loads((folder/f'{g}_T{t}_S{s}.json').read_text())
                row[g]=dict(accuracy=result['groups']['graft_ceval-valid']['acc,none'],macro=result['subject_macro_accuracy']['graft_ceval-valid'])
            for base in ('L',f'R_T{t}_S{s}'):
                output=folder/f'G_T{t}_S{s}_vs_{base}.comparison.json'
                subprocess.run([sys.executable,'-m','scripts.compare_results','--baseline',str(folder/f'{base}.json'),
                    '--grafted',str(folder/f'G_T{t}_S{s}.json'),'--output',str(output)],check=True,capture_output=True)
                row['vs_L' if base=='L' else 'vs_R']=json.loads(output.read_text())
            rows.append(row)
    (folder/'summary.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
