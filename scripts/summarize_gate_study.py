import json
import subprocess
import sys
from pathlib import Path

root=Path('runs/gate_study')
summary={}
for suffix in ('001','01','05'):
    for base in ('gate_L',f'gate_R_a{suffix}'):
        name=f'gate_G_a{suffix}'
        output=root/f'{name}_vs_{base}.comparison.json'
        subprocess.run([sys.executable,'-m','scripts.compare_results','--baseline',str(root/f'{base}.json'),
                        '--grafted',str(root/f'{name}.json'),'--output',str(output)],check=True)
        summary[output.stem]=json.loads(output.read_text())
(root/'comparisons.json').write_text(json.dumps(summary,indent=2))
