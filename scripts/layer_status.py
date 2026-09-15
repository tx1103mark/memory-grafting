from pathlib import Path
import json
for p in sorted(Path('runs').glob('layer_*/metrics.jsonl')):
    lines=p.read_text().splitlines()
    if lines:
        r=json.loads(lines[-1]);print(p.parent.name,r['tokens'],r.get('validation_loss',''))
for p in sorted(Path('runs/layer_study').glob('*.json')):
    r=json.loads(p.read_text())
    if isinstance(r,dict) and 'groups' in r:
        print(p.stem,round(100*r['groups']['graft_ceval-valid']['acc,none'],4))
