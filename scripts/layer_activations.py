import json
from pathlib import Path
for path in sorted(Path('runs').glob('layer_*/metrics.jsonl')):
    rows=[json.loads(line) for line in path.read_text().splitlines()]
    measured=[r for r in rows if 'gate_mean' in r]
    if measured:
        r=measured[-1]
        print(path.parent.name,r['step'],round(r['gate_mean'],3),round(100*r['effective_delta_ratio_mean'],3))
