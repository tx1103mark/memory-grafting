import collections
import json
from pathlib import Path
from scripts.prepare_data import benchmark_rows

root=Path('.')
rows=list(benchmark_rows(root))
out=root/'data/processed'
out.mkdir(exist_ok=True)
(out/'benchmarks.json').write_text(json.dumps(rows,ensure_ascii=False))
counts=collections.Counter(f"{r['dataset']}/{r['split']}" for r in rows)
missing=collections.Counter(f"{r['dataset']}/{r['split']}" for r in rows if r['answer'] not in list('ABCD'))
print(json.dumps({'counts':dict(counts),'missing_labels':dict(missing)},ensure_ascii=False,indent=2))
