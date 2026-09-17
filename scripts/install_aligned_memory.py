"""Install a selected alignment artifact as a standard memory directory."""
import argparse,hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,default=Path('runs/alignment_study'))
p.add_argument('--base-memory',type=Path,default=Path('memory_500k/T12'));p.add_argument('--dest',type=Path,default=Path('memory_aligned/T12'))
a=p.parse_args();source=ROOT/a.source;dest=ROOT/a.dest
dest.mkdir(parents=True,exist_ok=True);shutil.copy2(source/'aligned_table.pt',dest/'table.pt')
base=json.loads((ROOT/a.base_memory/'manifest.json').read_text())
base.update(aligned=True,alignment_method=json.loads((source/'results.json').read_text())['selected_transform'],
            alignment_objective=json.loads((source/'results.json').read_text())['selected_objective'],
            alignment_results_sha256=hashlib.sha256((source/'results.json').read_bytes()).hexdigest(),memory_dim=1024)
(dest/'manifest.json').write_text(json.dumps(base,indent=2))
print(dest)
