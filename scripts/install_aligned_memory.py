"""Install the selected alignment artifact as a standard memory directory."""
import hashlib,json,shutil
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];source=ROOT/'runs/alignment_study';dest=ROOT/'memory_aligned/T12'
dest.mkdir(parents=True,exist_ok=True);shutil.copy2(source/'aligned_table.pt',dest/'table.pt')
base=json.loads((ROOT/'memory_500k/T12/manifest.json').read_text())
base.update(aligned=True,alignment_method=json.loads((source/'results.json').read_text())['selected_transform'],
            alignment_objective=json.loads((source/'results.json').read_text())['selected_objective'],
            alignment_results_sha256=hashlib.sha256((source/'results.json').read_bytes()).hexdigest(),memory_dim=1024)
(dest/'manifest.json').write_text(json.dumps(base,indent=2))
print(dest)
