import json
from pathlib import Path

for path in sorted(Path('runs').glob('gate_*/metrics.jsonl')):
    rows=path.read_text().splitlines()
    if rows:
        row=json.loads(rows[-1])
        print(path.parent.name,json.dumps({k:row[k] for k in ('step','tokens','loss','alpha','validation_loss','gate_mean','delta_ratio_mean') if k in row}))
for path in sorted(Path('runs/gate_study').glob('*.json')):
    row=json.loads(path.read_text())
    if 'groups' in row:
        print(path.stem,'acc',row['groups']['graft_ceval-valid']['acc,none'],'macro',row['subject_macro_accuracy'])

