"""Compact read-only progress for the long-budget study."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
for name in ('L','G_T6_S1','R_T6_S1','G_T4_S6','R_T4_S6','G_T8_S12','R_T8_S12'):
    run=root/f'runs/long_{name}'
    row=dict(name=name,training_complete=(run/'complete.json').exists())
    path=run/'metrics.jsonl'
    if path.exists():
        lines=path.read_text().splitlines()
        for line in reversed(lines):
            try:metrics=json.loads(line)
            except json.JSONDecodeError:continue
            row.update({key:metrics[key] for key in ('step','tokens','tokens_per_second','validation_loss') if key in metrics})
            break
    row['evaluated_millions']=[n for n in (2,5,10) if (root/f'runs/long_study/{name}_{n}M.json').exists()]
    print(json.dumps(row))
for filename in ('errors.json','complete'):
    path=root/'runs/long_study'/filename
    if path.exists():print(filename,path.read_text())
