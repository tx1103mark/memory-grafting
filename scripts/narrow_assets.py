"""Select one deterministic Ultra-FineWeb shard for the 50M-token experiment."""
import json
from pathlib import Path

path=Path('assets.lock.json')
lock=json.loads(path.read_text())
selected=lock['corpus']['selected_files']
if not selected:
    raise ValueError('No locked Chinese corpus shard')
lock['corpus']['selected_files']=selected[:1]
lock['corpus']['selection_note']='One seed-42 shuffled Chinese shard; sufficient for fixed 50M target-token budget'
lock['teacher']['required_weight_files']=['model-00001-of-00005.safetensors','model-00002-of-00005.safetensors']
lock['teacher']['prefix_blocks']=12
path.write_text(json.dumps(lock,indent=2,ensure_ascii=False))
print(json.dumps({'corpus':lock['corpus']['selected_files'],'teacher':lock['teacher']['required_weight_files']},indent=2))
