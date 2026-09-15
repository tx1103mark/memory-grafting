"""Dependency-light CPU correctness runner; does not require pytest plugins."""
import inspect
import json
from pathlib import Path
import torch
from tests import test_core

torch.set_num_threads(2)
passed=[]
for name, fn in inspect.getmembers(test_core, inspect.isfunction):
    if name.startswith('test_'):
        fn()
        passed.append(name)
        print('PASS',name,flush=True)
Path('logs').mkdir(exist_ok=True)
Path('logs/core_test_results.json').write_text(json.dumps({'passed':passed,'device':'cpu','real_pretrained_model':False},indent=2))
