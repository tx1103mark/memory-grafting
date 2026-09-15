import json
from pathlib import Path
import torch


def lookup(ids, keys):
    result = [0] * len(ids)
    for t in range(len(ids)):
        for n in (4, 3, 2):
            if t + 1 >= n:
                row = keys.get(tuple(ids[t + 1 - n:t + 1]))
                if row is not None:
                    result[t] = row
                    break
    return result


def load_keys(path):
    rows = json.loads(Path(path).read_text())
    return {tuple(row['ids']): i + 1 for i, row in enumerate(rows)}


def chunks(ids, memory_ids, size=1024):
    # Three context tokens are repeated; each predictable target is supervised once.
    start = 0
    while start < len(ids) - 1:
        end = min(start + size, len(ids))
        context = 1 if start == 0 else 3
        labels = ids[start:end].copy()
        labels[:context] = [-100] * min(context, len(labels))
        if any(v != -100 for v in labels[1:]):
            yield {'input_ids': ids[start:end], 'memory_ids': memory_ids[start:end], 'labels': labels}
        if end == len(ids):
            break
        start = end - 3


def collate(rows, pad_id):
    length = max(len(r['input_ids']) for r in rows)
    result = {}
    for key, pad in [('input_ids', pad_id), ('memory_ids', 0), ('labels', -100)]:
        result[key] = torch.tensor([r[key] + [pad] * (length - len(r[key])) for r in rows], dtype=torch.long)
    result['attention_mask'] = torch.tensor([[1]*len(r['input_ids']) + [0]*(length-len(r['input_ids'])) for r in rows])
    return result


def normalized(text):
    import re
    import unicodedata
    return re.sub(r'\s+', '', unicodedata.normalize('NFKC', text)).lower()
