"""Deterministic Chinese sampling, benchmark exclusion, document split and keys."""
import argparse
import collections
import csv
import hashlib
import io
import json
import random
import re
import zipfile
from pathlib import Path
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoTokenizer
from graft.data import normalized, lookup, chunks, load_keys


def benchmark_rows(root):
    for dataset in ('ceval', 'cmmlu'):
        folder = root / 'data/raw' / dataset
        for file in folder.rglob('*.parquet'):
            split = file.name.split('-')[0]
            subject = file.parent.name
            for idx, row in enumerate(pq.read_table(file).to_pylist()):
                r = {k.lower():v for k,v in row.items()}
                yield dict(dataset=dataset, split=split, subject=subject, id=f'{dataset}/{subject}/{split}/{idx}',
                           question=r['question'], choices=[r[c] for c in 'abcd'], answer=str(r.get('answer') or '').strip().upper())
        csvs = [(p.relative_to(folder).as_posix(), p.read_bytes()) for p in folder.rglob('*.csv')]
        for path in folder.rglob('*.zip'):
            with zipfile.ZipFile(path) as z:
                csvs += [(name, z.read(name)) for name in z.namelist() if name.endswith('.csv')]
        for name, content in csvs:
            parts = name.replace('\\', '/').split('/')
            split = next((s for s in ('dev', 'val', 'test') if s in parts or f'_{s}.' in name), None)
            if split is None:
                continue
            subject = re.sub(r'_(dev|val|test)$', '', Path(name).stem)
            for idx, row in enumerate(csv.DictReader(io.StringIO(content.decode('utf-8-sig')))):
                row = {k.lower(): v for k, v in row.items() if k is not None}
                if 'question' not in row:
                    raise ValueError(f'Unexpected benchmark columns: {name}: {row.keys()}')
                yield dict(dataset=dataset, split=split, subject=subject, id=f'{dataset}/{name}/{idx}',
                           question=row['question'], choices=[row[c] for c in 'abcd'], answer=row.get('answer', '').strip().upper())


def main():
    from datasketch import MinHash, MinHashLSH
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--train-tokens', type=int, default=50_000_000)
    p.add_argument('--val-tokens', type=int, default=1_000_000)
    p.add_argument('--keys-per-order', type=int, default=20_000)
    a = p.parse_args()
    out = a.root/'data/processed'
    out.mkdir(parents=True, exist_ok=True)
    if (out/'manifest.json').exists():
        print('Already prepared; use a new output root to change data settings.')
        return
    rows = list(benchmark_rows(a.root))
    if not rows or set(r['dataset'] for r in rows) != {'ceval', 'cmmlu'}:
        raise RuntimeError('Both benchmark CSV archives required; inspect downloaded layout.')
    (out/'benchmarks.json').write_text(json.dumps(rows, ensure_ascii=False))
    contamination = set()
    for row in rows:
        # Exclude overlap with individual questions and options as well as the combined text.
        for text in [row['question'], *row['choices'], row['question'] + ''.join(row['choices'])]:
            s = normalized(text)
            contamination.update(s[i:i+30] for i in range(max(0, len(s)-29)))
    tokenizer = AutoTokenizer.from_pretrained(a.root/'models/student', local_files_only=True)
    special_ids = set(tokenizer.all_special_ids)
    lock = json.loads((a.root/'assets.lock.json').read_text())
    files = [a.root/'data/raw/corpus'/s for s in lock['corpus']['selected_files']]
    handles = {s: (out/f'{s}.jsonl').open('w', encoding='utf-8') for s in ('train', 'validation')}
    totals = collections.Counter()
    rejected = collections.Counter()
    exact = set()
    lsh = MinHashLSH(threshold=0.8, num_perm=64)
    counts = {n: collections.Counter() for n in (2, 3, 4)}
    ratio = a.val_tokens / (a.val_tokens + a.train_tokens)
    done = False
    try:
        for file in files:
            print('Reading', file, flush=True)
            for batch in pq.ParquetFile(file).iter_batches(batch_size=256):
                for row_index, row in enumerate(batch.to_pylist()):
                    text = row.get('content') or row.get('text')
                    if not isinstance(text, str):
                        raise ValueError(f'No text/content in {row.keys()}')
                    s = normalized(text)
                    if len(s) < 200 or sum('\u4e00' <= c <= '\u9fff' for c in s)/len(s) < .5:
                        rejected['quality'] += 1
                        continue
                    digest = hashlib.sha256(s.encode()).hexdigest()
                    if digest in exact:
                        rejected['exact_duplicate'] += 1
                        continue
                    exact.add(digest)
                    if any(s[i:i+30] in contamination for i in range(len(s)-29)):
                        rejected['benchmark_overlap'] += 1
                        continue
                    # Drop near duplicates before assigning splits; surviving documents represent clusters.
                    mh = MinHash(num_perm=64)
                    mh.update_batch([s[i:i+5].encode() for i in range(0, len(s)-4, 3)])
                    if lsh.query(mh):
                        rejected['near_duplicate'] += 1
                        continue
                    lsh.insert(digest, mh)
                    split = 'validation' if int(digest[:8],16)/2**32 < ratio else 'train'
                    budget = a.val_tokens if split == 'validation' else a.train_tokens
                    if totals[split] >= budget:
                        continue
                    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
                    record = dict(doc_id=digest, input_ids=ids, source_file=str(file.relative_to(a.root)), text=text)
                    handles[split].write(json.dumps(record, ensure_ascii=False)+'\n')
                    totals[split] += len(ids)-1
                    totals[split+'_documents'] += 1
                    if split == 'train':
                        for n in counts:
                            counts[n].update(set(tuple(ids[i:i+n]) for i in range(len(ids)-n+1)
                                                 if not any(t in special_ids for t in ids[i:i+n])))
                    if totals['train_documents'] % 1000 == 0:
                        print(dict(totals), dict(rejected), flush=True)
                    if totals['train'] >= a.train_tokens and totals['validation'] >= a.val_tokens:
                        done = True
                        break
                if done:
                    break
            if done:
                break
    finally:
        for h in handles.values():
            h.close()
    if not done:
        raise RuntimeError(f'Insufficient downloaded text: {dict(totals)}. Download more shards before training.')
    special = set(tokenizer.all_special_ids)
    keys = []
    for n in (2,3,4):
        # Round-trip eliminates incomplete byte fragments from teacher text encoding.
        candidates = sorted(counts[n].items(), key=lambda kv: (-kv[1], kv[0]))
        selected = 0
        for ids, count in candidates:
            if count < 2:
                break
            text = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            if '\ufffd' in text or tuple(tokenizer.encode(text, add_special_tokens=False)) != ids:
                continue
            keys.append(dict(ids=list(ids), text=text, document_frequency=count))
            selected += 1
            if selected == a.keys_per_order:
                break
        if selected != a.keys_per_order:
            raise RuntimeError(f'Only {selected} valid order-{n} keys; need more corpus or explicit smaller key budget')
    (out/'keys.json').write_text(json.dumps(keys, ensure_ascii=False))
    key_map = load_keys(out/'keys.json')
    for split in ('train','validation'):
        def generate(split=split):
            with (out/f'{split}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    ids = json.loads(line)['input_ids']
                    yield from chunks(ids, lookup(ids, key_map))
        Dataset.from_generator(generate).save_to_disk(str(out/split))
    manifest = dict(settings=vars(a)|{'root':str(a.root)}, counts=dict(totals), rejected=dict(rejected),
                    key_count=len(keys), source_revisions={k:v['revision'] for k,v in lock.items()},
                    benchmark_counts=dict(collections.Counter(f"{r['dataset']}/{r['split']}" for r in rows)),
                    decontamination='exact normalized 30-character substring; MinHash64 0.8, 5-char shingles stride3',
                    caveat='Approximate near-dedup and exact-substring exclusion do not guarantee semantic decontamination.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(json.dumps(manifest, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
