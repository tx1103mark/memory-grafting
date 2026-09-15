"""Recount document-frequency n-grams and rebuild memory IDs for a larger key bank."""
import argparse
import collections
import hashlib
import json
import shutil
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer
from graft.data import chunks,load_keys,lookup

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('.'))
p.add_argument('--source',type=Path,default=Path('data/processed'))
p.add_argument('--output',type=Path,default=Path('data/processed_500k'))
p.add_argument('--total-keys',type=int,default=500000)
a=p.parse_args();source=a.root/a.source;out=a.root/a.output
out.mkdir(parents=True,exist_ok=True)
if (out/'manifest.json').exists():raise FileExistsError(f'Refusing to overwrite {out}')
tok=AutoTokenizer.from_pretrained(a.root/'models/student',local_files_only=True)
special=set(tok.all_special_ids);counts={n:collections.Counter() for n in (2,3,4)}
documents=0
with (source/'train.jsonl').open(encoding='utf-8') as f:
    for line in f:
        ids=json.loads(line)['input_ids'];documents+=1
        for n in counts:
            counts[n].update(set(tuple(ids[i:i+n]) for i in range(len(ids)-n+1)
                                 if not any(x in special for x in ids[i:i+n])))
        if documents%1000==0:print('Counted',documents,flush=True)
targets={2:(a.total_keys+2)//3,3:(a.total_keys+1)//3,4:a.total_keys//3}
keys=[]
for n in (2,3,4):
    selected=0
    for ids,count in sorted(counts[n].items(),key=lambda x:(-x[1],x[0])):
        if count<2:break
        text=tok.decode(ids,clean_up_tokenization_spaces=False)
        if '\ufffd' in text or tuple(tok.encode(text,add_special_tokens=False))!=ids:continue
        keys.append({'ids':list(ids),'text':text,'document_frequency':count});selected+=1
        if selected==targets[n]:break
    if selected!=targets[n]:raise RuntimeError(f'Only {selected} valid order-{n} keys; wanted {targets[n]}')
(out/'keys.json').write_text(json.dumps(keys,ensure_ascii=False),encoding='utf-8')
mapping=load_keys(out/'keys.json')
for split in ('train','validation'):
    shutil.copy2(source/f'{split}.jsonl',out/f'{split}.jsonl')
    def generate(split=split):
        with (source/f'{split}.jsonl').open(encoding='utf-8') as f:
            for line in f:
                ids=json.loads(line)['input_ids'];yield from chunks(ids,lookup(ids,mapping))
    Dataset.from_generator(generate).save_to_disk(str(out/split))
manifest={'source_manifest_sha256':hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest(),
          'source':str(a.source),'key_count':len(keys),'keys_by_order':targets,'documents':documents,
          'selection':'document frequency desc, token tuple tie-break, round-trip valid, minimum df=2'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(manifest),flush=True)
