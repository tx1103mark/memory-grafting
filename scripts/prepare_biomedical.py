"""Prepare leakage-screened biomedical CLM data and deterministic n-gram keys."""
import argparse,collections,hashlib,json
from pathlib import Path
import pyarrow.parquet as pq
from datasets import Dataset,load_dataset
from transformers import AutoTokenizer
from graft.data import chunks,load_keys,lookup,normalized

def benchmark_spans():
    rows=[]
    for split in ('dev','validation','test'):
        for row in load_dataset('cais/mmlu','clinical_knowledge',split=split):
            rows.append({'split':split,'question':row['question'],'choices':row['choices'],'answer':row['answer']})
    spans=set()
    for row in rows:
        for text in (row['question'],*row['choices'],row['question']+' '.join(row['choices'])):
            value=normalized(text);spans.update(value[i:i+30] for i in range(max(0,len(value)-29)))
    return rows,spans

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--train-tokens',type=int,default=5_000_000);p.add_argument('--val-tokens',type=int,default=500_000)
    p.add_argument('--keys-per-order',type=int,default=20_000);p.add_argument('--output-dir',type=Path,default=Path('data/biomedical'))
    a=p.parse_args();out=a.root/a.output_dir;out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():print('Already prepared');return
    source=a.root/'data/raw/biomed/data/commercial-00000-of-00026.parquet'
    source_manifest=a.root/'data/raw/biomed/manifest.json'
    if not source.exists() or not source_manifest.exists():raise FileNotFoundError('Run scripts.download_biomedical first')
    benchmark,contamination=benchmark_spans();(out/'mmlu_clinical_snapshot.json').write_text(json.dumps(benchmark,ensure_ascii=False),encoding='utf-8')
    tok=AutoTokenizer.from_pretrained(a.root/'models/student',local_files_only=True);special=set(tok.all_special_ids)
    handles={s:(out/f'{s}.jsonl').open('w',encoding='utf-8') for s in ('train','validation')}
    totals=collections.Counter();rejected=collections.Counter();exact=set();counts={n:collections.Counter() for n in (2,3,4)}
    done=False
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=512,columns=['id','text','language','domain','educational_score']):
            for row in batch.to_pylist():
                if row['language']!='en' or row['domain']!='biomedical':rejected['not_english_biomedical']+=1;continue
                score=row.get('educational_score') or 0
                split='train' if score>4.0 else ('validation' if score<4.0 else None)
                if split is None:rejected['score_boundary']+=1;continue
                budget=a.train_tokens if split=='train' else a.val_tokens
                if totals[split]>=budget:continue
                text=row.get('text');value=normalized(text or '')
                if len(value)<100:rejected['short']+=1;continue
                digest=hashlib.sha256(value.encode()).hexdigest()
                if digest in exact:rejected['exact_duplicate']+=1;continue
                exact.add(digest)
                if any(value[i:i+30] in contamination for i in range(max(0,len(value)-29))):
                    rejected['mmlu_clinical_overlap']+=1;continue
                ids=tok.encode(text,add_special_tokens=False)+[tok.eos_token_id]
                handles[split].write(json.dumps({'doc_id':digest,'input_ids':ids,'text':text,
                    'educational_score':score,'source_id':row.get('id')},ensure_ascii=False)+'\n')
                totals[split]+=len(ids)-1;totals[split+'_documents']+=1
                if split=='train':
                    for n in counts:
                        counts[n].update(set(tuple(ids[i:i+n]) for i in range(len(ids)-n+1)
                                             if not any(x in special for x in ids[i:i+n])))
                if totals['train']>=a.train_tokens and totals['validation']>=a.val_tokens:done=True;break
            if done:break
    finally:
        for handle in handles.values():handle.close()
    if not done:raise RuntimeError(f'Insufficient data: {dict(totals)}')
    keys=[]
    for n in (2,3,4):
        selected=0
        for ids,frequency in sorted(counts[n].items(),key=lambda x:(-x[1],x[0])):
            if frequency<2:break
            text=tok.decode(ids,clean_up_tokenization_spaces=False)
            if '\ufffd' in text or tuple(tok.encode(text,add_special_tokens=False))!=ids:continue
            keys.append({'ids':list(ids),'text':text,'document_frequency':frequency});selected+=1
            if selected==a.keys_per_order:break
        if selected!=a.keys_per_order:raise RuntimeError(f'Only {selected} valid order-{n} keys')
    (out/'keys.json').write_text(json.dumps(keys,ensure_ascii=False),encoding='utf-8');key_map=load_keys(out/'keys.json')
    for split in ('train','validation'):
        def generate(split=split):
            with (out/f'{split}.jsonl').open(encoding='utf-8') as f:
                for line in f:
                    ids=json.loads(line)['input_ids'];yield from chunks(ids,lookup(ids,key_map))
        Dataset.from_generator(generate).save_to_disk(str(out/split))
    manifest={'settings':vars(a)|{'root':str(a.root),'output_dir':str(a.output_dir)},'counts':dict(totals),
              'rejected':dict(rejected),'key_count':len(keys),'source':json.loads(source_manifest.read_text()),
              'mmlu_clinical_counts':dict(collections.Counter(x['split'] for x in benchmark)),
              'split_rule':'TinyEngram-style: English biomedical, train educational_score>4, validation <4',
              'decontamination':'Exact normalized 30-character overlap against MMLU clinical dev/validation/test'}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(manifest))

if __name__=='__main__':main()
