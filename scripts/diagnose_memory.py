"""Paired on/off loss on identical target tokens; hit refers to predictor position."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer
from graft.data import collate
from graft.model import load_model, restore_trainable

@torch.no_grad()
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=100000)
    a=p.parse_args()
    torch.set_num_threads(8)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    memory_dir=Path(saved['manifest'].get('memory_dir','memory'))
    model=load_model('models/student',saved['manifest']['group'],memory_dir/'table.pt','cuda',False,
                     layer=saved['manifest'].get('student_block',3)-1,
                     engram_buckets=saved['manifest'].get('engram_buckets',0),
                     shortconv_kernel=saved['manifest'].get('shortconv_kernel',0),
                     teacher_memory=not saved['manifest'].get('disable_teacher_memory',False)).eval()
    restore_trainable(model,saved['model'])
    tok=AutoTokenizer.from_pretrained('models/student',local_files_only=True)
    data=load_from_disk(str(Path(saved['manifest'].get('data_dir','data/processed'))/'validation'))
    sums={mode:{s:[0.,0] for s in ('hit','miss')} for mode in ('normal','zero')}
    measured=[]; seen=0
    for row in data:
        batch={k:v.cuda() for k,v in collate([row],tok.pad_token_id).items()}
        valid=batch['labels'][:,1:].ne(-100)
        indices=valid.nonzero()
        if len(indices)>a.tokens-seen:
            extra=indices[a.tokens-seen:]; valid[extra[:,0],extra[:,1]]=False
        hits=batch['memory_ids'][:,:-1].ne(0)
        for mode in ('normal','zero'):
            model.adapter.measure=mode=='normal'
            model.adapter.disabled=mode=='zero'
            logits=model(**batch)
            model.adapter.disabled=False
            losses=F.cross_entropy(logits[:,:-1].float().transpose(1,2),batch['labels'][:,1:],ignore_index=-100,reduction='none')
            for name,mask in [('hit',valid&hits),('miss',valid&~hits)]:
                sums[mode][name][0]+=losses[mask].sum().item(); sums[mode][name][1]+=mask.sum().item()
            if mode=='normal' and model.adapter.measurements:
                measured.append(model.adapter.measurements.copy())
        seen+=valid.sum().item()
        if seen>=a.tokens: break
    result=dict(checkpoint=str(a.checkpoint),tokens=seen,
        loss={m:{s:dict(mean=v[0]/max(v[1],1),count=v[1]) for s,v in d.items()} for m,d in sums.items()},
        activation_document_mean={k:sum(x[k] for x in measured if k in x)/sum(k in x for x in measured)
                                  for k in sorted({k for x in measured for k in x})} if measured else {},
        note='Activation summaries average per-document statistics. Hit at t predicts target t+1. Miss loss can change through earlier injected context.')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
