"""Held-out linear alignment and CKA for correct versus shuffled teacher rows."""
import argparse
import json
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path('.'))
p.add_argument('--memory-dir',type=Path,required=True)
p.add_argument('--student-block',type=int,required=True)
p.add_argument('--samples',type=int,default=20000)
p.add_argument('--steps',type=int,default=300)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.manual_seed(20260915)
keys=json.loads((a.root/'data/processed/keys.json').read_text(encoding='utf-8'))
n=min(a.samples,len(keys));indices=torch.randperm(len(keys),generator=torch.Generator().manual_seed(7))[:n]
table=torch.load(a.root/a.memory_dir/'table.pt',map_location='cpu',weights_only=True)['teacher'][1:][indices].float()
tok=AutoTokenizer.from_pretrained(a.root/'models/student',local_files_only=True,padding_side='right');tok.pad_token=tok.eos_token
model=AutoModel.from_pretrained(a.root/'models/student',torch_dtype=torch.bfloat16,local_files_only=True,
                                attn_implementation='sdpa').cuda().eval()
targets=[]
with torch.inference_mode():
    for start in range(0,n,256):
        batch=tok([keys[i]['text'] for i in indices[start:start+256]],add_special_tokens=False,padding=True,return_tensors='pt').to('cuda')
        out=model(**batch,output_hidden_states=True,use_cache=False).hidden_states[a.student_block]
        targets.append(out[torch.arange(len(out),device='cuda'),batch.attention_mask.sum(-1)-1].float().cpu())
target=torch.cat(targets);del model
split=int(.8*n);order=torch.randperm(n,generator=torch.Generator().manual_seed(11));train,test=order[:split],order[split:]
shuffle=torch.randperm(n,generator=torch.Generator().manual_seed(20260915))
def cka(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    return (x.T@y).square().sum().div(torch.sqrt((x.T@x).square().sum()*(y.T@y).square().sum())).item()
def fit(x):
    layer=torch.nn.Linear(x.shape[1],target.shape[1],bias=False,device='cuda');torch.nn.init.normal_(layer.weight,std=.001)
    opt=torch.optim.AdamW(layer.parameters(),lr=3e-4,weight_decay=.01)
    x=x.cuda();y=target.cuda()
    gen=torch.Generator(device='cuda').manual_seed(23)
    for _ in range(a.steps):
        batch=train.cuda()[torch.randint(len(train),(256,),generator=gen,device='cuda')]
        loss=1-F.cosine_similarity(layer(F.normalize(x[batch],dim=-1)),F.normalize(y[batch],dim=-1)).mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad():
        pred=layer(F.normalize(x[test.cuda()],dim=-1));gold=y[test.cuda()]
        return {'cosine':F.cosine_similarity(pred,gold).mean().item(),
                'normalized_mse':F.mse_loss(F.normalize(pred,dim=-1),F.normalize(gold,dim=-1)).item()}
probe=min(4000,len(test));xt=table[test[:probe]];yt=target[test[:probe]]
result={'samples':n,'student_block':a.student_block,'memory_dir':str(a.memory_dir),
        'linear_correct':fit(table),'linear_shuffled':fit(table[shuffle]),
        'cka_correct':cka(xt,yt),'cka_shuffled':cka(table[shuffle][test[:probe]],yt),
        'note':'Separate identical-recipe linear maps; held-out rows. Shuffling preserves teacher row multiset.'}
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
