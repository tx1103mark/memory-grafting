"""Whitening screen, held-out alignment confirmation, and aligned 500k table export."""
import argparse
import json
import math
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModel

def transform(x,method,mean,components,eigenvalues):
    if method=='raw': return x
    z=x-mean
    if method=='center': return z
    if method=='remove16': return z-(z@components[:,:16])@components[:,:16].T
    z=z@components
    if method.startswith('whiten'): z=z/torch.sqrt(eigenvalues.clamp_min(1e-6))
    if method.endswith('_l2'): z=F.normalize(z,dim=-1)
    return z

def cka(x,y):
    x=x-x.mean(0);y=y-y.mean(0)
    return ((x.T@y).square().sum()/torch.sqrt((x.T@x).square().sum()*(y.T@y).square().sum())).item()

def fit(x,y,train,val,test,seed,steps,objective,return_weight=False):
    torch.manual_seed(seed);device='cuda';dim=x.shape[1]
    layer=torch.nn.Linear(dim,y.shape[1],bias=True,device=device)
    torch.nn.init.normal_(layer.weight,std=.001);torch.nn.init.zeros_(layer.bias)
    opt=torch.optim.AdamW(layer.parameters(),lr=3e-4,weight_decay=.01)
    x=x.to(device);y=y.to(device);gen=torch.Generator(device=device).manual_seed(seed+1000)
    ti=train.to(device)
    for _ in range(steps):
        ix=ti[torch.randint(len(ti),(512,),generator=gen,device=device)]
        pred=F.normalize(layer(F.normalize(x[ix].float(),dim=-1)),dim=-1)
        gold=F.normalize(y[ix].float(),dim=-1)
        loss=1-(pred*gold).sum(-1).mean()
        if objective=='mixed':
            loss=loss+0.1*F.cross_entropy(pred@gold.T/0.07,torch.arange(len(ix),device=device))
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(layer.parameters(),1.);opt.step()
    def evaluate(indices):
        indices=indices.to(device);pred=F.normalize(layer(F.normalize(x[indices].float(),dim=-1)),dim=-1)
        gold=F.normalize(y[indices].float(),dim=-1);cos=(pred*gold).sum(-1)
        probe=min(2048,len(indices));sim=pred[:probe]@gold[:probe].T
        target=torch.arange(probe,device=device);rank=(sim.argsort(-1,descending=True)==target[:,None]).nonzero()[:,1]
        return {'cosine':cos.mean().item(),'normalized_mse':F.mse_loss(pred,gold).item(),
                'recall_at_1':(rank<1).float().mean().item(),'recall_at_10':(rank<10).float().mean().item(),
                'cka':cka(pred[:probe],gold[:probe])}
    result={'validation':evaluate(val),'test':evaluate(test)}
    if return_weight: result['state']={k:v.detach().cpu() for k,v in layer.state_dict().items()}
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--samples',type=int,default=60000);p.add_argument('--pca-rank',type=int,default=256)
    p.add_argument('--screen-steps',type=int,default=400);p.add_argument('--confirm-steps',type=int,default=800)
    p.add_argument('--output-dir',type=Path,default=Path('runs/alignment_study'));a=p.parse_args()
    torch.manual_seed(20260916);root=a.root;out=root/a.output_dir;out.mkdir(parents=True,exist_ok=True)
    keys=json.loads((root/'data/processed_500k/keys.json').read_text(encoding='utf-8'))
    table_obj=torch.load(root/'memory_500k/T12/table.pt',map_location='cpu',weights_only=True)
    full=table_obj['teacher'];n=min(a.samples,len(keys));order=torch.randperm(len(keys),generator=torch.Generator().manual_seed(17))[:n]
    teacher=full[1:][order].float();perm=torch.randperm(n,generator=torch.Generator().manual_seed(20260915))
    ntrain=int(.7*n);nval=int(.15*n);train=torch.arange(ntrain);val=torch.arange(ntrain,ntrain+nval);test=torch.arange(ntrain+nval,n)
    # Extract exact-token isolated n-gram targets after student block 1.
    model=AutoModel.from_pretrained(root/'models/student',torch_dtype=torch.bfloat16,local_files_only=True,
                                    attn_implementation='sdpa').cuda().eval();targets=[]
    with torch.inference_mode():
        for start in range(0,n,512):
            seq=[keys[i]['ids'] for i in order[start:start+512].tolist()];lengths=torch.tensor([len(x) for x in seq])
            width=max(map(len,seq));ids=torch.zeros(len(seq),width,dtype=torch.long);mask=torch.zeros_like(ids)
            for j,row in enumerate(seq):ids[j,:len(row)]=torch.tensor(row);mask[j,:len(row)]=1
            hidden=model(input_ids=ids.cuda(),attention_mask=mask.cuda(),output_hidden_states=True,use_cache=False).hidden_states[1]
            targets.append(hidden[torch.arange(len(seq),device='cuda'),lengths.cuda()-1].float().cpu())
    target=torch.cat(targets);del model;torch.cuda.empty_cache()
    # PCA statistics use train keys only; held-out keys never affect whitening.
    stats=teacher[train].cuda();mean=stats.mean(0,keepdim=True);centered=stats-mean
    _,singular,components=torch.pca_lowrank(centered,q=a.pca_rank,center=False,niter=4)
    eigenvalues=singular.square()/(len(train)-1);mean=mean.cpu();components=components.cpu();eigenvalues=eigenvalues.cpu();del stats,centered
    methods=('raw','center','remove16','pca256','whiten256','whiten256_l2');screen={}
    transformed={}
    for method in methods:
        x=transform(teacher,method,mean,components,eigenvalues).half();transformed[method]=x
        screen[method]={
            'correct':fit(x,target,train,val,test,31,a.screen_steps,'cosine'),
            'shuffled':fit(x[perm],target,train,val,test,31,a.screen_steps,'cosine')}
    selected=max(methods,key=lambda m:screen[m]['correct']['validation']['cosine']-screen[m]['shuffled']['validation']['cosine'])
    confirm={};seeds=(31,32,33)
    for objective in ('cosine','mixed'):
        confirm[objective]={}
        for mapping,x in (('correct',transformed[selected]),('shuffled',transformed[selected][perm])):
            confirm[objective][mapping]=[fit(x,target,train,val,test,s,a.confirm_steps,objective) for s in seeds]
    def margin(objective):
        return sum(confirm[objective]['correct'][i]['validation']['recall_at_10']-
                   confirm[objective]['shuffled'][i]['validation']['recall_at_10'] for i in range(3))/3
    selected_objective=max(('cosine','mixed'),key=margin)
    # Fixed seed final projector; train+validation are allowed, test remains untouched.
    train_final=torch.cat((train,val));final=fit(transformed[selected],target,train_final,test,test,31,
                                                       a.confirm_steps*2,selected_objective,True)
    state=final.pop('state')
    artifact={'method':selected,'objective':selected_objective,'mean':mean,'components':components,
              'eigenvalues':eigenvalues,'projection':state,'student_block':1,'teacher_block':12,
              'samples':n,'split':{'train':len(train),'validation':len(val),'test':len(test)}}
    torch.save(artifact,out/'projector.pt')
    # Export a compact 1024-d aligned table for the downstream gated experiment.
    aligned=[torch.zeros(1,target.shape[1],dtype=torch.bfloat16)];weight=state['weight'];bias=state['bias']
    for start in range(0,len(full)-1,4096):
        rows=full[1+start:1+start+4096].float();z=transform(rows,selected,mean,components,eigenvalues)
        aligned.append(F.linear(F.normalize(z.float(),dim=-1),weight,bias).to(torch.bfloat16))
    torch.save({'teacher':torch.cat(aligned)},out/'aligned_table.pt')
    report={'samples':n,'split':artifact['split'],'pca_rank':a.pca_rank,'selected_transform':selected,
            'selected_objective':selected_objective,'screen':screen,'confirm':confirm,'final_test':final,
            'success':all(confirm[selected_objective]['correct'][i]['test']['cosine']>
                          confirm[selected_objective]['shuffled'][i]['test']['cosine'] for i in range(3)),
            'note':'PCA uses train keys only. Model selection uses validation only. Final metrics use held-out keys.'}
    (out/'results.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
