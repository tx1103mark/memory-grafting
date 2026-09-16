import copy
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from graft.model import MemoryAdapter, GraftedLM, trainable_state, restore_trainable, load_model
from graft.data import lookup, chunks, collate
from scripts.train import loss_sum


def tiny():
    torch.manual_seed(7)
    config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=4,
                      num_attention_heads=4,num_key_value_heads=2,head_dim=8,
                      attention_dropout=0.,pad_token_id=0)
    config._attn_implementation='sdpa'
    return Qwen3ForCausalLM(config).eval()


def test_stock_forward_parity_with_padding():
    base=tiny()
    x=torch.tensor([[1,2,3,4,5],[4,3,2,0,0]])
    mask=x.ne(0).long()
    with torch.no_grad():
        expected=base(input_ids=x,attention_mask=mask,use_cache=False).logits
        actual=GraftedLM(base)(x,mask)
    torch.testing.assert_close(actual[mask.bool()],expected[mask.bool()],atol=1e-6,rtol=1e-5)


def test_lookup_longest_causal_and_document_reset():
    keys={(1,2):1,(1,2,3):2,(1,2,3,4):3,(3,4):4}
    assert lookup([1,2,3,4],keys)==[0,1,2,3]
    assert lookup([3,4],keys)==[0,4]
    assert lookup([1,2,3,9],keys)[:3]==lookup([1,2,3,4],keys)[:3]


def test_chunk_targets_and_eos_padding():
    ids=list(range(1,26))
    rows=list(chunks(ids,[0]*len(ids),size=8))
    targets=[v for r in rows for v in r['labels'][1:] if v!=-100]
    assert targets==ids[1:]
    batch=collate([{'input_ids':[1,2,9],'labels':[-100,2,9],'memory_ids':[0,0,0]},
                   {'input_ids':[3,9],'labels':[-100,9],'memory_ids':[0,0]}],pad_id=9)
    assert batch['labels'].tolist()==[[-100,2,9],[-100,9,-100]]


def test_misses_are_exact_identity_and_no_future_leakage():
    base=tiny()
    table=torch.randn(5,16)
    table[0]=0
    model=GraftedLM(base,table).eval()
    x=torch.tensor([[1,2,3,4]])
    with torch.no_grad():
        expected=GraftedLM(base)(x)
        actual=model(x,memory_ids=torch.zeros_like(x))
        torch.testing.assert_close(actual,expected,atol=0,rtol=0)
        a=model(x,memory_ids=torch.tensor([[0,1,2,3]]))
        b=model(torch.tensor([[1,2,3,8]]),memory_ids=torch.tensor([[0,1,2,4]]))
        torch.testing.assert_close(a[:,:3],b[:,:3],atol=1e-6,rtol=1e-5)


def test_gradient_frozen_suffix_checkpoint_and_roundtrip():
    base=tiny().requires_grad_(False)
    table=torch.randn(5,16);table[0]=0
    model=GraftedLM(base,table,layer=1,checkpointing=True).train()
    x=torch.tensor([[1,2,3,4]])
    mids=torch.tensor([[0,1,2,3]])
    loss=loss_sum(model(x,memory_ids=mids),x)/3
    loss.backward()
    assert model.memory.weight.grad is None
    assert all(p.grad is None for p in base.parameters())
    for p in model.adapter.parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all() and p.grad.abs().sum()>0
    optimizer=torch.optim.AdamW(model.adapter.parameters(),lr=.001)
    optimizer.step()
    replica=GraftedLM(copy.deepcopy(base),table,layer=1)
    restore_trainable(replica,trainable_state(model))
    torch.testing.assert_close(model(x,memory_ids=mids),replica(x,memory_ids=mids))


def test_accumulated_loss_weights_tokens():
    logits=torch.randn(2,5,64,requires_grad=True)
    labels=torch.tensor([[-100,1,2,3,4],[-100,5,6,-100,-100]])
    whole=loss_sum(logits,labels)/6
    parts=sum(loss_sum(logits[i:i+1],labels[i:i+1])/6 for i in range(2))
    torch.testing.assert_close(whole,parts)


def test_shuffled_teacher_preserves_rows_and_breaks_mapping(tmp_path,monkeypatch):
    table=torch.arange(80,dtype=torch.float32).reshape(5,16);table[0]=0
    memory=tmp_path/'table.pt';torch.save({'teacher':table,'random':-table},memory)
    monkeypatch.setattr('graft.model.AutoModelForCausalLM.from_pretrained',lambda *args,**kwargs:tiny())
    shuffled=load_model(tmp_path,'S',memory,'cpu',False).memory.weight.detach()
    assert torch.equal(shuffled[0],table[0])
    assert {tuple(row.tolist()) for row in shuffled[1:]}=={tuple(row.tolist()) for row in table[1:]}
    assert not torch.equal(shuffled[1:],table[1:])


def test_engram_fallback_shortconv_is_causal_and_trainable():
    base=tiny().requires_grad_(False);table=torch.randn(5,16);table[0]=0
    model=GraftedLM(base,table,layer=1,engram_buckets=31,shortconv_kernel=4).train()
    x=torch.tensor([[1,2,3,4]]);mids=torch.zeros_like(x);mask=torch.ones_like(x)
    loss=loss_sum(model(x,mask,mids),x)/3;loss.backward()
    for name in ('engram2.weight','engram3.weight','engram_key.weight','engram_value.weight','shortconv.weight'):
        grad=dict(model.adapter.named_parameters())[name].grad
        assert grad is not None and torch.isfinite(grad).all()
    model.eval()
    with torch.no_grad():
        a=model(x,mask,mids);b=model(torch.tensor([[1,2,3,8]]),mask,mids)
    torch.testing.assert_close(a[:,:3],b[:,:3],atol=1e-6,rtol=1e-5)


def test_teacher_memory_can_be_disabled_while_fallback_remains_active():
    torch.manual_seed(5)
    adapter=MemoryAdapter(7,8,engram_buckets=31,engram_dim=4,teacher_memory=False)
    h=torch.randn(1,4,8); memory=torch.randn(1,4,7); ids=torch.tensor([[1,2,3,4]])
    hit=torch.ones(1,4,dtype=torch.bool); valid=torch.ones_like(hit)
    y1=adapter(h,memory,hit,ids,valid)
    y2=adapter(h,torch.randn_like(memory),hit,ids,valid)
    torch.testing.assert_close(y1,y2)
    assert not torch.equal(y1,h)


def test_aligned_identity_initialization_and_freezing():
    adapter=MemoryAdapter(8,8,aligned_init='frozen')
    torch.testing.assert_close(adapter.key.weight,torch.eye(8))
    torch.testing.assert_close(adapter.value.weight,torch.eye(8))
    assert not adapter.key.weight.requires_grad and not adapter.value.weight.requires_grad
