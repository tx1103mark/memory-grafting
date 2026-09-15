"""Explicit, cache-free Qwen3 forward with a single residual memory injection.

The stock decoder blocks and RoPE are reused. Memory never travels through
mutable hooks; non-reentrant checkpointing captures each block explicitly.
"""
import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model


def rms(x):
    return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype)


class MemoryAdapter(nn.Module):
    def __init__(self, memory_dim, hidden_dim, engram_buckets=0, engram_dim=128, shortconv_kernel=0):
        super().__init__()
        self.key = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.value = nn.Linear(memory_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.tensor(-2.0))
        self.alpha = nn.Parameter(torch.tensor(0.001))
        self.engram_buckets = engram_buckets
        if engram_buckets:
            self.engram2 = nn.Embedding(engram_buckets, engram_dim)
            self.engram3 = nn.Embedding(engram_buckets, engram_dim)
            self.engram_key = nn.Linear(2*engram_dim, hidden_dim, bias=False)
            self.engram_value = nn.Linear(2*engram_dim, hidden_dim, bias=False)
        else:
            self.engram2 = self.engram3 = self.engram_key = self.engram_value = None
        self.shortconv_kernel = shortconv_kernel
        self.shortconv = nn.Conv1d(hidden_dim,hidden_dim,shortconv_kernel,groups=hidden_dim,bias=False) if shortconv_kernel else None
        if self.shortconv is not None:
            nn.init.zeros_(self.shortconv.weight)
        self.measure = False
        self.disabled = False
        self.measurements = {}

    def _engram(self, input_ids):
        ids=input_ids.long()
        p1=torch.cat((torch.zeros_like(ids[:,:1]),ids[:,:-1]),1)
        p2=torch.cat((torch.zeros_like(ids[:,:2]),ids[:,:-2]),1)
        h2=((p1.remainder(self.engram_buckets)*1000003).remainder(self.engram_buckets)+ids*9176).remainder(self.engram_buckets)
        h3=((p2.remainder(self.engram_buckets)*1000033).remainder(self.engram_buckets)+
            (p1.remainder(self.engram_buckets)*9176).remainder(self.engram_buckets)+ids*6361).remainder(self.engram_buckets)
        return torch.cat((self.engram2(h2),self.engram3(h3)),-1)

    def forward(self, h, memory, hit, input_ids=None, valid=None):
        if self.disabled:
            self.measurements = {}
            return h
        e = rms(memory).to(self.key.weight.dtype)
        key, value = self.key(e), self.value(e)
        if self.engram_buckets:
            if input_ids is None:raise ValueError('Engram fallback requires input_ids')
            eng=self._engram(input_ids)
            key=torch.where(hit.unsqueeze(-1),key,self.engram_key(rms(eng)))
            value=torch.where(hit.unsqueeze(-1),value,self.engram_value(rms(eng)))
            write=valid.bool() if valid is not None else torch.ones_like(hit)
        else:
            write=hit
        score = (rms(h).float() * rms(key).float()).sum(-1, keepdim=True) / math.sqrt(h.shape[-1])
        gate = torch.sigmoid(score + self.bias.float())
        update=gate*value.float()
        if self.shortconv is not None:
            causal=torch.nn.functional.pad(update.transpose(1,2),(self.shortconv_kernel-1,0))
            update=update+self.shortconv(causal).transpose(1,2).float()
        delta = self.alpha.float() * write.unsqueeze(-1) * update
        if self.measure:
            with torch.no_grad():
                rounded = (h + delta.to(h.dtype)).float() - h.float()
                self.measurements = {}
                masks = {'hit': hit, 'fallback': write & ~hit} if self.engram_buckets else {'hit': hit}
                for prefix, selected in masks.items():
                    if not selected.any():
                        continue
                    gates = gate.squeeze(-1)[selected].float()
                    denom = h.float().norm(dim=-1)[selected].clamp_min(1e-8)
                    ratio = delta.float().norm(dim=-1)[selected] / denom
                    effective = rounded.norm(dim=-1)[selected] / denom
                    self.measurements.update({
                        f'{prefix}_gate_mean': gates.mean().item(),
                        f'{prefix}_gate_p50': gates.median().item(),
                        f'{prefix}_gate_p90': torch.quantile(gates,.9).item(),
                        f'{prefix}_delta_ratio_mean': ratio.mean().item(),
                        f'{prefix}_effective_delta_ratio_mean': effective.mean().item(),
                        f'{prefix}_rounded_zero_fraction': (rounded[selected]==0).float().mean().item(),
                    })
        return h + delta.to(h.dtype)


class GraftedLM(nn.Module):
    def __init__(self, backbone, table=None, layer=2, checkpointing=False,engram_buckets=0,shortconv_kernel=0):
        super().__init__()
        self.backbone = backbone
        self.layer = layer
        self.checkpointing = checkpointing
        base = self.base
        self.config = base.config
        if table is not None and not 0 <= layer < self.config.num_hidden_layers:
            raise ValueError('Injection layer index outside student blocks')
        self.memory = None if table is None else nn.Embedding.from_pretrained(table, freeze=True, padding_idx=0)
        self.adapter = None if table is None else MemoryAdapter(table.shape[1], self.config.hidden_size,
            engram_buckets=engram_buckets,shortconv_kernel=shortconv_kernel)
        if self.adapter is not None:
            # Keep trainable adapter master weights and small gates in FP32.
            self.adapter.to(device=base.model.embed_tokens.weight.device)

    @property
    def base(self):
        return self.backbone.get_base_model() if hasattr(self.backbone, 'get_base_model') else self.backbone

    def forward(self, input_ids, attention_mask=None, memory_ids=None, **unused):
        core = self.base.model
        h = core.embed_tokens(input_ids)
        batch, length = input_ids.shape
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        positions = torch.arange(length, device=h.device)
        position_ids = positions.unsqueeze(0).expand(batch, -1)
        allowed = positions[None, :] <= positions[:, None]
        allowed = allowed[None, None] & attention_mask[:, None, None, :].bool()
        mask = torch.zeros((batch, 1, length, length), device=h.device, dtype=h.dtype)
        mask.masked_fill_(~allowed, torch.finfo(h.dtype).min)
        rotary = core.rotary_emb(h, position_ids)
        if self.memory is not None:
            if memory_ids is None:
                raise ValueError('Explicit memory_ids required')
            values = self.memory(memory_ids)
        for index, block in enumerate(core.layers):
            def run(x, block=block):
                result = block(x, attention_mask=mask, position_ids=position_ids,
                               position_embeddings=rotary, cache_position=positions, use_cache=False)
                return result[0] if isinstance(result, tuple) else result
            h = checkpoint(run, h, use_reentrant=False) if self.checkpointing and self.training else run(h)
            if self.adapter is not None and index == self.layer:
                h = self.adapter(h, values, memory_ids.ne(0),input_ids,attention_mask)
        return self.base.lm_head(core.norm(h))


def load_model(path, group='G', memory_path=None, device='cuda', checkpointing=True, layer=2,
               engram_buckets=0,shortconv_kernel=0):
    dtype = torch.bfloat16 if str(device).startswith('cuda') else torch.float32
    base = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dtype, attn_implementation='sdpa', local_files_only=True)
    base.config.use_cache = False
    if group != 'B0':
        base = get_peft_model(base, LoraConfig(r=16, lora_alpha=32, lora_dropout=0,
                             target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
                             task_type='CAUSAL_LM', bias='none'))
    table = None
    if group in ('G', 'R', 'S'):
        obj = torch.load(memory_path, map_location='cpu', weights_only=True)
        table = obj['random'] if group == 'R' else obj['teacher']
        if group == 'S':
            # Preserve the exact teacher-vector multiset while breaking key/value semantics.
            order = torch.randperm(len(table)-1, generator=torch.Generator().manual_seed(20260915)) + 1
            table = torch.cat((table[:1], table[order]), dim=0)
    return GraftedLM(base, table, layer=layer, checkpointing=checkpointing,
                     engram_buckets=engram_buckets,shortconv_kernel=shortconv_kernel).to(device)


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: t.detach().cpu() for n, t in model.state_dict().items() if n in names}


def restore_trainable(model, state):
    expected = {n for n, p in model.named_parameters() if p.requires_grad}
    if set(state) != expected:
        raise ValueError(f'Checkpoint parameter mismatch: {set(state) ^ expected}')
    model.load_state_dict(state, strict=False)
