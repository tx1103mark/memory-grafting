"""Single-device, token-budgeted CLM with LoRA and frozen memory controls."""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_from_disk
from transformers import AutoTokenizer
from graft.data import collate
from graft.model import load_model, trainable_state, restore_trainable


def loss_sum(logits, labels):
    return F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1),
                           ignore_index=-100, reduction='sum')


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_save(value, path):
    temp = path.with_suffix('.tmp')
    torch.save(value, temp)
    os.replace(temp, path)


@torch.no_grad()
def validation(model, data, pad, device, limit=100_000):
    model.eval()
    total, count = 0., 0
    for start in range(0, len(data), 2):
        batch = collate([data[i] for i in range(start, min(start+2, len(data)))], pad)
        batch = {k:v.to(device) for k,v in batch.items()}
        n = batch['labels'][:,1:].ne(-100).sum().item()
        total += loss_sum(model(**batch), batch['labels']).item()
        count += n
        if count >= limit:
            break
    model.train()
    return total/count


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--group', choices=['L','R','G','S'], required=True)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--tokens', type=int, default=2_000_000)
    p.add_argument('--micro-batch', type=int, default=2)
    p.add_argument('--accumulation', type=int, default=8)
    p.add_argument('--save-every', type=int, default=100)
    p.add_argument('--eval-every', type=int, default=100)
    p.add_argument('--device', default='cuda')
    p.add_argument('--resume', type=Path)
    p.add_argument('--run-name')
    p.add_argument('--snapshot-tokens',type=int,nargs='*',default=[],help='Save first optimizer boundary at/above each token threshold')
    p.add_argument('--no-checkpointing', action='store_true')
    p.add_argument('--alpha-init', type=float, default=.001)
    p.add_argument('--student-block',type=int,default=3,help='One-based block AFTER which memory is injected')
    p.add_argument('--memory-dir',type=Path,default=Path('memory'))
    p.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    p.add_argument('--engram-buckets',type=int,default=0)
    p.add_argument('--shortconv-kernel',type=int,default=0)
    p.add_argument('--disable-teacher-memory',action='store_true')
    p.add_argument('--aligned-init',choices=['none','frozen','low-lr'],default='none')
    p.add_argument('--stop-after-steps',type=int,help='Debug interruption; saves last.pt without marking the run complete')
    a = p.parse_args()
    if a.tokens <= 0 or a.micro_batch <= 0 or a.accumulation <= 0:
        raise ValueError('Positive token and batch budgets required')
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    torch.set_num_threads(8)
    processed = a.root/a.data_dir
    name = a.run_name or f'{a.group}_seed{a.seed}_{a.tokens}tokens'
    run = a.root/'runs'/name
    run.mkdir(parents=True, exist_ok=True)
    if (run/'last.pt').exists() and not a.resume:
        raise RuntimeError('Run exists; use --resume explicitly or a new --run-name')
    tokenizer = AutoTokenizer.from_pretrained(a.root/'models/student', local_files_only=True)
    data = load_from_disk(str(processed/'train'))
    val = load_from_disk(str(processed/'validation'))
    memory_dir=a.root/a.memory_dir
    if a.group in ('G','R','S'):
        memory_manifest=json.loads((memory_dir/'manifest.json').read_text())
        if memory_manifest['key_sha256'] != file_hash(processed/'keys.json'):
            raise ValueError('Memory table was built with different key ordering')
    model = load_model(a.root/'models/student', a.group, memory_dir/'table.pt', a.device, not a.no_checkpointing,
                       layer=a.student_block-1,engram_buckets=a.engram_buckets,shortconv_kernel=a.shortconv_kernel,
                       teacher_memory=not a.disable_teacher_memory,aligned_init=a.aligned_init)
    if model.adapter is not None:
        with torch.no_grad():
            model.adapter.alpha.fill_(a.alpha_init)
    groups = {}
    for n, param in model.named_parameters():
        if not param.requires_grad:
            continue
        adapter = n.startswith('adapter.')
        aligned_projection = a.aligned_init=='low-lr' and n.startswith(('adapter.key.','adapter.value.'))
        decay = param.ndim >= 2 and 'norm' not in n
        lr = 2e-6 if aligned_projection else (1e-4 if adapter else 2e-5)
        groups.setdefault((lr,decay), []).append(param)
    optimizer = torch.optim.AdamW([dict(params=params,lr=lr,initial_lr=lr,weight_decay=.01 if decay else 0.)
                                  for (lr,decay), params in groups.items()], betas=(.9,.95), eps=1e-8)
    manifest = dict(group=a.group, seed=a.seed, tokens=a.tokens, micro_batch=a.micro_batch, accumulation=a.accumulation,
                    data_hash=file_hash(processed/'manifest.json'), assets_hash=file_hash(a.root/'assets.lock.json'),
                    keys_hash=file_hash(processed/'keys.json'),
                    memory_hash=file_hash(memory_dir/'manifest.json') if a.group in ('G','R','S') else None,
                    trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                    forward='explicit stock Qwen3 blocks; full sequence; no KV cache', learning_rates={'lora':2e-5,'adapter':1e-4})
    manifest.update(data_dir=str(a.data_dir),engram_buckets=a.engram_buckets,shortconv_kernel=a.shortconv_kernel,
                    disable_teacher_memory=a.disable_teacher_memory)
    manifest['aligned_init']=a.aligned_init
    if a.alpha_init != .001:
        manifest['alpha_init'] = a.alpha_init
    if a.student_block != 3 or a.memory_dir != Path('memory'):
        manifest.update(student_block=a.student_block,memory_dir=str(a.memory_dir),
                        teacher_block=memory_manifest['teacher_block'] if a.group in ('G','R','S') else None)
    if a.group == 'S':
        manifest['shuffle_seed'] = 20260915
    step, tokens, epoch, cursor, best = 0,0,0,0,float('inf')
    if a.resume:
        saved = torch.load(a.resume, map_location='cpu', weights_only=False)
        if saved['manifest'] != manifest:
            raise ValueError('Resume manifest mismatch')
        restore_trainable(model, saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        step,tokens,epoch,cursor,best = [saved[k] for k in ['step','tokens','epoch','cursor','best']]
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    (run/'manifest.json').write_text(json.dumps(manifest, indent=2))
    def save(path):
        atomic_save(dict(model=trainable_state(model), optimizer=optimizer.state_dict(), manifest=manifest,
                         step=step,tokens=tokens,epoch=epoch,cursor=cursor,best=best,
                         python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []), path)
    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
    log = (run/'metrics.jsonl').open('a', buffering=1)
    model.train()
    started = time.monotonic()
    initial_tokens = tokens
    while tokens < a.tokens:
        batches = []
        valid = 0
        for _ in range(a.accumulation):
            rows = []
            for _ in range(a.micro_batch):
                if cursor == len(order):
                    epoch += 1
                    cursor = 0
                    order = torch.randperm(len(data), generator=torch.Generator().manual_seed(a.seed+epoch)).tolist()
                rows.append(data[order[cursor]])
                cursor += 1
            batch = collate(rows, tokenizer.pad_token_id or tokenizer.eos_token_id)
            remaining = a.tokens-tokens-valid
            locations = batch['labels'][:,1:].ne(-100).nonzero()
            if len(locations) > remaining:
                masked = locations[remaining:]
                batch['labels'][masked[:,0],masked[:,1]+1] = -100
            n = batch['labels'][:,1:].ne(-100).sum().item()
            if n:
                valid += n
                batches.append(batch)
            if valid == a.tokens-tokens:
                break
        optimizer.zero_grad(set_to_none=True)
        if model.adapter is not None:
            model.adapter.measure = step % 10 == 0
        total_loss, hits, input_count = 0., 0, 0
        # Weight every target equally, including variable-length examples and final partial steps.
        for batch in batches:
            batch = {k:v.to(a.device) for k,v in batch.items()}
            logits = model(**batch)
            value = loss_sum(logits, batch['labels'])
            (value / valid).backward()
            total_loss += value.detach().item()
            hits += (batch['memory_ids'].ne(0) & batch['attention_mask'].bool()).sum().item()
            input_count += batch['attention_mask'].sum().item()
            del logits, value
        grad_norm = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.)
        if not math.isfinite(total_loss) or not torch.isfinite(grad_norm):
            raise RuntimeError('Non-finite loss/gradient; no invalid optimizer step saved')
        fraction = (tokens+.5*valid)/a.tokens
        scale = min(1., fraction/.05) if fraction <= .05 else .5*(1+math.cos(math.pi*(fraction-.05)/.95))
        for pg in optimizer.param_groups:
            pg['lr'] = pg['initial_lr'] * scale
        optimizer.step()
        tokens += valid
        step += 1
        for threshold in sorted(set(a.snapshot_tokens)):
            if tokens-valid < threshold <= tokens:
                save(run/f'tokens-{threshold}.pt')
        metrics = dict(step=step, tokens=tokens, loss=total_loss/valid, grad_norm=float(grad_norm),
                       hit_rate=hits/input_count, tokens_per_second=(tokens-initial_tokens)/(time.monotonic()-started))
        if model.adapter is not None:
            metrics.update(alpha=float(model.adapter.alpha.detach()), gate_bias=float(model.adapter.bias.detach()))
            if model.adapter.measure:
                metrics.update(model.adapter.measurements)
                metrics.update({f'{name}_grad_norm': float(param.grad.norm()) for name,param in model.adapter.named_parameters()
                                if param.grad is not None})
        if step % a.eval_every == 0 or tokens == a.tokens:
            metrics['validation_loss'] = validation(model,val,tokenizer.pad_token_id or tokenizer.eos_token_id,a.device,
                                                     limit=1_000_000 if tokens == a.tokens else 100_000)
            if metrics['validation_loss'] < best:
                best = metrics['validation_loss']
                save(run/'best.pt')
        log.write(json.dumps(metrics)+'\n')
        if step == 1 or step % 10 == 0 or tokens == a.tokens:
            print(json.dumps(metrics), flush=True)
        if step % a.save_every == 0 or tokens == a.tokens:
            save(run/'last.pt')
        if a.stop_after_steps and step >= a.stop_after_steps and tokens < a.tokens:
            save(run/'last.pt')
            log.close()
            print('Debug interruption saved; resume with the same token budget.',flush=True)
            return
    log.close()
    (run/'complete.json').write_text(json.dumps(dict(step=step,tokens=tokens,best_validation_loss=best)))


if __name__ == '__main__':
    main()
