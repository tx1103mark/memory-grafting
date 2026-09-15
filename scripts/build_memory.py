import argparse
import hashlib
import json
from pathlib import Path
import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


def load_teacher_prefix(path, blocks, device, dtype):
    config=AutoConfig.from_pretrained(path,local_files_only=True)
    if not 1 <= blocks <= config.num_hidden_layers:
        raise ValueError('Teacher block count out of range')
    config.num_hidden_layers=blocks
    config._attn_implementation='sdpa'
    with init_empty_weights():
        model=AutoModel.from_config(config)
    # The original repository is a CausalLM checkpoint; AutoModel keys omit
    # its `model.` prefix. Loading only matching tensors makes missing suffix
    # shards unnecessary and keeps the teacher definition auditable.
    wanted=set(model.state_dict())
    model.norm=torch.nn.Identity()  # raw output of requested block, without final norm
    loaded=set()
    for shard in ('model-00001-of-00005.safetensors','model-00002-of-00005.safetensors'):
        raw=load_file(str(path/shard),device='cpu')
        state={k.removeprefix('model.'):v for k,v in raw.items()
               if k.startswith('model.') and k.removeprefix('model.') in wanted}
        model.load_state_dict(state,strict=False,assign=True)
        loaded.update(state)
        del raw,state
    required={k for k in wanted if not k.startswith('norm.')}
    if loaded != required:
        raise RuntimeError(f'Teacher prefix tensor mismatch: missing={sorted(required-loaded)[:5]}, extra={sorted(loaded-required)[:5]}')
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError('Teacher prefix contains unmaterialized parameters')
    return model.to(device=device,dtype=dtype)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('.'))
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--device', default='cuda')
    p.add_argument('--teacher-block',type=int,default=12)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--data-dir',type=Path,default=Path('data/processed'))
    p.add_argument('--no-random',action='store_true')
    a = p.parse_args()
    keys_path = a.root/a.data_dir/'keys.json'
    keys = json.loads(keys_path.read_text())
    teacher = a.root/'models/teacher'
    tok = AutoTokenizer.from_pretrained(teacher, local_files_only=True, padding_side='right')
    tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if a.device.startswith('cuda') else torch.float32
    model = load_teacher_prefix(teacher,a.teacher_block,a.device,dtype).eval()
    table = torch.zeros((len(keys)+1, model.config.hidden_size), dtype=torch.bfloat16)
    with torch.inference_mode():
        for start in range(0, len(keys), a.batch_size):
            batch = tok([r['text'] for r in keys[start:start+a.batch_size]], add_special_tokens=False,
                        padding=True, return_tensors='pt').to(a.device)
            output = model(**batch, use_cache=False).last_hidden_state
            value = output[torch.arange(len(output),device=a.device), batch.attention_mask.sum(-1)-1]
            table[start+1:start+1+len(value)] = value.cpu().to(torch.bfloat16)
            if start % (a.batch_size*10) == 0:
                print('Encoded', start, '/', len(keys), flush=True)
    folder = a.output_dir or a.root/'memory'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'table.pt').exists():
        raise FileExistsError(f'Refusing to overwrite memory table: {folder}')
    payload={'teacher':table}
    if not a.no_random:
        g = torch.Generator().manual_seed(20260914)
        real = table[1:].float()
        rand = torch.randn(real.shape, generator=g) * real.std(0).clamp_min(1e-6) + real.mean(0)
        rand *= real.norm(dim=-1,keepdim=True)/rand.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        payload['random']=torch.cat([torch.zeros_like(table[:1]),rand.to(table.dtype)])
    torch.save(payload, folder/'table.pt')
    lock = json.loads((a.root/'assets.lock.json').read_text())
    manifest = dict(teacher=lock['teacher'], teacher_block=a.teacher_block, teacher_prefix_only=True,
                    required_weight_files=lock['teacher']['required_weight_files'],
                    key_sha256=hashlib.sha256(keys_path.read_bytes()).hexdigest(),
                    shape=list(table.shape), random_seed=20260914, dtype='bfloat16', final_norm=False,
                    data_dir=str(a.data_dir),random_included=not a.no_random,
                    random_control=None if a.no_random else 'per-dimension moment initialization followed by exact teacher row-norm matching')
    (folder/'manifest.json').write_text(json.dumps(manifest, indent=2))
    print('Memory saved', manifest['shape'], flush=True)


if __name__ == '__main__':
    main()
