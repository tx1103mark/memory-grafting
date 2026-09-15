import json
import hashlib
import subprocess
import sys
from pathlib import Path
import torch
import pytest
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from tests.test_core import tiny


@pytest.mark.parametrize('student_block',[1,3])
def test_real_training_entrypoint_and_exact_resume(tmp_path,student_block):
    root=tmp_path
    student=root/'models/student'
    student.mkdir(parents=True)
    tiny().save_pretrained(student)
    backend=Tokenizer(WordLevel({'<eos>':0,'<unk>':1,**{f't{i}':i for i in range(2,64)}},unk_token='<unk>'))
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<eos>',pad_token='<eos>',unk_token='<unk>')
    tok.save_pretrained(student)
    data=root/'data/processed'
    data.mkdir(parents=True)
    rows=[{'input_ids':list(range(i,i+9)),'labels':[-100]+list(range(i+1,i+9)),
           'memory_ids':[0,0]+[1]*7} for i in range(2,8)]
    for split in ('train','validation'):
        Dataset.from_list(rows).save_to_disk(str(data/split))
    for path in [data/'keys.json',data/'manifest.json',root/'assets.lock.json']:
        path.write_text('{}')
    memory=root/'memory'
    memory.mkdir()
    table=torch.randn(2,16);table[0]=0
    torch.save({'teacher':table,'random':table.clone()},memory/'table.pt')
    (memory/'manifest.json').write_text(json.dumps({'key_sha256':hashlib.sha256((data/'keys.json').read_bytes()).hexdigest(),'teacher_block':2}))
    command=[sys.executable,'-m','scripts.train','--root',str(root),'--group','G','--device','cpu',
             '--tokens','48','--micro-batch','1','--accumulation','2','--student-block',str(student_block),
             '--snapshot-tokens','15','31']
    def run(args):
        subprocess.run(command+args,check=True,capture_output=True,text=True)
    run(['--run-name','full'])
    run(['--run-name','resumed','--stop-after-steps','1'])
    interrupted=root/'runs/resumed/last.pt'
    assert not (root/'runs/resumed/complete.json').exists()
    run(['--run-name','resumed','--resume',str(interrupted)])
    full=torch.load(root/'runs/full/last.pt',weights_only=False)
    resumed=torch.load(interrupted,weights_only=False)
    assert full['tokens']==resumed['tokens']==48
    assert full['step']==resumed['step']==3
    assert full['manifest'].get('student_block',3)==student_block
    for threshold,actual in ((15,16),(31,32)):
        snapshot=torch.load(root/f'runs/resumed/tokens-{threshold}.pt',weights_only=False)
        assert snapshot['tokens']==actual
        reference=torch.load(root/f'runs/full/tokens-{threshold}.pt',weights_only=False)
        for name,value in reference['model'].items():
            torch.testing.assert_close(value,snapshot['model'][name],atol=0,rtol=0)
    for name,value in full['model'].items():
        torch.testing.assert_close(value,resumed['model'][name],atol=0,rtol=0)
    first=torch.load(root/'runs/full/best.pt',weights_only=False)
    assert abs(first['model']['adapter.alpha'].item()-torch.tensor(.001).item())>1e-8
