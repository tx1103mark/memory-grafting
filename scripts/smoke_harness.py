"""Exercise real task loading and harness evaluation using an untrained tiny model.

Any numerical accuracy from this script is meaningless; it is an integration test.
"""
import json
from pathlib import Path
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from graft.model import GraftedLM
from graft.harness import GraftHFLM
from scripts.prepare_harness_tasks import prepare
from tests.test_core import tiny

torch.set_num_threads(2)
root=Path('.')
folder=prepare(root)
backend=Tokenizer(WordLevel({'<eos>':0,'<unk>':1,**{f't{i}':i for i in range(2,60)},
                            **dict(zip('ABCD',range(60,64)))},unk_token='<unk>'))
backend.pre_tokenizer=Whitespace()
tok=PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<eos>',pad_token='<eos>',unk_token='<unk>')
model=GraftHFLM(GraftedLM(tiny()),tok,{},max_length=4096)
result=evaluator.simple_evaluate(model=model,
    tasks=['graft_ceval-valid_accountant','graft_cmmlu_agronomy'],
    task_manager=TaskManager(include_path=str(folder),include_defaults=False),
    num_fewshot=0,limit=1,bootstrap_iters=0,log_samples=True,apply_chat_template=False)
assert all(len(v)==1 for v in result['samples'].values())
Path('logs/harness_smoke.json').write_text(json.dumps({
    'passed':True,'synthetic_untrained_model':True,'benchmark_result':False,
    'tasks':list(result['samples']),'examples_per_task':1},indent=2))
print('PASS harness end-to-end task loading and scoring (synthetic model; not a benchmark result)',flush=True)
