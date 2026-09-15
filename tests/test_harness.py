import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast
from lm_eval.models.huggingface import HFLM
from graft.harness import GraftHFLM
from graft.model import GraftedLM
from tests.test_core import tiny


def test_harness_likelihood_parity_multitoken_padding():
    backend=Tokenizer(WordLevel({'<eos>':0,'<unk>':1,**{f't{i}':i for i in range(2,64)}},unk_token='<unk>'))
    tok=PreTrainedTokenizerFast(tokenizer_object=backend,eos_token='<eos>',pad_token='<eos>',unk_token='<unk>')
    base=tiny()
    standard=HFLM(pretrained=base,tokenizer=tok,backend='causal',batch_size=2,add_bos_token=False,max_length=32)
    graft=GraftHFLM(GraftedLM(base),tok,{},batch_size=2,max_length=32)
    requests=[(('a','b'),[2,3,4],[5]),(('x','y'),[2,8],[9,10]),(('z','q'),[2,3,4],[6])]
    expected=standard._loglikelihood_tokens(requests,disable_tqdm=True)
    actual=graft._loglikelihood_tokens(requests,disable_tqdm=True)
    for a,b in zip(actual,expected):
        assert a[1]==b[1]
        assert abs(a[0]-b[0])<1e-5
