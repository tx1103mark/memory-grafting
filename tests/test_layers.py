import pytest
import torch
from safetensors.torch import save_file
from tests.test_core import tiny
from graft.model import GraftedLM
from scripts.build_memory import load_teacher_prefix

@pytest.mark.parametrize('block',[1,2,3])
def test_teacher_prefix_matches_full_model_intermediate(tmp_path,block):
    model=tiny().eval()
    model.config.save_pretrained(tmp_path)
    state={k:v.contiguous() for k,v in model.state_dict().items() if k!='model.norm.weight'}
    keys=list(state)
    for i,part in enumerate((keys[::2],keys[1::2]),1):
        save_file({k:state[k] for k in part},str(tmp_path/f'model-0000{i}-of-00005.safetensors'))
    prefix=load_teacher_prefix(tmp_path,block,'cpu',torch.float32).eval()
    x=torch.tensor([[1,2,3,4]])
    with torch.no_grad():
        expected=model(x,output_hidden_states=True,use_cache=False).hidden_states[block]
        actual=prefix(x,use_cache=False).last_hidden_state
    torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-5)

@pytest.mark.parametrize('layer',[0,1,3])
def test_injection_occurs_once_after_requested_block(layer):
    model=GraftedLM(tiny(),torch.randn(2,16),layer=layer)
    events=[]
    for i,block in enumerate(model.base.model.layers):
        block.register_forward_hook(lambda m,a,o,i=i: events.append(i))
    model.adapter.register_forward_hook(lambda *args: events.append('memory'))
    with torch.no_grad():
        model(torch.tensor([[1,2,3]]),memory_ids=torch.tensor([[0,1,1]]))
    expected=list(range(4));expected.insert(layer+1,'memory')
    assert events==expected

def test_invalid_injection_layer_fails():
    with pytest.raises(ValueError):
        GraftedLM(tiny(),torch.randn(2,16),layer=4)
