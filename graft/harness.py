"""Reuse harness HFLM tokenization, batching and likelihood scoring unchanged."""
import torch
from lm_eval.models.huggingface import HFLM
from graft.data import lookup


class GraftHFLM(HFLM):
    def __init__(self, grafted, tokenizer, keys, batch_size=1, memory_mode='normal', max_length=4096):
        super().__init__(pretrained=grafted.base, tokenizer=tokenizer, backend='causal',
                         batch_size=batch_size, max_length=max_length, add_bos_token=False,
                         truncation=False, logits_cache=True)
        self.grafted = grafted.eval()
        self.keys = keys
        self.memory_mode = memory_mode

    @torch.no_grad()
    def _model_call(self, inps, attn_mask=None, labels=None):
        if labels is not None:
            raise NotImplementedError('Only causal likelihood evaluation is supported')
        rows = [lookup(row, self.keys) for row in inps.cpu().tolist()]
        memory_ids = torch.tensor(rows, device=inps.device, dtype=torch.long)
        disabled = self.memory_mode == 'zero' and self.grafted.adapter is not None
        if disabled:
            self.grafted.adapter.disabled = True
        try:
            return self.grafted(inps, attention_mask=attn_mask, memory_ids=memory_ids)
        finally:
            if disabled:
                self.grafted.adapter.disabled = False

    def _loglikelihood_tokens(self, requests, *args, **kwargs):
        for _, context, continuation in requests:
            if len(context) + len(continuation) > self.max_length:
                raise ValueError('Evaluation example exceeds max_length; do not silently truncate choices')
        return super()._loglikelihood_tokens(requests, *args, **kwargs)

    def generate_until(self, requests, **kwargs):
        raise NotImplementedError('Memory-aware cached generation is outside this experiment')
