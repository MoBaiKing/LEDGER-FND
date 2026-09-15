"""Conditional-on-fixed-Z controls; never randomize a live shared text encoder."""
import copy
import numpy as np
import torch
from torch import nn
from mmfnd.revision_masked_r1 import MaskedR1Deliberation
from mmfnd.r1_cache import digest


def fixed_role_permutations(sample_ids,seed):
    if len(set(sample_ids))!=len(sample_ids) or len(sample_ids)<2:raise ValueError('at least two unique dataset-wide IDs required')
    order=sorted(map(str,sample_ids));rng=np.random.default_rng(seed);result={}
    for role in range(4):
        # A seeded cyclic shift of a randomized list is a derangement, even B=1 at inference.
        shuffled=list(rng.permutation(order));shift=int(rng.integers(1,len(order)))
        result[str(role)]={sid:shuffled[(i+shift)%len(order)] for i,sid in enumerate(shuffled)}
    return dict(seed=seed,id_hash=digest(order),roles=result,kind='eval_only_sensitivity')


def shuffle_cached_views(z,sample_ids,permutations):
    index={str(sid):i for i,sid in enumerate(sample_ids)}
    if permutations['id_hash']!=digest(sorted(index)):raise ValueError('shuffle sample universe mismatch')
    return torch.stack([z[[index[permutations['roles'][str(role)][str(sid)]] for sid in sample_ids],role] for role in range(4)],1)


class FrozenJudgeBackbone(nn.Module):
    def __init__(self,qwen,random_init=False):
        super().__init__()
        self.config=copy.deepcopy(qwen.config)
        self.layers=nn.ModuleList(copy.deepcopy(list(qwen.layers)[-2:]))
        self.rotary_emb=copy.deepcopy(qwen.rotary_emb);self.norm=copy.deepcopy(qwen.norm)
        if random_init:
            self.apply(qwen._init_weights)
        self.config.num_hidden_layers=2


class FrozenInputJudge(nn.Module):
    def __init__(self,qwen,dim,cfg,random_init=False):
        super().__init__()
        self.qwen=FrozenJudgeBackbone(qwen,random_init)
        self.judge=MaskedR1Deliberation(dim,qwen.config.hidden_size,cfg)
        self.classifier=nn.Sequential(nn.LayerNorm(dim),nn.Dropout(.2),nn.Linear(dim,2))

    def forward(self,z,available):
        out=self.judge(z.detach(),self.qwen,available);out['logits']=self.classifier(out['fused']);return out
