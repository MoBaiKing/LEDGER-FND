"""Deterministic clean+one-augmentation pool; no global RNG perturbation."""
import hashlib
import random
import numpy as np
import torch
from mmfnd.data import CUTEFNDMultimodalDataset


def augmentation_seed(seed, sample_id, augmentation_id):
    key = f'{seed}:{sample_id}:{augmentation_id}'
    return int(hashlib.sha256(key.encode()).hexdigest()[:8],16)


class ReplayDataset(CUTEFNDMultimodalDataset):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        cfg = args[3] if len(args)>3 else kwargs['config']
        self.pool_seed = int(cfg['replay_seed'])
        self.pool = self.train and self.image_preprocessing.get('train_augmentation',{}).get('enabled',False)

    def __len__(self): return len(self.records)*(2 if self.pool else 1)

    def __getitem__(self,index):
        base,variant = divmod(index,2) if self.pool else (index,0)
        augmentation_id = 'aug1' if variant else 'clean'
        state = random.getstate()
        train = self.train
        try:
            random.seed(augmentation_seed(self.pool_seed,self.records[base]['id'],augmentation_id))
            self.train = bool(variant)
            item = super().__getitem__(base)
        finally:
            self.train = train
            random.setstate(state)
        item['augmentation_id'] = augmentation_id
        return item

    def collate_fn(self,batch):
        out = super().collate_fn(batch)
        out['augmentation_ids'] = [x['augmentation_id'] for x in batch]
        out['availability'] = torch.ones((len(batch),4),dtype=torch.bool)
        return out
