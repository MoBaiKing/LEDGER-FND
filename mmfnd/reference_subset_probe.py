"""Offline-only label-supervised small probe. Its forward never accepts labels."""
import torch
from torch import nn
from torch.nn import functional as F
from mmfnd.evidence_utility import subset_masks


class ReferenceSubsetProbe(nn.Module):
    def __init__(self, dim, width=256):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(4*dim+4, width), nn.GELU(), nn.Linear(width,2))

    def forward(self, z, availability):
        if z.shape[-2] != 4 or availability.shape != z.shape[:-1]:
            raise ValueError('fixed four views and explicit matching availability required')
        masked = self.norm(z) * availability[..., None].to(z.dtype)
        return self.mlp(torch.cat([masked.flatten(-2), availability.to(z.dtype)], -1))

    def all_logits(self, z):
        masks = subset_masks(z.device)
        return self(z[:, None].expand(-1,16,-1,-1), masks[None].expand(len(z),-1,-1))


def subset_ce(logits, labels, availability):
    masks = subset_masks(logits.device)
    legal = (~masks[None] | availability[:,None].bool()).all(-1)
    legal[:,0] = False
    if not legal.any(-1).all():
        raise ValueError('reference fit requires a nonempty view set')
    ce = F.cross_entropy(logits.float().transpose(1,2), labels[:,None].expand(-1,16), reduction='none')
    return ((ce*legal).sum(-1)/legal.sum(-1)).mean()


def smoothed_prior(fit_labels, smoothing=1.0):
    counts = torch.bincount(torch.as_tensor(fit_labels).long(), minlength=2).float()
    return (counts+smoothing)/(counts.sum()+2*smoothing)


@torch.no_grad()
def subset_probabilities(probe, z, prior, temperature=1.0):
    if temperature <= 0:
        raise ValueError('temperature must be positive')
    q = (probe.all_logits(z).float()/temperature).softmax(-1)
    q[:,0] = prior.to(q)
    return q
