"""Task-reference relations, ordered [A,M,C]; not human semantic labels."""
import torch
from torch import nn
from torch.nn import functional as F
from mmfnd.latent_evidence_deliberation import PAIR_EVIDENCE_INDICES as PAIRS


def relation_targets(single_fake):
    r = 2 * single_fake.detach().float() - 1
    product = torch.stack([r[..., i] * r[..., j] for i, j in PAIRS], -1)
    return torch.stack([product.clamp_min(0), 1 - product.abs(), (-product).clamp_min(0)], -1)


def dirichlet_statistics(alpha):
    alpha = alpha.float()
    strength = alpha.sum(-1)
    p = alpha / strength.unsqueeze(-1)
    u = 3 / strength
    b = (alpha - 1) / strength.unsqueeze(-1)
    return dict(relation_alpha=alpha, relation_strength=strength, relation_probs=p,
                relation_vacuity=u, relation_uncertainty=u, relation_belief=b,
                conflict_belief=b[..., 2])


class RelationHead(nn.Module):
    def __init__(self, hidden, evidential=True):
        super().__init__()
        self.output = nn.Linear(hidden, 3)
        self.evidential = evidential

    def forward(self, states):
        logits = self.output(states).float()
        if self.evidential:
            return dict(relation_logits=logits, **dirichlet_statistics(F.softplus(logits) + 1))
        return dict(relation_logits=logits, relation_probs=logits.softmax(-1))


def dirichlet_kl_uniform(alpha):
    a = alpha.float()
    s = a.sum(-1)
    k = a.shape[-1]
    return (torch.lgamma(s) - torch.lgamma(a).sum(-1) - torch.lgamma(a.new_tensor(float(k)))
            + ((a - 1) * (torch.digamma(a) - torch.digamma(s).unsqueeze(-1))).sum(-1))


def soft_edl_loss(alpha, target, valid, beta=0.01):
    """R1 soft-label weighted extension of supervised EDL, in FP32."""
    with torch.autocast(device_type=alpha.device.type, enabled=False):
        a, t = alpha.float(), target.detach().float()
        eye = torch.eye(3, device=a.device)
        # [..., target class, concentration component]: reset chosen class to 1.
        adjusted = a.unsqueeze(-2) * (1 - eye) + eye
        terms = (torch.digamma(a.sum(-1)).unsqueeze(-1) - torch.digamma(a)
                 + float(beta) * dirichlet_kl_uniform(adjusted))
        losses = (t * terms).sum(-1)
        return (losses * valid).sum() / valid.sum().clamp_min(1)
