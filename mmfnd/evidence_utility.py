"""Signed reference-game effects and one supervised utility scorer."""
import math
import warnings
import torch
from torch import nn


def subset_masks(device=None):
    return ((torch.arange(16, device=device)[:, None] >> torch.arange(4, device=device)) & 1).bool()


def contribution_targets(q, labels, available=None, epsilon=1e-7):
    """q[B,16,2], bit 0=T,1=V,2=E,3=X. Exact available-player game."""
    q = q.detach().double()
    if q.ndim != 3 or q.shape[1:] != (16, 2):
        raise ValueError('expected [B,16,2] probabilities')
    if available is None:
        available = torch.ones((len(q), 4), dtype=torch.bool, device=q.device)
    available = available.bool()
    loss = -q.gather(-1, labels[:, None, None].expand(-1, 16, 1)).squeeze(-1).clamp_min(epsilon).log()
    codes = (available.long() * (2 ** torch.arange(4, device=q.device))).sum(-1)
    phi = torch.zeros((len(q), 4), dtype=q.dtype, device=q.device)
    delta, unsigned = torch.zeros_like(phi), torch.zeros_like(phi)
    # Only 16 games and 4 players; vectorized over all examples in each game.
    for code in range(1, 16):
        rows = codes == code
        players = [i for i in range(4) if code & (1 << i)]
        n = len(players)
        for i in players:
            without = code ^ (1 << i)
            delta[rows, i] = loss[rows, without] - loss[rows, code]
            unsigned[rows, i] = 0.5 * (q[rows, without] - q[rows, code]).abs().sum(-1)
            for s in range(16):
                if s & ~without:
                    continue
                k = s.bit_count()
                weight = math.factorial(k) * math.factorial(n-k-1) / math.factorial(n)
                phi[rows, i] += weight * (loss[rows, s] - loss[rows, s | (1 << i)])
    return dict(phi=phi, delta=delta, unsigned_tv=unsigned, losses=loss)


def fit_utility_scale(phi, available, floor=1e-6):
    scale = phi[available.bool()].double().square().mean().sqrt().item()
    if not math.isfinite(scale):
        raise ValueError('invalid OOF utility scale')
    if scale < floor:
        warnings.warn('OOF utility targets degenerate: almost all zero', RuntimeWarning)
    return max(scale, floor)


def masked_weights(utility, available, tau=1.0):
    if tau <= 0 or not torch.isfinite(torch.tensor(tau)):
        raise ValueError('tau must be finite and positive')
    if not available.bool().any(-1).all():
        raise ValueError('main model requires at least one available view')
    return (utility.float() / tau).masked_fill(~available.bool(), float('-inf')).softmax(-1)


class EvidenceUtilityHead(nn.Module):
    def __init__(self, hidden, width=256, role_dim=16, relation_dim=4, use_global=True):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.global_norm = nn.LayerNorm(hidden) if use_global else None
        self.role = nn.Embedding(4, role_dim)
        self.output = nn.Sequential(nn.Linear(2*hidden+relation_dim+1+role_dim, width),
                                    nn.GELU(), nn.Linear(width, 1))
        nn.init.normal_(self.output[-1].weight, std=1e-4)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, evidence_states, global_state, relations, fraction, use_global=True):
        b = evidence_states.shape[0]
        context = self.global_norm(global_state) if use_global else torch.zeros_like(global_state)
        roles = self.role(torch.arange(4, device=evidence_states.device))[None].expand(b, -1, -1)
        inputs = torch.cat([self.norm(evidence_states), context[:, None].expand(-1,4,-1),
                            relations.detach().to(evidence_states.dtype), fraction[..., None].to(evidence_states.dtype), roles.to(evidence_states.dtype)], -1)
        return self.output(inputs).squeeze(-1).float()
