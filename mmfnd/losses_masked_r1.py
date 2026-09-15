import torch
from torch.nn import functional as F
from mmfnd.relation_supervision import soft_edl_loss, relation_targets
from mmfnd.evidence_utility import contribution_targets


def masked_r1_loss(output, labels, q, scale, cfg, epoch):
    available,pairs = output['evidence_available'],output['pair_available']
    rel_target = relation_targets(q[:,[1,2,4,8],0])
    effects = contribution_targets(q,labels,available,float(cfg.get('epsilon',1e-7)))
    target = effects['delta' if cfg.get('utility_target','shapley') == 'signed_loo' else 'phi']/scale
    cls = F.cross_entropy(output['logits'].float(),labels)
    beta = float(cfg.get('beta_edl',.01))*min(1.,max(0.,(epoch-1)/max(1,int(cfg.get('edl_warmup_epochs',3)))))
    if 'relation_alpha' in output:
        relation = soft_edl_loss(output['relation_alpha'],rel_target,pairs,beta)
    else:
        losses = -(rel_target*output['relation_logits'].float().log_softmax(-1)).sum(-1)
        relation = (losses*pairs).sum()/pairs.sum().clamp_min(1)
    utility = (F.smooth_l1_loss(output['utility'],target.detach().float(),reduction='none')*available).sum()/available.sum()
    total = cls+float(cfg.get('lambda_relation',.2))*relation+float(cfg.get('lambda_utility',1))*utility
    return total, {k:float(v.detach()) for k,v in dict(classification=cls,relation=relation,utility=utility).items()}
