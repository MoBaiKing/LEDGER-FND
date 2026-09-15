"""Strict masked R1. No online teacher, IURD or second fusion path."""
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from mmfnd.latent_evidence_deliberation import (
    PAIR_EVIDENCE_INDICES as PAIRS, SharedEvidenceProjector,
    EvidenceRoleEmbedding, LLMGuidedLatentEvidenceDeliberation,
    strict_latent_attention_mask,
)
from mmfnd.relation_supervision import RelationHead
from mmfnd.evidence_utility import EvidenceUtilityHead, masked_weights

VERSION = 'qwen_lora_lgled_masked_r1'
VIEW_KEYS = ('text', 'vision', 'intrinsic_feature', 'interaction')


def availability_mask(available, use_global=True, visibility='strict'):
    pair_valid = torch.stack([available[:,i] & available[:,j] for i,j in PAIRS], -1)
    valid = torch.cat([available, pair_valid], -1)
    if use_global:
        valid = torch.cat([valid, torch.ones_like(valid[:,:1])], -1)
    n = valid.shape[-1]
    if visibility == 'strict':
        allowed = ~strict_latent_attention_mask(n, available.device)
    elif visibility == 'causal':
        allowed = torch.ones((n,n), device=available.device, dtype=torch.bool).tril()
    else:
        raise ValueError('visibility must be strict or causal')
    allowed = allowed[None] & valid[:,None,:] & valid[:,:,None]
    # Invalid queries retain ONLY self; valid queries cannot read invalid keys.
    allowed |= torch.eye(n, device=available.device, dtype=torch.bool)[None]
    return allowed, pair_valid


class MaskedR1Deliberation(nn.Module):
    selected_qwen_layers = LLMGuidedLatentEvidenceDeliberation.selected_qwen_layers
    runtime = LLMGuidedLatentEvidenceDeliberation.runtime

    def __init__(self, dim, hidden, cfg):
        super().__init__()
        self.cfg = dict(cfg)
        self.latent_judge_num_layers = int(cfg.get('latent_judge_num_layers',2))
        self.use_global = cfg.get('use_global',True) and cfg.get('fusion','utility') != 'uniform'
        self.judge_type = cfg.get('judge_type','qwen_latent')
        self.shared_projector = SharedEvidenceProjector(dim,hidden)
        self.role_embedding = EvidenceRoleEmbedding(hidden)
        self.role_norm = nn.LayerNorm(hidden)
        self.judge_tokens = nn.Embedding(7 if self.use_global else 6,hidden)
        nn.init.normal_(self.judge_tokens.weight,std=.02)
        self.relation_head = RelationHead(hidden,cfg.get('evidential',True))
        self.relation_dim = (3 + int(cfg.get('explicit_vacuity',True) and cfg.get('evidential',True))
                             if cfg.get('explicit_relations',True) else 0)
        self.utility_head = (EvidenceUtilityHead(hidden,int(cfg.get('head_hidden_dim',256)),
                                                relation_dim=self.relation_dim,use_global=self.use_global)
                             if cfg.get('fusion','utility') != 'uniform' else None)
        if self.judge_type == 'mlp':
            self.pair_mlp = nn.Sequential(nn.Linear(2*hidden,hidden),nn.GELU(),nn.Linear(hidden,hidden))
            self.global_mlp = (nn.Sequential(nn.Linear(4*hidden+4,hidden),nn.GELU(),nn.Linear(hidden,hidden))
                               if self.use_global else None)
            # MLP control has no slot tokens, so no unused slot parameters.
            self.judge_tokens = None
        elif self.judge_type != 'qwen_latent':
            raise ValueError('unsupported judge_type')
        membership = torch.zeros(4,6)
        for p,(i,j) in enumerate(PAIRS): membership[i,p]=membership[j,p]=1
        self.register_buffer('membership',membership,persistent=False)

    def run_layers(self, sequence, qwen, available, permutation=None):
        if qwen.config._attn_implementation not in ('eager','sdpa'):
            raise ValueError('custom strict mask requires Qwen eager or SDPA')
        allowed,_ = availability_mask(available,self.use_global,self.cfg.get('visibility','strict'))
        b,n = sequence.shape[:2]
        positions = torch.arange(n,device=sequence.device)
        if permutation is not None:
            if sorted(permutation.tolist()) != list(range(n)):
                raise ValueError('invalid token permutation')
            sequence = sequence[:,permutation]
            allowed = allowed[:,permutation][:,:,permutation]
            positions = positions[permutation]
        h = sequence.to(next(qwen.parameters()).dtype)
        position_ids = positions[None].expand(b,-1)
        mask = torch.zeros((b,1,n,n),device=h.device,dtype=h.dtype).masked_fill(~allowed[:,None],float('-inf'))
        rotary = qwen.rotary_emb(h,position_ids)
        for layer in self.selected_qwen_layers(qwen):
            def run(x, layer=layer):
                return layer(x,attention_mask=mask,position_ids=position_ids,past_key_value=None,
                             output_attentions=False,use_cache=False,cache_position=positions,
                             position_embeddings=rotary)[0]
            h = (checkpoint(run,h,use_reentrant=False) if self.training and self.cfg.get('gradient_checkpointing',True)
                 else run(h))
        h = qwen.norm(h)
        return h[:,torch.argsort(permutation)] if permutation is not None else h

    def forward(self,z,qwen,available=None,permutation=None):
        if z.ndim != 3 or z.shape[1] != 4:
            raise ValueError('R1 expects [B,4,D]')
        if available is None: available = torch.ones(z.shape[:2],device=z.device,dtype=torch.bool)
        available = available.bool()
        if available.shape != z.shape[:2] or not available.any(-1).all():
            raise ValueError('main input must have at least one available view per sample')
        clean = z.masked_fill(~available[...,None],0)
        h = self.role_norm(self.shared_projector(clean)+self.role_embedding(len(z)))
        _,pair_valid = availability_mask(available,self.use_global,self.cfg.get('visibility','strict'))
        if self.judge_type == 'mlp':
            pair = self.pair_mlp(torch.stack([torch.cat([h[:,i],h[:,j]],-1) for i,j in PAIRS],1))
            g = (self.global_mlp(torch.cat([(h*available[...,None]).flatten(1),available.to(h)],-1))
                 if self.use_global else torch.zeros_like(h[:,0]))
            latent = torch.cat([h,pair,*([g[:,None]] if self.use_global else [])],1)
        else:
            tokens = self.judge_tokens.weight[None].expand(len(z),-1,-1).to(h)
            latent = self.run_layers(torch.cat([h,tokens],1),qwen,available,permutation)
            h,pair = latent[:,:4],latent[:,4:10]
            g = latent[:,10] if self.use_global else torch.zeros_like(h[:,0])
        relation = self.relation_head(pair)
        membership = self.membership.float()[None]*pair_valid[:,None]
        count = membership.sum(-1)
        features = relation['relation_probs']
        if self.relation_dim == 4:
            features = torch.cat([features,relation['relation_vacuity'][...,None]],-1)
        with torch.autocast(device_type=z.device.type, enabled=False):
            means = torch.einsum('bip,bpc->bic',membership.float(),features.float())/count.clamp_min(1)[...,None]
        means = means[...,:self.relation_dim]
        utility = (self.utility_head(h,g,means,count/3,self.use_global)
                   if self.utility_head is not None else z.new_zeros(z.shape[:2]))
        weights = masked_weights(utility,available,float(self.cfg.get('tau',1)))
        routed = weights if self.cfg.get('allow_task_grad_into_routing',False) else weights.detach()
        fused = (routed[...,None]*clean).sum(1)
        return dict(**relation,utility=utility,final_evidence_weights=weights,fused=fused,
                    evidence_available=available,pair_available=pair_valid,latent_output=latent,
                    evidence_features=z,architecture_version=VERSION)


class MaskedR1Model(nn.Module):
    architecture_version = VERSION

    def __init__(self,config,encoder=None):
        super().__init__()
        from mmfnd.model import MultimodalIntrinsicEvidenceEncoder
        self.runtime_config = config
        cfg = config['model']
        if cfg['architecture_version'] != VERSION: raise ValueError('wrong R1 architecture')
        d = int(cfg['hidden_dim'])
        self.encoder = encoder if encoder is not None else MultimodalIntrinsicEvidenceEncoder(cfg,d,int(cfg['num_heads']))
        self.encoder.general_last_valid_index = True
        self.lgled = MaskedR1Deliberation(d,self.encoder.shared_qwen_model().config.hidden_size,cfg.get('r1',{}))
        self.classifier = nn.Sequential(nn.LayerNorm(d),nn.Dropout(cfg['dropout']),nn.Linear(d,2))

    def encode_views(self,batch):
        # Only explicit tensor inputs cross prediction boundary; labels/IDs never do.
        encoded = self.encoder({k:batch[k] for k in ('text_input_ids','text_attention_mask','pixel_values','image_owner')})
        return torch.stack([encoded[k] for k in VIEW_KEYS],1)

    def predict_views(self,z,latent_view_mask=None):
        output = self.lgled(z,self.encoder.shared_qwen_model(),latent_view_mask)
        output['logits'] = self.classifier(output['fused'])
        return output

    def forward(self,batch,latent_view_mask=None):
        return self.predict_views(self.encode_views(batch),latent_view_mask)

    def raw_modality_intervention(self,batch,transform):
        """Transform raw token/image tensors and recompute ALL dependent E/X."""
        changed = transform({k:batch[k].clone() for k in ('text_input_ids','text_attention_mask','pixel_values','image_owner')})
        return self.forward(changed)


class EncoderFusionBaseline(MaskedR1Model):
    """Same frontend/head, mean or ordinary single-path task-trained scorer."""
    def __init__(self,config,encoder=None):
        nn.Module.__init__(self)
        from mmfnd.model import MultimodalIntrinsicEvidenceEncoder
        self.runtime_config=config;c=config['model'];d=int(c['hidden_dim'])
        self.encoder=encoder if encoder is not None else MultimodalIntrinsicEvidenceEncoder(c,d,int(c['num_heads']))
        self.encoder.general_last_valid_index=True
        self.classifier=nn.Sequential(nn.LayerNorm(d),nn.Dropout(c['dropout']),nn.Linear(d,2))
        self.scorer=(nn.Sequential(nn.LayerNorm(d),nn.Linear(d,256),nn.GELU(),nn.Linear(256,1))
                     if c['r1']['baseline']=='scorer' else None)

    def predict_views(self,z,latent_view_mask=None):
        available=(torch.ones(z.shape[:2],device=z.device,dtype=torch.bool) if latent_view_mask is None else latent_view_mask)
        clean=z.masked_fill(~available[...,None],0)
        u=self.scorer(clean).squeeze(-1) if self.scorer is not None else z.new_zeros(z.shape[:2])
        w=masked_weights(u,available)
        fused=(w[...,None]*clean).sum(1)
        _,pairs=availability_mask(available)
        return dict(architecture_version=VERSION,logits=self.classifier(fused),utility=u,final_evidence_weights=w,
                    evidence_available=available,pair_available=pairs,evidence_features=z,
                    relation_probs=z.new_full((len(z),6,3),1/3),relation_logits=z.new_zeros((len(z),6,3)))
