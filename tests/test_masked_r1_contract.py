"""26 acceptance contracts on CPU synthetic fixtures; GPU items explicitly separate."""
import copy
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen2Config,Qwen2Model
from mmfnd.relation_supervision import relation_targets,dirichlet_statistics,dirichlet_kl_uniform,soft_edl_loss
from mmfnd.evidence_utility import subset_masks,contribution_targets,masked_weights
from mmfnd.reference_subset_probe import ReferenceSubsetProbe,subset_probabilities,smoothed_prior,subset_ce
from mmfnd.revision_masked_r1 import MaskedR1Deliberation,MaskedR1Model,VERSION,availability_mask,PAIRS
from mmfnd.r1_cache import grouped_folds,validate_fold,TargetCache,file_sha
from mmfnd.evaluation import compute_classification_metrics,find_best_macro_f1_threshold
from mmfnd.evaluation_masked_r1 import deduplicate_rows

torch.set_num_threads(1)


def tiny(backend='eager',dtype=torch.float32,**kwargs):
    torch.manual_seed(73)
    config=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=48,num_hidden_layers=3,
                       num_attention_heads=4,num_key_value_heads=2,attention_dropout=0.,use_cache=False)
    config._attn_implementation=backend
    q=Qwen2Model(config).to(dtype).eval()
    module=MaskedR1Deliberation(16,32,dict(head_hidden_dim=16,gradient_checkpointing=False,**kwargs)).to(dtype).eval()
    return module,q


def test_01_targets_probability_label_contract():
    t=relation_targets(torch.rand(10,4))
    assert (t>=0).all() and (t<=1).all()
    torch.testing.assert_close(t.sum(-1),torch.ones(10,6))
    q=torch.tensor([[[.8,.2]]*16]);effects=contribution_targets(q,torch.tensor([0]))
    assert effects['losses'][0,0]==pytest.approx(-math.log(.8))


def test_02_operational_cases_and_wrong_agreement():
    p=torch.tensor([[1.,1.,0.,.5]])
    t=relation_targets(p)
    torch.testing.assert_close(t[0,0],torch.tensor([1.,0.,0.]))
    torch.testing.assert_close(t[0,1],torch.tensor([0.,0.,1.]))
    torch.testing.assert_close(t[0,2],torch.tensor([0.,1.,0.]))
    # Label could be Real=1: both confident Fake views are wrong, but A remains 1.
    assert 'labels' not in inspect.signature(relation_targets).parameters


def test_03_exchange_changes_loss():
    a=torch.tensor([[[9.,2.,3.]]],requires_grad=True);t=torch.tensor([[[.8,.15,.05]]]);v=torch.ones(1,1,dtype=torch.bool)
    assert not torch.isclose(soft_edl_loss(a,t,v),soft_edl_loss(a[:,:,[1,0,2]],t,v))


def test_04_dirichlet_identities():
    a=F.softplus(torch.randn(5,6,3))+1;s=dirichlet_statistics(a);p=s['relation_probs'];u=s['relation_vacuity'];b=s['relation_belief']
    assert (a>=1).all()
    torch.testing.assert_close(p.sum(-1),torch.ones_like(u))
    torch.testing.assert_close(b.sum(-1)+u,torch.ones_like(u))
    torch.testing.assert_close(p,b+u[...,None]/3)
    torch.testing.assert_close(s['conflict_belief'],p[...,2]-u/3)


def test_05_impossible_vacuity_and_uniform_strong_evidence():
    assert .9>1-2*.8/3
    s=dirichlet_statistics(torch.full((1,3),100.))
    torch.testing.assert_close(s['relation_probs'],torch.full((1,3),1/3))
    assert s['relation_vacuity'].item()==pytest.approx(.01)


def test_06_kl_finite_and_zero_valid():
    a=(torch.rand(4,6,3)*8+1).requires_grad_()
    expected=torch.distributions.kl_divergence(torch.distributions.Dirichlet(a),torch.distributions.Dirichlet(torch.ones_like(a)))
    torch.testing.assert_close(dirichlet_kl_uniform(a),expected,atol=1e-5,rtol=1e-5)
    t=relation_targets(torch.rand(4,4));v=torch.ones(4,6,dtype=torch.bool)
    loss=soft_edl_loss(a,t,v);loss.backward();assert torch.isfinite(a.grad).all()
    zero=soft_edl_loss(a,t,~v);assert zero.item()==0 and zero.requires_grad


def test_07_unsigned_same_signed_opposite():
    q=torch.full((2,16,2),.5);q[:,15]=torch.tensor([.5,.5]);q[0,14]=torch.tensor([.25,.75]);q[1,14]=torch.tensor([.75,.25])
    e=contribution_targets(q,torch.zeros(2,dtype=torch.long))
    assert e['unsigned_tv'][0,0]==e['unsigned_tv'][1,0]
    assert e['delta'][0,0]>0>e['delta'][1,0]


def additive_q(values):
    loss=5-subset_masks().double()@torch.tensor(values,dtype=torch.float64)
    return torch.stack([(-loss).exp(),1-(-loss).exp()],-1)[None]


def test_08_all_subsets_prior_shapley_axioms():
    masks=subset_masks();assert masks.shape==(16,4)
    assert ((masks.long()*(2**torch.arange(4))).sum(-1)==torch.arange(16)).all()
    prior=smoothed_prior([0,0,1]);torch.testing.assert_close(prior,torch.tensor([.6,.4]))
    probe=ReferenceSubsetProbe(16,8);q=subset_probabilities(probe,torch.randn(2,4,16),prior)
    torch.testing.assert_close(q[:,0],prior.expand(2,-1))
    e=contribution_targets(additive_q([.3,.3,-.2,0]),torch.tensor([0]))
    torch.testing.assert_close(e['phi'][0],torch.tensor([.3,.3,-.2,0],dtype=torch.float64))
    torch.testing.assert_close(e['phi'].sum(-1),e['losses'][:,0]-e['losses'][:,15])


def test_09_missing_subgame_and_no_labels():
    a=torch.tensor([[True,False,True,False]])
    e=contribution_targets(additive_q([.3,.3,-.2,0]),torch.tensor([0]),a)
    torch.testing.assert_close(e['phi'][0],torch.tensor([.3,0,-.2,0],dtype=torch.float64))
    assert set(inspect.signature(ReferenceSubsetProbe.forward).parameters)=={'self','z','availability'}
    probe=ReferenceSubsetProbe(16,8).eval();z=torch.randn(1,4,16);baseline=probe(z,a);z[:,~a[0]]=999
    torch.testing.assert_close(probe(z,a),baseline)


def test_10_cross_fit_label_sets_and_variants():
    records=[dict(id=str(i),text=f'news {i//2}',label=i%2) for i in range(60)]
    folds=grouped_folds(records,199)['folds'];mapping={}
    for fold in folds:
        validate_fold(fold)
        for sid in fold['target_ids']:mapping[sid]=fold['fold']
    assert len(mapping)==60
    assert all(mapping[str(i)]==mapping[str(i+1)] for i in range(0,60,2))
    bad=copy.deepcopy(folds[0]);bad['fit_ids']+=bad['target_ids'][:1]
    with pytest.raises(ValueError):validate_fold(bad)


def test_11_replay_and_fingerprint_failure(tmp_path):
    from PIL import Image
    from mmfnd.r1_data import ReplayDataset
    Image.fromarray(np.random.default_rng(4).integers(0,256,(20,20,3),dtype=np.uint8)).save(tmp_path/'img.png')
    manifest=tmp_path/'train.jsonl';manifest.write_text(json.dumps(dict(id='a',dataset='fixture',text='news',label=0,images=['img.png'])))
    cfg=dict(max_images=2,max_text_length=8,dataset_name='fixture',replay_seed=18,
             image_preprocessing={'train_augmentation':{'enabled':True,'random_crop_scale':[.5,.8],'brightness':.2}})
    data=ReplayDataset(manifest,tmp_path,None,cfg,train=True)
    assert data[1]['augmentation_id']=='aug1'
    assert data[1]['images'][0].tobytes()==data[1]['images'][0].tobytes()
    path=tmp_path/'targets.pt';fp={'preprocess':'a','model':'b','split':'c'}
    torch.save(dict(fingerprint=fp,scale=1.,rows=[]),path)
    path.with_suffix('.json').write_text(json.dumps({'payload_sha256':file_sha(path)}))
    for key in fp:
        changed=dict(fp);changed[key]='wrong'
        with pytest.raises(ValueError):TargetCache(path,changed)


@pytest.mark.parametrize('backend,dtype',[('eager',torch.float32),('sdpa',torch.float32),('sdpa',torch.bfloat16)])
def test_12_actual_qwen_nonendpoint_jacobian(backend,dtype):
    m,q=tiny(backend,dtype);z=torch.randn(2,4,16,dtype=dtype,requires_grad=True);out=m(z,q)
    for pair,(i,j) in enumerate(PAIRS):
        grad,=torch.autograd.grad(out['relation_probs'][:,pair,0].sum(),z,retain_graph=True)
        other=[k for k in range(4) if k not in (i,j)]
        assert torch.count_nonzero(grad[:,other])==0
        changed=z.detach().clone();changed[:,other]=torch.randn_like(changed[:,other])*7
        torch.testing.assert_close(m(changed,q)['relation_probs'][:,pair],out['relation_probs'][:,pair],atol=0,rtol=0)


def test_13_endpoint_and_missing_global_two_layers():
    m,q=tiny();z=torch.randn(2,4,16,requires_grad=True);o=m(z,q)
    altered=z.detach().clone();altered[:,0]=torch.randn_like(altered[:,0])*5
    assert not torch.allclose(o['latent_output'][:,4],m(altered,q)['latent_output'][:,4])
    a=torch.tensor([[True,True,False,False]]*2);base=m(z,q,a)
    altered=z.detach().clone();altered[:,2:]=999
    torch.testing.assert_close(base['latent_output'][:,10],m(altered,q,a)['latent_output'][:,10])
    allowed,_=availability_mask(a)
    assert not allowed[:,10,2:4].any() and not allowed[:,10,5:10].any()


@pytest.mark.parametrize('global_on',[True,False])
def test_14_shapes_single_empty_no_nan(global_on):
    m,q=tiny(use_global=global_on)
    for b in (1,3):
        z=torch.randn(b,4,16);a=torch.tensor([[1,0,0,0]]*b).bool();o=m(z,q,a)
        assert torch.isfinite(o['fused']).all() and not o['pair_available'].any()
        assert o['latent_output'].shape[1]==(11 if global_on else 10)
        with pytest.raises(ValueError):m(z,q,torch.zeros_like(a))
    with pytest.raises(ValueError):m(torch.randn(2,3,16),q)


def test_15_controlled_permutation():
    m,q=tiny();z=torch.randn(2,4,16);p=torch.randperm(11)
    torch.testing.assert_close(m(z,q)['latent_output'],m(z,q,permutation=p)['latent_output'],atol=2e-6,rtol=2e-6)


def test_16_gradient_routes():
    m,q=tiny();z=torch.randn(3,4,16,requires_grad=True);out=m(z,q);cls=nn.Linear(16,2)
    F.cross_entropy(cls(out['fused']),torch.tensor([0,1,0])).backward(retain_graph=True)
    assert all(p.grad is None for p in m.utility_head.parameters())
    assert all(p.grad is None for p in m.relation_head.parameters())
    F.smooth_l1_loss(out['utility'],torch.randn(3,4)).backward(retain_graph=True)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.utility_head.parameters())
    assert all(p.grad is None for p in m.relation_head.parameters())
    soft_edl_loss(out['relation_alpha'],relation_targets(torch.rand(3,4)),out['pair_available']).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.relation_head.parameters())


def test_17_multistep_lora_parameter_sources():
    from peft import get_peft_model,LoraConfig,TaskType
    m,q=tiny();q=get_peft_model(q,LoraConfig(task_type=TaskType.FEATURE_EXTRACTION,r=2,lora_alpha=4,target_modules=['q_proj','v_proj'])).get_base_model()
    params=list(m.parameters())+[p for p in q.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.01);seen=set()
    for step in range(3):
        # Include text path: all-layer LoRA is shared but only final two judge layers execute.
        text=q(input_ids=torch.randint(0,32,(3,8)),attention_mask=torch.ones(3,8)).last_hidden_state
        z=text[:,:4,:16];o=m(z,q)
        loss=F.smooth_l1_loss(o['utility'],torch.randn(3,4))+soft_edl_loss(o['relation_alpha'],relation_targets(torch.rand(3,4)),o['pair_available'])
        opt.zero_grad();loss.backward()
        seen.update(id(p) for p in params if p.grad is not None and p.grad.abs().sum()>0)
        opt.step()
    missing=[n for n,p in list(m.named_parameters())+list(q.named_parameters()) if p.requires_grad and id(p) not in seen]
    assert not missing,missing


def test_18_no_teacher_old_parameters_or_duplicates():
    m,q=tiny();names=[n for n,p in m.named_parameters()]
    assert not any(any(old in n for old in ('iurd','confidence_head','minority_head','global_judge_head','direct_fusion','adjudication_head')) for n in names)
    assert not any(p is qp for p in m.parameters() for qp in q.parameters())
    assert len({id(p) for p in m.parameters()})==len(list(m.parameters()))
    teacher=ReferenceSubsetProbe(16,8);z=torch.randn(2,4,16)
    qref=subset_probabilities(teacher,z,torch.tensor([.5,.5]));targets=contribution_targets(qref,torch.tensor([0,1]))
    m(z,q)['utility'].sum().backward()
    assert not targets['phi'].requires_grad and all(p.grad is None for p in teacher.parameters())


def test_19_weights_simplex_monotonic_missing():
    u=torch.tensor([[1.,2.,-3.,9.]]);a=torch.tensor([[1,1,1,0]]).bool();w=masked_weights(u,a)
    assert w[0,1]>w[0,0]>w[0,2]>w[0,3] and w[0,3]==0
    torch.testing.assert_close(w.sum(-1),torch.ones(1))


def test_20_stop_gradient_forward_equal():
    m,q=tiny();other=copy.deepcopy(m);other.cfg['allow_task_grad_into_routing']=True
    z=torch.randn(2,4,16)
    torch.testing.assert_close(m(z,q)['fused'],other(z,q)['fused'])


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__();self.qwen=tiny()[1];self.proj=nn.Linear(32,16)
    def shared_qwen_model(self):return self.qwen
    def forward(self,batch):
        h=self.qwen(input_ids=batch['text_input_ids'],attention_mask=batch['text_attention_mask']).last_hidden_state
        z=self.proj(h[:,:4]);return {k:z[:,i] for i,k in enumerate(('text','vision','intrinsic_feature','interaction'))}


def model_fixture():
    cfg=dict(seed=7,model=dict(architecture_version=VERSION,hidden_dim=16,num_heads=4,dropout=0.,r1={'head_hidden_dim':16}),
             dataset=dict(name='fixture',positive_label=0))
    model=MaskedR1Model(cfg,FakeEncoder()).eval()
    batch=dict(text_input_ids=torch.randint(0,32,(2,6)),text_attention_mask=torch.ones(2,6),pixel_values=torch.randn(2,3,8,8),image_owner=torch.arange(2))
    return model,batch


def test_21_metadata_labels_not_prediction_inputs():
    m,b=model_fixture();base=m(b)['logits'];b.update(labels=torch.tensor([0,1]),ids=['x','y']);torch.testing.assert_close(m(b)['logits'],base)
    b.update(labels=torch.tensor([1,0]),ids=['z','a']);torch.testing.assert_close(m(b)['logits'],base)


def test_22_standalone_student_no_reference(tmp_path):
    m,b=model_fixture();p=tmp_path/'student.pth';torch.save(m.state_dict(),p)
    restored,_=model_fixture();restored.load_state_dict(torch.load(p,weights_only=True),strict=True)
    torch.testing.assert_close(m(b)['logits'],restored(b)['logits'])
    assert not any('reference' in n for n,_ in restored.named_modules())


def test_23_amp_resume_and_dedup(tmp_path):
    m,q=tiny('sdpa',torch.bfloat16);out=m(torch.randn(2,4,16,dtype=torch.bfloat16),q)
    assert torch.isfinite(out['utility']).all()
    opt=torch.optim.Adam(m.parameters());out['utility'].sum().backward();opt.step()
    p=tmp_path/'resume.pt';torch.save(dict(model=m.state_dict(),optimizer=opt.state_dict()),p)
    restored,_=tiny('sdpa',torch.bfloat16);new_opt=torch.optim.Adam(restored.parameters());saved=torch.load(p,weights_only=True)
    restored.load_state_dict(saved['model'],strict=True);new_opt.load_state_dict(saved['optimizer'])
    assert len(opt.state)==len(new_opt.state)
    assert len(deduplicate_rows([{'id':'a','p':.4},{'id':'a','p':.4}]))==1
    with pytest.raises(ValueError):deduplicate_rows([{'id':'a','p':.4},{'id':'a','p':.5}])


def test_24_threshold_contract_and_averaged_reselection():
    with pytest.raises(ValueError):find_best_macro_f1_threshold([0,1],[.8,.2],split='test')
    r=find_best_macro_f1_threshold([0,1],[.8,.2],[.4,.6],split='val');assert r['threshold']==.4
    r=find_best_macro_f1_threshold([0,1],[.8,.2],[.4,.5,.6],split='val');assert r['threshold']==.5
    from mmfnd.evaluation import checkpoint_threshold_selection
    with pytest.raises(ValueError):checkpoint_threshold_selection({})
    source=(Path(__file__).parents[1]/'train.py').read_text()
    assert source.index('average_checkpoints(top_paths')<source.index('checkpoint_reference={"kind": "top_k_average"')


def test_25_brier_factor_and_top_label_ece():
    y=np.array([0,1]);p=np.array([[.7,.3],[.6,.4]])
    a=compute_classification_metrics(y,p,.5);b=compute_classification_metrics(y,p,.65)
    double=((p-np.eye(2)[y])**2).sum(-1).mean()
    assert double==pytest.approx(2*a['brier']) and a['ece']==b['ece'] and a['accuracy']!=b['accuracy']


def test_26_positive_temperature_preserves_auc_metadata():
    logits=torch.tensor([[2.,0.],[1.,0.],[0.,1.],[-2.,0.]],dtype=torch.float64)
    y=[0,1,0,1];p=logits.softmax(-1).numpy();q=(logits/2).softmax(-1).numpy()
    assert np.array_equal(np.argsort(p[:,0]),np.argsort(q[:,0]))
    assert compute_classification_metrics(y,p,.5)['auc']==compute_classification_metrics(y,q,.5)['auc']
    assert not np.array_equal(p,q)


@pytest.mark.parametrize('options',[
    {'use_global':False},{'explicit_relations':False},{'explicit_vacuity':False},
    {'evidential':False},{'fusion':'uniform'},{'judge_type':'mlp'},
    {'visibility':'causal'},{'allow_task_grad_into_routing':True},
])
def test_ablation_multistep_no_unused_head_parameters(options):
    from mmfnd.losses_masked_r1 import masked_r1_loss
    m,q=tiny(**options);optimizer=torch.optim.Adam(m.parameters(),lr=.01);seen=set()
    for epoch in range(1,4):
        out=m(torch.randn(3,4,16),q);out['logits']=nn.Linear(16,2)(out['fused'])
        probabilities=torch.rand(3,16,2).softmax(-1)
        loss,_=masked_r1_loss(out,torch.tensor([0,1,0]),probabilities,1.,{},epoch)
        optimizer.zero_grad();loss.backward();seen.update(n for n,p in m.named_parameters() if p.grad is not None and p.grad.abs().sum()>0);optimizer.step()
    assert not [n for n,p in m.named_parameters() if p.requires_grad and n not in seen]


def test_reference_subset_mean_per_sample_and_empty_mask_bias():
    logits=torch.randn(2,16,2,requires_grad=True);y=torch.tensor([0,1]);available=torch.tensor([[1,0,0,0],[1,1,1,1]]).bool()
    expected=(F.cross_entropy(logits[:1,1],y[:1])+F.cross_entropy(logits[1,1:],y[1:].expand(15)))/2
    torch.testing.assert_close(subset_ce(logits,y,available),expected)


def test_frozen_random_judge_and_cross_sample_shuffle():
    from mmfnd.frozen_judge_controls import FrozenJudgeBackbone,fixed_role_permutations,shuffle_cached_views
    _,q=tiny();before={k:v.clone() for k,v in q.state_dict().items()}
    random=FrozenJudgeBackbone(q,True)
    for k,v in q.state_dict().items():torch.testing.assert_close(v,before[k])
    assert not torch.equal(random.layers[0].self_attn.q_proj.weight,q.layers[-2].self_attn.q_proj.weight)
    ids=['a','b','c'];z=torch.arange(3*4*2).reshape(3,4,2);perm=fixed_role_permutations(ids,9)
    mixed=shuffle_cached_views(z,ids,perm)
    assert all(perm['roles'][str(r)][sid]!=sid for sid in ids for r in range(4))
    assert all(torch.equal(z[:,r].sort(dim=0).values,mixed[:,r].sort(dim=0).values) for r in range(4))
    # Lookup is global and works when downstream inference takes a one-row batch.
    assert mixed[:1].shape==(1,4,2)


def test_runtime_rng_resume_and_calibration_split_guard():
    from mmfnd.r1_runtime import capture_rng,restore_rng,fit_scalar_temperature
    state=capture_rng();expected=torch.randn(12);restore_rng(state);torch.testing.assert_close(torch.randn(12),expected)
    with pytest.raises(ValueError):fit_scalar_temperature(torch.randn(3,2),torch.tensor([0,1,0]),calibration_ids=['a'],selection_ids=['a'],split='val_cal')
    with pytest.raises(ValueError):fit_scalar_temperature(torch.randn(3,2),torch.tensor([0,1,0]),calibration_ids=['a'],selection_ids=['b'],split='test')


def test_offline_mechanism_groups_do_not_choose_by_student():
    from mmfnd.mechanisms_masked_r1 import mechanism_report
    rows=[];refs=[];labels={}
    for i,single in enumerate(([.9,.1,.1,.1],[.1,.9,.9,.9],[.9]*4,[.1]*4)):
        q=torch.full((16,2),.5)
        for code,p in zip((1,2,4,8),single):q[code]=torch.tensor([p,1-p])
        refs.append(dict(sample_id=str(i),augmentation_id='clean',q=q))
        labels[str(i)]=0
        rows.append(dict(id=str(i),probabilities=[.6,.4],utility=[99,-8,0,4],weights=[.25]*4,
                         relation_probs=[[1/3]*3]*6,pair_available=[True]*6,relation_vacuity=[.5]*6,
                         deleted_probabilities=[[.5,.5]]*4))
    report=mechanism_report(rows,refs,labels,.5,1.)
    assert all(g['samples']==1 for g in report['groups'].values())
    assert report['groups']['minority_correct']['view_identity_counts']==[1,0,0,0]


def test_mixed_parameter_cpu_autocast_actual_qwen():
    m,b=model_fixture();m.encoder.qwen.bfloat16()
    with torch.autocast('cpu',dtype=torch.bfloat16):
        out=m(b);loss=out['logits'].float().square().mean()+out['utility'].square().mean()
    loss.backward()
    assert torch.isfinite(out['logits']).all()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_raw_intervention_recomputes_views_and_padding_general_index():
    m,b=model_fixture();z=m.encode_views(b);a=torch.tensor([[1,0,1,1]]*2).bool()
    internal=m.predict_views(z,a)
    assert internal['evidence_features'].data_ptr()==z.data_ptr()
    def change(batch):
        batch['text_input_ids']=(batch['text_input_ids']+3)%32
        return batch
    raw=m.raw_modality_intervention(b,change)
    assert not torch.equal(raw['evidence_features'],z)
    # Same last valid index operation used in the actual encoder, left/right padding.
    masks=torch.tensor([[1,1,0,0],[0,0,1,1],[0,1,1,0]]).bool()
    indices=torch.arange(4)[None].expand_as(masks).masked_fill(~masks,-1).amax(-1)
    assert indices.tolist()==[1,3,2]


def test_missing_token_bias_states_cannot_reach_global():
    m,q=tiny();a=torch.tensor([[1,1,0,0],[1,0,0,0]]).bool()
    sequence=torch.randn(2,11,32,requires_grad=True)
    allowed,_=availability_mask(a)
    out=m.run_layers(sequence,q,a)
    altered=sequence.detach().clone()
    for b in range(2):altered[b,~allowed[b,10]]=torch.randn_like(altered[b,~allowed[b,10]])*100
    torch.testing.assert_close(m.run_layers(altered,q,a)[:,10],out[:,10],rtol=0,atol=0)
    gradient,=torch.autograd.grad(out[:,10,0].sum(),sequence)
    for b in range(2):assert gradient[b,~allowed[b,10]].count_nonzero()==0


def test_reference_prediction_rejects_label_fields():
    from mmfnd.reference_pipeline import predict_reference,label_free_batches
    with pytest.raises(ValueError):predict_reference(nn.Identity(),[{'labels':torch.tensor([0])}],torch.device('cpu'),'fp32',torch.tensor([.5,.5]))
    batches=list(label_free_batches([{'labels':torch.tensor([0]),'ids':['x'],'text_input_ids':torch.ones(1,2)}]))
    assert 'labels' not in batches[0]
