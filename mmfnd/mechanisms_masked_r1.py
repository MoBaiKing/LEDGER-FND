"""Offline labeled diagnostics; never used to compute deployed weights."""
import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import confusion_matrix
from mmfnd.evidence_utility import contribution_targets
from mmfnd.relation_supervision import relation_targets
from mmfnd.evaluation import compute_classification_metrics


def correlation(x,y):
    x,y=np.asarray(x).ravel(),np.asarray(y).ravel()
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(spearmanr(x,y).statistic)


def risk_coverage(uncertainty,error):
    u,e=np.asarray(uncertainty).ravel(),np.asarray(error).ravel()
    if not len(u):return {'status':'EMPTY'}
    order=np.argsort(u,kind='stable');risk=np.cumsum(e[order])/np.arange(1,len(e)+1)
    return dict(coverage=[float(k/len(e)) for k in range(1,len(e)+1)],risk=risk.tolist(),aurc=float(risk.mean()),
                spearman_with_error=correlation(u,e))


def mechanism_report(predictions,reference_rows,labels_by_id,threshold,scale,zero_band=.01,min_group=10):
    ref={str(r['sample_id']):r for r in reference_rows if r['augmentation_id']=='clean'}
    ids=[str(r['id']) for r in predictions]
    if any(i not in ref or i not in labels_by_id for i in ids):raise ValueError('diagnostic ID coverage mismatch')
    y=np.array([labels_by_id[i] for i in ids]);q=torch.stack([torch.as_tensor(ref[i]['q']) for i in ids]).double()
    effects=contribution_targets(q,torch.tensor(y));phi=effects['phi'].numpy()
    p=np.array([r['probabilities'] for r in predictions]);w=np.array([r['weights'] for r in predictions]);u=np.array([r['utility'] for r in predictions])
    valid=np.array([r['pair_available'] for r in predictions],dtype=bool)
    student_rel=np.array([r['relation_probs'] for r in predictions]);target_rel=relation_targets(q[:,[1,2,4,8],0]).numpy()
    single_pred=np.where(q[:,[1,2,4,8],0].numpy()>=.5,0,1);correct=single_pred==y[:,None];count=correct.sum(-1)
    nll=-np.log(p[np.arange(len(y)),y].clip(1e-7,1))
    deletion=None
    if all('deleted_probabilities' in r for r in predictions):
        deleted=np.array([r['deleted_probabilities'] for r in predictions])
        deleted_nll=-np.log(np.take_along_axis(deleted,y[:,None,None].repeat(4,axis=1),axis=2).squeeze(-1).clip(1e-7,1))
        deletion=deleted_nll-nll[:,None]
    groups={}
    for name,number in [('minority_correct',1),('minority_wrong',3),('unanimous_correct',4),('unanimous_wrong',0)]:
        mask=count==number;n=int(mask.sum())
        info=dict(samples=n,status='EMPTY' if n==0 else 'TOO_SMALL' if n<min_group else 'DESCRIPTIVE',
                  labels={str(k):int((y[mask]==k).sum()) for k in (0,1)})
        if n:
            metric=compute_classification_metrics(y[mask],p[mask],threshold)
            info.update(accuracy=metric['accuracy'],nll=metric['nll'],macro_f1=metric['macro_f1'] if len(set(y[mask]))==2 else None)
            minority=(correct if number==1 else ~correct) if number in (1,3) else None
            if minority is not None:
                members=minority[mask]
                info.update(view_identity_counts=members.sum(0).tolist(),minority_weight=float(w[mask][members].mean()),
                            reference_phi=float(phi[mask][members].mean()),student_utility=float(u[mask][members].mean()),
                            student_actual_deletion_effect=float(deletion[mask][members].mean()) if deletion is not None else None)
        groups[name]=info
    target=phi/scale;covered=np.abs(target)>zero_band
    rel_error=-(target_rel*np.log(student_rel.clip(1e-7,1))).sum(-1)
    utility=dict(mae=float(np.abs(u-target).mean()),spearman=correlation(u,target),zero_band=zero_band,
                 sign_coverage=float(covered.mean()),sign_accuracy=float(((u>0)==(target>0))[covered].mean()) if covered.any() else None,
                 top1_beneficial_view_accuracy=float((u.argmax(-1)==target.argmax(-1)).mean()),
                 reference_vs_student_deletion_spearman=correlation(phi,deletion) if deletion is not None else None,
                 utility_vs_student_deletion_spearman=correlation(u,deletion) if deletion is not None else None)
    rankings={'entropy':-(student_rel*np.log(student_rel.clip(1e-7,1))).sum(-1),'one_minus_max':1-student_rel.max(-1)}
    if all('relation_vacuity' in r for r in predictions):rankings['vacuity']=np.array([r['relation_vacuity'] for r in predictions])
    relation=dict(target_mean=target_rel[valid].mean(0).tolist() if valid.any() else None,
                  soft_ce=float(rel_error[valid].mean()) if valid.any() else None,
                  brier=float(((student_rel-target_rel)**2).sum(-1)[valid].mean()) if valid.any() else None,
                  argmax_confusion=confusion_matrix(target_rel.argmax(-1)[valid],student_rel.argmax(-1)[valid],labels=[0,1,2]).tolist(),
                  risk_coverage={name:risk_coverage(value[valid],rel_error[valid]) for name,value in rankings.items()})
    return dict(samples=len(ids),groups=groups,utility=utility,relations=relation,
                claims='Reference-task consistency only; no independent human semantic ground truth or causal identification')
