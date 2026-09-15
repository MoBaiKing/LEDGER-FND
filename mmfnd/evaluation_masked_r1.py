"""R1 prediction/diagnostic separation and frozen validation threshold metadata."""
import time
import numpy as np
import torch
from mmfnd.evaluation import (PROTOCOL,LABEL_SEMANTICS,validate_threshold_selection,
    find_best_macro_f1_threshold,threshold_grid,compute_classification_metrics)
from mmfnd.r1_cache import digest


def model_hash(model):
    import hashlib
    h=hashlib.sha256()
    # Adapter checkpoint hash + exact pretrained fingerprint identify full deployment model.
    for name,p in model.named_parameters():
        if p.requires_grad:
            h.update(name.encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    h.update(digest(model.runtime_config.get('r1_backbone_identity',{})).encode())
    return h.hexdigest()


def deduplicate_rows(rows):
    seen={}
    for row in rows:
        sid=str(row['id'])
        if sid in seen:
            if digest(row)!=digest(seen[sid]):raise ValueError('inconsistent duplicate prediction')
        else:seen[sid]=row
    return list(seen.values())


@torch.no_grad()
def predict_student(model,loader,device,precision='fp32',deletions=False):
    """No labels in saved prediction records. sample ID is external join key only."""
    from mmfnd.engine import autocast_context
    model.eval();rows=[];start=time.monotonic()
    for raw in loader:
        inputs={k:raw[k].to(device) for k in ('text_input_ids','text_attention_mask','pixel_values','image_owner')}
        with autocast_context(device,precision):
            z=model.encode_views(inputs)
            out=model.predict_views(z)
            deleted=[]
            if deletions:
                for i in range(4):
                    a=torch.ones(z.shape[:2],device=device,dtype=torch.bool);a[:,i]=False
                    deleted.append(model.predict_views(z,a)['logits'].float().softmax(-1).cpu())
        p=out['logits'].float().softmax(-1).cpu()
        for i,sid in enumerate(raw['ids']):
            row=dict(id=str(sid),probabilities=p[i].tolist(),utility=out['utility'][i].float().cpu().tolist(),
                     weights=out['final_evidence_weights'][i].float().cpu().tolist(),
                     relation_probs=out['relation_probs'][i].float().cpu().tolist(),
                     pair_available=out['pair_available'][i].cpu().tolist())
            if 'relation_vacuity' in out:row['relation_vacuity']=out['relation_vacuity'][i].cpu().tolist()
            if deletions:row['deleted_probabilities']=[x[i].tolist() for x in deleted]
            rows.append(row)
    return deduplicate_rows(rows),dict(inference_seconds=time.monotonic()-start)


def evaluate_r1(model,loader,device,positive_label,class_names,decision_threshold=None,precision='fp32',
                show_progress=True,*,split,tune_threshold=False,threshold_selection=None,checkpoint_reference=None):
    if positive_label!=0 or {int(k):v for k,v in class_names.items()}!={0:'fake',1:'real'}:
        raise ValueError('R1 requires Fake=0 Real=1')
    if split not in ('val','test'):raise ValueError('invalid evaluation split')
    if tune_threshold and split!='val':raise ValueError('test tuning forbidden')
    if not tune_threshold:
        t=validate_threshold_selection(threshold_selection)
        if threshold_selection.get('probability_type')!='raw' or threshold_selection.get('temperature')!=1.:
            raise ValueError('raw/calibrated metadata mismatch')
        if decision_threshold is not None and decision_threshold!=t:raise ValueError('frozen threshold mismatch')
        decision_threshold=t
    rows,timing=predict_student(model,loader,device,precision)
    # Labels read only after predictions; training only invokes this on validation.
    dataset=loader.dataset
    if not hasattr(dataset,'records'):raise ValueError('evaluation dataset requires original records for external label join')
    labels_by_id={str(r['id']):int(r['label']) for r in dataset.records}
    y=np.array([labels_by_id[r['id']] for r in rows]);p=np.array([r['probabilities'] for r in rows])
    identity=model_hash(model)
    if tune_threshold:
        if not checkpoint_reference:raise ValueError('checkpoint reference required')
        best=find_best_macro_f1_threshold(y,p[:,0],threshold_grid(model.runtime_config['train']),split=split)
        decision_threshold=best['threshold']
        threshold_selection=dict(evaluation_protocol=PROTOCOL,threshold=decision_threshold,decision_threshold=decision_threshold,
            threshold_source='validation',threshold_objective='macro_f1',threshold_tie_break='closest_to_0.5_then_lower',
            threshold_grid=best['threshold_grid'],val_best_macro_f1=best['macro_f1'],positive_label=0,positive_class='fake',
            label_semantics=LABEL_SEMANTICS,checkpoint_reference=checkpoint_reference,model_hash=identity,
            selection_split_hash=digest([(r['id'],int(label)) for r,label in zip(rows,y)]),
            probability_type='raw',calibrated=False,temperature=1.,seed=model.runtime_config['seed'])
    elif threshold_selection.get('model_hash')!=identity:raise ValueError('threshold belongs to different model weights')
    metrics=compute_classification_metrics(y,p,decision_threshold,positive_label=0)
    metrics.update(split=split,threshold_selection=threshold_selection,**timing)
    metrics.update({k:threshold_selection[k] for k in ('evaluation_protocol','threshold_source','threshold_objective','val_best_macro_f1')})
    for row,label in zip(rows,y):
        row.update(label=int(label),prediction=0 if row['probabilities'][0]>=decision_threshold else 1,threshold=decision_threshold,
                   probability_type='raw',temperature=1.)
    return metrics,rows,float(decision_threshold)
