"""Strict cache provenance and grouped encoder-level cross-fit manifests."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def ids_hash(ids): return digest(sorted(map(str,ids)))


def grouped_folds(records, seed, n_splits=3):
    ids=[str(x['id']) for x in records]
    if len(set(ids)) != len(ids): raise ValueError('duplicate original IDs')
    # Union real grouping fields and normalized exact-text duplicates. No invented event labels.
    parent=list(range(len(records)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    seen={}
    for i,r in enumerate(records):
        keys=[('text',re.sub(r'\s+','',r['text']).casefold())]
        keys += [(k,str(r[k])) for k in ('event_id','source_id','near_duplicate_group','original_id') if r.get(k)]
        for key in keys:
            if key in seen: parent[find(i)]=find(seen[key])
            else: seen[key]=i
    groups=np.array([find(i) for i in range(len(records))]); labels=np.array([r['label'] for r in records])
    if len(set(groups)) < max(6,n_splits): raise ValueError('not enough independent groups for cross-fit/calibration')
    folds=[]
    splitter=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for fold,(rest,target) in enumerate(splitter.split(ids,labels,groups)):
        inner_splits=min(5,len(set(groups[rest])))
        inner=StratifiedGroupKFold(n_splits=inner_splits,shuffle=True,random_state=seed+fold+1)
        fit_local,cal_local=next(inner.split(rest,labels[rest],groups[rest]))
        fit,cal=rest[fit_local],rest[cal_local]
        entry=dict(fold=fold,fit_ids=[ids[i] for i in fit],cal_ids=[ids[i] for i in cal],target_ids=[ids[i] for i in target],
                   fit_groups=sorted(set(groups[fit].tolist())),cal_groups=sorted(set(groups[cal].tolist())),
                   target_groups=sorted(set(groups[target].tolist())))
        validate_fold(entry)
        entry.update(fit_id_hash=ids_hash(entry['fit_ids']),cal_id_hash=ids_hash(entry['cal_ids']))
        folds.append(entry)
    return dict(seed=seed,rule='union of real event/source/near_duplicate/original IDs and whitespace-normalized casefold exact text',folds=folds)


def validate_fold(fold):
    for suffix in ('ids','groups'):
        sets=[set(fold[f'{k}_{suffix}']) for k in ('fit','cal','target')]
        if any(sets[i]&sets[j] for i in range(3) for j in range(i)):
            raise ValueError('encoder/probe/calibration/target leakage')
        if any(not s for s in sets): raise ValueError('empty reference partition')


def student_source_fingerprint(root):
    root=Path(root)
    names=('revision_masked_r1.py','relation_supervision.py','evidence_utility.py','losses_masked_r1.py',
           'evaluation_masked_r1.py','model.py','factory.py','r1_data.py','image_preprocessing.py')
    return digest({n:file_sha(root/'mmfnd'/n) for n in names})


def backbone_fingerprint(root,config):
    root=Path(root)
    model={k:v for k,v in config['model'].items() if k not in ('r1','lgled','architecture_version','view_dropout_probability','causal_effect_scale')}
    assets={}
    for key in ('text_backbone','vision_backbone'):
        path=Path(model[key]);path=path if path.is_absolute() else root/path
        for f in sorted(path.iterdir()):
            if f.is_file() and f.suffix in ('.json','.txt','.model','.safetensors','.bin'):
                assets[f'{key}/{f.name}']=file_sha(f)
    return model,digest([model,assets])


def verify_backbone_identity(root,config):
    expected=config.get("r1_backbone_identity",{}).get("model_fingerprint")
    if not expected or backbone_fingerprint(root,config)[1]!=expected:
        raise ValueError("R1 frozen pretrained backbone fingerprint mismatch")
    if config.get("r1_student_source_fingerprint") != student_source_fingerprint(root):
        raise ValueError("R1 checkpoint student implementation fingerprint mismatch")


def experiment_fingerprint(root, config, folds):
    root=Path(root)
    # Hash backbone bytes once per process; no downloads and no trained checkpoint initialization.
    model,model_hash=backbone_fingerprint(root,config)
    data={k:v for k,v in config['data'].items() if k not in ('num_workers','pin_memory','persistent_workers','prefetch_factor')}
    processed=Path(data['processed_dir']);processed=processed if processed.is_absolute() else root/processed
    records=[json.loads(x) for x in (processed/'train.jsonl').read_text().splitlines() if x]
    image_root=Path(data.get('image_root',data['root']));image_root=image_root if image_root.is_absolute() else root/image_root
    images={}
    for rel in sorted({p for r in records for p in r['images'][:int(data['max_images'])]}):
        images[rel]=file_sha(image_root/rel)
    sources={f.name:file_sha(f) for f in (root/'mmfnd').glob('*.py') if f.name in
             ('model.py','r1_data.py','reference_subset_probe.py','reference_pipeline.py','r1_cache.py','image_preprocessing.py','factory.py')}
    from importlib.metadata import version
    runtime={name:version(name) for name in ('torch','transformers','peft','Pillow','tokenizers')}
    return dict(schema='r1_oof_v1',runtime=runtime,reference_seed=int(config['seed']),data=data,model=model,
                preprocess_fingerprint=digest(data),model_fingerprint=model_hash,
                split_fingerprint=digest(folds),train_manifest_sha256=file_sha(processed/'train.jsonl'),
                image_content_fingerprint=digest(images),source_fingerprint=digest(sources),
                reference=config['reference'],reference_optimizer={k:config['train'][k] for k in
                    ('lora_learning_rate','vision_learning_rate','head_learning_rate','weight_decay','per_gpu_batch_size','grad_accum_steps','precision','gradient_clip_norm')},
                augmentation_protocol='replay_clean_plus_aug1_v1',
                probability_order=['fake','real'],subset_order=list(range(16)),folds=folds)


def validate_probabilities(q):
    q=torch.as_tensor(q)
    if q.shape[-2:] != (16,2) or not torch.isfinite(q).all() or (q<0).any() or (q>1).any():
        raise ValueError('invalid cached probabilities')
    torch.testing.assert_close(q.sum(-1),torch.ones_like(q[...,0]),atol=1e-5,rtol=0)


class TargetCache:
    def __init__(self,path,expected):
        path=Path(path)
        payload=torch.load(path,map_location='cpu',weights_only=False)
        manifest=json.loads(path.with_suffix('.json').read_text())
        if manifest['payload_sha256']!=file_sha(path): raise ValueError('target cache payload hash mismatch')
        if payload['fingerprint'] != expected: raise ValueError('cache preprocess/model/split/reference seed mismatch')
        self.scale=float(payload['scale']);self.rows=payload['rows'];self.epsilon=float(payload.get('epsilon',1e-7))
        if not np.isfinite(self.scale) or self.scale<=0: raise ValueError('invalid training-only scale')
        self.fingerprint=expected
        self.index={}
        checkpoint_hashes={}
        for row in self.rows:
            key=(str(row['sample_id']),row['augmentation_id'])
            if key in self.index: raise ValueError('duplicate cached augmentation')
            fold=expected['folds']['folds'][row['fold']]
            validate_fold(fold)
            if key[0] not in set(fold['target_ids']): raise ValueError('not a held-out prediction')
            if row['fit_id_hash']!=fold['fit_id_hash'] or row['cal_id_hash']!=fold['cal_id_hash']:
                raise ValueError('reference label-use hashes mismatch')
            fold_number=row['fold']
            if fold_number not in checkpoint_hashes:
                checkpoint_path=path.parent/f'fold_{fold_number}.pth'
                if not checkpoint_path.is_file():raise ValueError('missing reference checkpoint provenance')
                checkpoint_hashes[fold_number]=file_sha(checkpoint_path)
            if checkpoint_hashes[fold_number]!=row['reference_checkpoint_sha256']:
                raise ValueError('reference checkpoint differs from cached OOF predictions')
            validate_probabilities(row['q'])
            self.index[key]=row

    def lookup(self,ids,augmentation_ids,availability,device):
        result=[]
        for sid,aid,available in zip(ids,augmentation_ids,availability):
            key=(str(sid),aid)
            if key not in self.index: raise ValueError(f'missing exact cached augmentation {key}')
            row=self.index[key]
            if row['availability']!=available.cpu().bool().tolist(): raise ValueError('cached availability mismatch')
            result.append(torch.as_tensor(row['q']))
        return torch.stack(result).to(device).detach()
