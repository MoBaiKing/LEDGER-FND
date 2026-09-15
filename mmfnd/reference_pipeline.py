"""Sequential encoder-level cross-fitting, with optional DDP within one fold."""
import copy
import gc
import json
import math
import time
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from mmfnd.model import MultimodalIntrinsicEvidenceEncoder
from mmfnd.revision_masked_r1 import VIEW_KEYS
from mmfnd.reference_subset_probe import ReferenceSubsetProbe, subset_ce, smoothed_prior, subset_probabilities
from mmfnd.factory import build_loader,build_processor
from mmfnd.data import move_batch
from mmfnd.utils import seed_everything,dump_json
from mmfnd.engine import autocast_context,unwrap_model
from mmfnd.r1_cache import grouped_folds,experiment_fingerprint,file_sha,digest,validate_probabilities
from mmfnd.evidence_utility import contribution_targets,fit_utility_scale
from mmfnd.relation_supervision import relation_targets


class ReferenceModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        c=config['model'];d=int(c['hidden_dim'])
        self.encoder=MultimodalIntrinsicEvidenceEncoder(c,d,int(c['num_heads']))
        self.encoder.general_last_valid_index=True
        self.probe=ReferenceSubsetProbe(d,int(config['reference'].get('width',256)))

    def encode(self,inputs):
        encoded=self.encoder(inputs)
        return torch.stack([encoded[k] for k in VIEW_KEYS],1)

    def forward(self,inputs): return self.probe.all_logits(self.encode(inputs))


def input_tensors(batch):
    return {k:batch[k] for k in ('text_input_ids','text_attention_mask','pixel_values','image_owner')}


def partition_loader(dataset,ids,config,context,train=False):
    selected=set(ids)
    factor=2 if dataset.pool else 1
    indices=[i for i in range(len(dataset)) if str(dataset.records[i//factor]['id']) in selected]
    subset=Subset(dataset,indices)
    sampler=DistributedSampler(subset,num_replicas=context.world_size,rank=context.rank,shuffle=True) if train and context.distributed else None
    return DataLoader(subset,batch_size=config['train']['per_gpu_batch_size'],shuffle=train and sampler is None,
                      sampler=sampler,num_workers=config['data']['num_workers'],collate_fn=dataset.collate_fn)


def label_free_batches(loader):
    """External prediction boundary: no target-label fields enter the predictor."""
    keys=('text_input_ids','text_attention_mask','pixel_values','image_owner','ids','augmentation_ids','availability')
    for batch in loader:
        yield {k:batch[k] for k in keys if k in batch}


@torch.no_grad()
def predict_reference(model,loader,device,precision,prior,temperature=1.):
    """Prediction function NEVER reads or accepts target labels; encoder runs once/batch."""
    model.eval();rows=[]
    for raw in loader:
        if 'labels' in raw or 'label' in raw:
            raise ValueError('predict_reference requires label-free input batches')
        inputs={k:raw[k].to(device) for k in ('text_input_ids','text_attention_mask','pixel_values','image_owner')}
        with autocast_context(device,precision):
            z=model.encode(inputs)
            q=subset_probabilities(model.probe,z,prior,temperature)
        for i,(sid,aid) in enumerate(zip(raw['ids'],raw['augmentation_ids'])):
            rows.append(dict(sample_id=str(sid),augmentation_id=aid,availability=raw['availability'][i].tolist(),q=q[i].cpu()))
    return rows


@torch.no_grad()
def calibration_logits(model,loader,device,precision):
    model.eval();logits=[];labels=[];availability=[]
    for raw in loader:
        batch=move_batch(raw,device)
        with autocast_context(device,precision): value=model(input_tensors(batch))
        logits.append(value.float().cpu());labels.append(batch['labels'].cpu());availability.append(batch['availability'].cpu())
    return torch.cat(logits),torch.cat(labels),torch.cat(availability)


def prepare_identity(root,config):
    path=Path(config['data']['processed_dir']);path=path if path.is_absolute() else root/path
    records=[json.loads(s) for s in (path/'train.jsonl').read_text().splitlines() if s]
    folds=grouped_folds(records,int(config['seed']),int(config['reference'].get('folds',3)))
    return records,folds,experiment_fingerprint(root,config,folds)


def build_reference(root,config,context,directory):
    from train import build_optimizer
    if (config['reference'].get('initialization') != 'pretrained_only'
            or config['reference'].get('label_smoothing',0) != 0
            or config['reference'].get('class_weight') is not None):
        raise ValueError('reference requires pretrained-only initialization and unweighted, unsmoothed CE')
    directory=Path(directory);started=time.monotonic()
    if context.is_main:
        directory.mkdir(parents=True,exist_ok=True)
        if any(p.name != 'preflight.json' for p in directory.iterdir()):
            raise FileExistsError(f"Reference directory already contains run artifacts: {directory}")
        records,folds,fingerprint=prepare_identity(root,config)
        dump_json(dict(fingerprint=fingerprint,status='running',config=config),directory/'reference_manifest.json')
    if context.distributed: dist.barrier()
    manifest=json.loads((directory/'reference_manifest.json').read_text())
    fingerprint=manifest['fingerprint'];folds=fingerprint['folds']
    processor=build_processor(root,config)
    dataset=build_loader(root,config,'train',processor,shuffle=True).dataset
    records=dataset.records
    labels_by_id={str(r['id']):r['label'] for r in records}
    all_rows=[];val_folds=[];costs=[]
    for fold in folds['folds']:
        fold_start=time.monotonic()
        # Same pretrained and new front-end/probe init for every fold, no old best ckpt.
        seed_everything(int(config['seed']))
        model=ReferenceModel(config).to(context.device)
        fit_loader=partition_loader(dataset,fold['fit_ids'],config,context,True)
        cal_loader=partition_loader(dataset,fold['cal_ids'],config,context)
        target_loader=partition_loader(dataset,fold['target_ids'],config,context)
        prior=smoothed_prior([labels_by_id[i] for i in fold['fit_ids']],float(config['reference'].get('prior_smoothing',1)))
        if context.distributed:
            model=DDP(model,device_ids=[context.local_rank] if context.device.type=='cuda' else None,broadcast_buffers=False)
        optimizer=build_optimizer(model,config)
        precision=config['train']['precision'];accum=int(config['train']['grad_accum_steps'])
        best=float('inf');bad=0;fit_steps=0;history=[]
        raw_model=unwrap_model(model)
        ckpt_path=directory/f"fold_{fold['fold']}.pth"
        for epoch in range(1,int(config['reference'].get('epochs',3))+1):
            model.train();optimizer.zero_grad(set_to_none=True)
            if hasattr(fit_loader.sampler,'set_epoch'):fit_loader.sampler.set_epoch(epoch)
            for step,raw in enumerate(fit_loader,1):
                batch=move_batch(raw,context.device)
                window=min(accum,len(fit_loader)-((step-1)//accum)*accum)
                with autocast_context(context.device,precision):
                    logits=model(input_tensors(batch))
                    loss=subset_ce(logits,batch['labels'],batch['availability'])
                if not torch.isfinite(loss):raise FloatingPointError('reference loss nonfinite')
                (loss/window).backward()
                if step%accum==0 or step==len(fit_loader):
                    norm=nn.utils.clip_grad_norm_(model.parameters(),config['train']['gradient_clip_norm'])
                    if not torch.isfinite(norm):raise FloatingPointError('reference gradient nonfinite')
                    optimizer.step();optimizer.zero_grad(set_to_none=True);fit_steps+=1
            if context.distributed:dist.barrier()
            stop=False
            if context.is_main:
                logits,y,a=calibration_logits(raw_model,cal_loader,context.device,precision)
                nll=float(subset_ce(logits,y,a));history.append(dict(epoch=epoch,cal_nll=nll))
                if nll<best:
                    best=nll;bad=0
                    trainable={n for n,p in raw_model.named_parameters() if p.requires_grad}
                    torch.save(dict(state={n:t.detach().cpu() for n,t in raw_model.state_dict().items() if n in trainable},
                                    trainable_names=sorted(trainable),fold=fold,prior=prior,epoch=epoch,
                                    fingerprint=fingerprint),ckpt_path)
                else:bad+=1
                stop=bad>=int(config['reference'].get('patience',2))
                print(json.dumps(dict(reference_fold=fold['fold'],epoch=epoch,cal_nll=nll)),flush=True)
            if context.distributed:
                signal=torch.tensor(int(stop),device=context.device);dist.broadcast(signal,0);stop=bool(signal.item());dist.barrier()
            if stop:break
        if context.is_main:
            saved=torch.load(ckpt_path,map_location='cpu',weights_only=False)
            state=raw_model.state_dict();state.update(saved['state']);raw_model.load_state_dict(state,strict=True)
            logits,y,a=calibration_logits(raw_model,cal_loader,context.device,precision)
            candidates=config['reference'].get('temperature_grid',[.5,1.,2.,4.])
            temperature=min(candidates,key=lambda t:(float(subset_ce(logits/float(t),y,a)),abs(float(t)-1)))
            saved['temperature']=temperature;torch.save(saved,ckpt_path)
            del saved,state  # Do not retain a full frozen Qwen state across folds.
            rows=predict_reference(raw_model,label_free_batches(target_loader),context.device,precision,prior,temperature)
            checkpoint_hash=file_sha(ckpt_path)
            for row in rows:row.update(fold=fold['fold'],fit_id_hash=fold['fit_id_hash'],cal_id_hash=fold['cal_id_hash'],reference_checkpoint_sha256=checkpoint_hash)
            # OOF probability file precedes all target-label-based target construction.
            torch.save(dict(fingerprint=fingerprint,rows=rows),directory/f"oof_{fold['fold']}.pt")
            all_rows.extend(rows)
            val_loader=build_loader(root,config,'val',processor)
            val_rows=predict_reference(raw_model,label_free_batches(val_loader),context.device,precision,prior,temperature)
            val_folds.append(val_rows)
            costs.append(dict(fold=fold['fold'],seconds=time.monotonic()-fold_start,optimizer_steps=fit_steps,
                              world_size=context.world_size,history=history,temperature=temperature,
                              total_parameters=sum(p.numel() for p in raw_model.parameters()),
                              trainable_parameters=sum(p.numel() for p in raw_model.parameters() if p.requires_grad),
                              peak_cuda_bytes=torch.cuda.max_memory_allocated() if context.device.type=='cuda' else 0))
        del optimizer,model,raw_model
        gc.collect()
        if context.device.type=='cuda':torch.cuda.empty_cache()
        if context.distributed:dist.barrier()
    if context.is_main:
        torch.save(dict(fingerprint=fingerprint,rows=all_rows),directory/'oof_probabilities.pt')
        # Separate process/API entry also available: build_targets_from_oof.
        val_rows=val_folds[0]
        for i,row in enumerate(val_rows):
            if any(v[i]['sample_id']!=row['sample_id'] for v in val_folds):raise ValueError('val ensemble ID mismatch')
            row['q']=torch.stack([v[i]['q'] for v in val_folds]).mean(0)
        torch.save(dict(fingerprint=fingerprint,rows=val_rows,split='val',labels_read=False),directory/'val_probabilities.pt')
        dump_json(dict(fingerprint=fingerprint,status='probabilities_complete',config=config,
                       costs=costs,total_seconds=time.monotonic()-started),directory/'reference_manifest.json')
    if context.distributed:dist.barrier()


def build_targets_from_oof(directory,train_records):
    """Independent label-consuming stage, only AFTER OOF probabilities exist."""
    directory=Path(directory)
    payload=torch.load(directory/'oof_probabilities.pt',map_location='cpu',weights_only=False)
    rows=payload['rows'];labels={str(r['id']):int(r['label']) for r in train_records}
    if set(r['sample_id'] for r in rows)!=set(labels):raise ValueError('OOF coverage mismatch')
    q=torch.stack([r['q'] for r in rows]);validate_probabilities(q)
    y=torch.tensor([labels[r['sample_id']] for r in rows]);a=torch.tensor([r['availability'] for r in rows])
    effects=contribution_targets(q,y,a)
    scale=fit_utility_scale(effects['phi'],a)
    t=relation_targets(q[:,[1,2,4,8],0])
    for i,row in enumerate(rows):
        row.update(phi=effects['phi'][i],delta=effects['delta'][i],unsigned_tv=effects['unsigned_tv'][i],relation_target=t[i])
    payload.update(scale=scale,epsilon=1e-7,scale_fit_split='OOF_train',rows=rows)
    target_path=directory/'targets.pt'
    torch.save(payload,target_path)
    dump_json(dict(payload_sha256=file_sha(target_path),fingerprint_sha256=digest(payload['fingerprint']),scale=scale),target_path.with_suffix('.json'))
    nll=-q.gather(-1,y[:,None,None].expand(-1,16,1)).squeeze(-1).clamp_min(1e-7).log()
    accuracy=((q[...,0]>=.5)==(y[:,None]==0)).float().mean(0)
    report=dict(oof_nll=nll.mean(0).tolist(),binary_brier=((q[...,0]-(y[:,None]==0).float())**2).mean(0).tolist(),
                subset_accuracy=accuracy.tolist(),single_view_accuracy=accuracy[[1,2,4,8]].tolist(),
                relation_target_mean=t.mean((0,1)).tolist(),contribution_variance=effects['phi'].var(0).tolist(),scale=scale,
                warnings=[])
    if t[...,1].mean()>.9:report['warnings'].append('relation targets almost all M; not balanced artificially')
    if effects['phi'].abs().max()<1e-6:report['warnings'].append('degenerate utility')
    if accuracy[15] <= max(float((y==0).float().mean()),float((y==1).float().mean())):
        report['warnings'].append('reference full-subset accuracy no better than majority prior')
    dump_json(report,directory/'reference_diagnostics.json')
    return target_path
