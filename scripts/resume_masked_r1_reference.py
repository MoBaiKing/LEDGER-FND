#!/usr/bin/env python3
"""Regenerate reference exports from completed fold checkpoints without fitting."""
import argparse
import gc
import json
import time
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from mmfnd.dataset_contract import bind_dataset_workspace
from mmfnd.engine import unwrap_model
from mmfnd.factory import build_loader,build_processor
from mmfnd.reference_pipeline import (ReferenceModel,calibration_logits,label_free_batches,
    partition_loader,predict_reference,prepare_identity)
from mmfnd.reference_subset_probe import smoothed_prior,subset_ce
from mmfnd.r1_cache import digest,file_sha
from mmfnd.utils import cleanup_distributed,dump_json,init_distributed,load_config,seed_everything


def verify(directory,fingerprint,folds):
    manifest=json.loads((directory/'reference_manifest.json').read_text())
    if manifest['fingerprint'] != fingerprint:raise ValueError('paused reference fingerprint mismatch')
    saved=[]
    for fold in folds['folds']:
        path=directory/f"fold_{fold['fold']}.pth"
        checkpoint=torch.load(path,map_location='cpu',weights_only=False)
        if checkpoint['fingerprint'] != fingerprint or checkpoint['fold'] != fold:
            raise ValueError(f'fold {fold["fold"]} checkpoint contract mismatch')
        saved.append(checkpoint)
    return saved


def resume_exports(root,config,context,directory,verify_only=False):
    if context.distributed:raise ValueError('reference export resume requires nproc=1')
    directory=Path(directory);started=time.monotonic()
    records,folds,fingerprint=prepare_identity(root,config)
    checkpoints=verify(directory,fingerprint,folds)
    if verify_only:
        print(json.dumps({'status':'verified','folds':len(checkpoints),'fingerprint':digest(fingerprint)}));return
    processor=build_processor(root,config)
    dataset=build_loader(root,config,'train',processor,shuffle=True).dataset
    labels_by_id={str(r['id']):r['label'] for r in dataset.records}
    all_rows=[];val_folds=[];costs=[]
    for fold,saved in zip(folds['folds'],checkpoints):
        fold_start=time.monotonic();seed_everything(int(config['seed']))
        model=ReferenceModel(config).to(context.device);raw_model=unwrap_model(model)
        trainable={n for n,p in raw_model.named_parameters() if p.requires_grad}
        if set(saved['state']) != trainable or set(saved['trainable_names']) != trainable:
            raise ValueError(f'fold {fold["fold"]} trainable key mismatch')
        state=raw_model.state_dict();state.update(saved['state']);raw_model.load_state_dict(state,strict=True);del state
        cal_loader=partition_loader(dataset,fold['cal_ids'],config,context)
        target_loader=partition_loader(dataset,fold['target_ids'],config,context)
        prior=smoothed_prior([labels_by_id[i] for i in fold['fit_ids']],float(config['reference'].get('prior_smoothing',1)))
        if not torch.equal(saved['prior'].cpu(),prior.cpu()):raise ValueError('saved reference prior mismatch')
        temperature=saved.get('temperature')
        if temperature is None:
            logits,y,a=calibration_logits(raw_model,cal_loader,context.device,config['train']['precision'])
            candidates=config['reference'].get('temperature_grid',[.5,1.,2.,4.])
            temperature=min(candidates,key=lambda t:(float(subset_ce(logits/float(t),y,a)),abs(float(t)-1)))
            saved['temperature']=temperature;torch.save(saved,directory/f"fold_{fold['fold']}.pth")
        checkpoint_hash=file_sha(directory/f"fold_{fold['fold']}.pth")
        oof_path=directory/f"oof_{fold['fold']}.pt"
        if oof_path.is_file():
            payload=torch.load(oof_path,map_location='cpu',weights_only=False)
            if payload['fingerprint'] != fingerprint:raise ValueError('saved OOF fingerprint mismatch')
            rows=payload['rows']
            if {r['sample_id'] for r in rows} != set(fold['target_ids']):raise ValueError('saved OOF coverage mismatch')
            if any(r['reference_checkpoint_sha256'] != checkpoint_hash for r in rows):raise ValueError('saved OOF checkpoint hash mismatch')
        else:
            rows=predict_reference(raw_model,label_free_batches(target_loader),context.device,config['train']['precision'],prior,temperature)
            for row in rows:row.update(fold=fold['fold'],fit_id_hash=fold['fit_id_hash'],cal_id_hash=fold['cal_id_hash'],reference_checkpoint_sha256=checkpoint_hash)
            torch.save(dict(fingerprint=fingerprint,rows=rows),oof_path)
        all_rows.extend(rows)
        val_loader=build_loader(root,config,'val',processor)
        val_folds.append(predict_reference(raw_model,label_free_batches(val_loader),context.device,config['train']['precision'],prior,temperature))
        costs.append(dict(fold=fold['fold'],seconds=time.monotonic()-fold_start,optimizer_steps=0,resumed_export=True,
                          checkpoint_epoch=int(saved['epoch']),temperature=temperature,
                          total_parameters=sum(p.numel() for p in raw_model.parameters()),
                          trainable_parameters=sum(p.numel() for p in raw_model.parameters() if p.requires_grad),
                          peak_cuda_bytes=torch.cuda.max_memory_allocated() if context.device.type=='cuda' else 0))
        del model,raw_model,saved;gc.collect()
        if context.device.type=='cuda':torch.cuda.empty_cache()
    torch.save(dict(fingerprint=fingerprint,rows=all_rows),directory/'oof_probabilities.pt')
    val_rows=val_folds[0]
    for i,row in enumerate(val_rows):
        if any(values[i]['sample_id']!=row['sample_id'] for values in val_folds):raise ValueError('val ensemble ID mismatch')
        row['q']=torch.stack([values[i]['q'] for values in val_folds]).mean(0)
    torch.save(dict(fingerprint=fingerprint,rows=val_rows,split='val',labels_read=False),directory/'val_probabilities.pt')
    dump_json(dict(fingerprint=fingerprint,status='probabilities_complete_resumed',config=config,costs=costs,
                   optimization_repeated=False,total_seconds=time.monotonic()-started),directory/'reference_manifest.json')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,choices=['weibo21','weibo','gossipcop'])
    parser.add_argument('--config',required=True,type=Path);parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--seed',required=True,type=int);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args();config=load_config(args.config);config['seed']=args.seed;config['data']['replay_seed']=args.seed
    bind_dataset_workspace(ROOT,config,args.dataset,config.get('dataset',{}).get('manifest_dir',f'datasets/{args.dataset}/ready'))
    context=init_distributed()
    try:resume_exports(ROOT,config,context,args.output,args.verify_only)
    finally:cleanup_distributed()


if __name__=='__main__':main()
