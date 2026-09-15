#!/usr/bin/env python3
"""Train on an explicitly frozen view cache; same inputs/budget for pretrained/random judge."""
import argparse,json,time
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from transformers import AutoModel
from mmfnd.frozen_judge_controls import FrozenInputJudge,fixed_role_permutations,shuffle_cached_views
from mmfnd.losses_masked_r1 import masked_r1_loss
from mmfnd.evaluation import find_best_macro_f1_threshold,threshold_grid,compute_classification_metrics
from mmfnd.utils import seed_everything,get_device,load_config,dump_json
from mmfnd.engine import autocast_context
from mmfnd.r1_cache import file_sha


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',required=True,type=Path)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--judge-init',choices=['pretrained','random'],required=True);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--epochs',type=int,default=3);p.add_argument('--eval-shuffle',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False);c=load_config(a.config)
    cache=torch.load(a.cache,map_location='cpu',weights_only=False)
    if cache.get('protocol')!='r1_fixed_views_v1' or cache.get('splits')!=['train','val']:
        raise ValueError('expected labeled train/val-only frozen cache; test is not accepted')
    for split in ('train','val'):
        if cache[split]['z'].requires_grad:raise ValueError('frozen inputs required')
    seed_everything(a.seed);device=get_device();start=time.monotonic()
    qwen=AutoModel.from_pretrained(c['model']['text_backbone'],local_files_only=True,torch_dtype=torch.bfloat16 if device.type=='cuda' else torch.float32)
    model=FrozenInputJudge(qwen,c['model']['hidden_dim'],c['model']['r1'],a.judge_init=='random');del qwen
    # Full selected judge-layer training in BOTH controls; no claim of random frozen adapters.
    model=model.to(device);optimizer=torch.optim.AdamW(model.parameters(),lr=c['train']['head_learning_rate'],weight_decay=c['train']['weight_decay'])
    batch_size=c['train']['per_gpu_batch_size'];best=-1.;history=[]
    for epoch in range(1,a.epochs+1):
        model.train();order=torch.randperm(len(cache['train']['z']))
        for indices in order.split(batch_size):
            data=cache['train'];z=data['z'][indices].to(device);available=data['availability'][indices].to(device)
            with autocast_context(device,c['train']['precision']):
                out=model(z,available)
                loss,_=masked_r1_loss(out,data['labels'][indices].to(device),data['q'][indices].to(device),cache['scale'],c['loss'],epoch)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),c['train']['gradient_clip_norm']);optimizer.step()
        model.eval();probs=[];data=cache['val']
        with torch.no_grad():
            for indices in torch.arange(len(data['z'])).split(batch_size):
                with autocast_context(device,c['train']['precision']):out=model(data['z'][indices].to(device),data['availability'][indices].to(device))
                probs.append(out['logits'].float().softmax(-1).cpu())
        probs=torch.cat(probs).numpy();labels=data['labels'].numpy()
        selection=find_best_macro_f1_threshold(labels,probs[:,0],threshold_grid(c['train']),split='val')
        history.append(dict(epoch=epoch,selection=selection))
        if selection['macro_f1']>best:
            best=selection['macro_f1'];torch.save(dict(state=model.state_dict(),selection=selection,cache_sha256=file_sha(a.cache)),a.output/'best.pt')
    result=dict(judge_init=a.judge_init,scope='full selected two judge layers + heads; fixed external Z',history=history,
                seconds=time.monotonic()-start,cache_sha256=file_sha(a.cache),total_parameters=sum(p.numel() for p in model.parameters()),
                limitation='Controlled frozen-input comparison; not the main shared end-to-end model',test='NOT RUN')
    if a.eval_shuffle:
        saved=torch.load(a.output/'best.pt',map_location=device,weights_only=False);model.load_state_dict(saved['state'],strict=True)
        data=cache['val'];permutation=fixed_role_permutations(data['ids'],a.seed);z=shuffle_cached_views(data['z'],data['ids'],permutation)
        probs=[]
        with torch.no_grad():
            for indices in torch.arange(len(z)).split(batch_size):
                with autocast_context(device,c['train']['precision']):out=model(z[indices].to(device),data['availability'][indices].to(device))
                probs.append(out['logits'].float().softmax(-1).cpu())
        result['eval_only_shuffle']=compute_classification_metrics(data['labels'].numpy(),torch.cat(probs).numpy(),saved['selection']['threshold'])
        dump_json(permutation,a.output/'role_permutation.json')
    dump_json(result,a.output/'report.json');print(a.output/'report.json')


if __name__=='__main__':main()
