#!/usr/bin/env python3
"""Export shared frozen input cache for conditional pretrained/random judge controls."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mmfnd.factory import build_model,build_loader,build_processor
from mmfnd.engine import load_checkpoint,autocast_context
from mmfnd.r1_cache import file_sha,verify_backbone_identity,TargetCache
from mmfnd.utils import load_config,get_device


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True,type=Path)
    p.add_argument('--checkpoint',required=True,type=Path);p.add_argument('--reference-dir',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--encoder-selection',choices=['predeclared_train_epoch'],required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[1];c=load_config(a.config);verify_backbone_identity(root,c)
    device=get_device();model=build_model(c).to(device);load_checkpoint(a.checkpoint,model,device);model.eval()
    processor=build_processor(root,c);target=TargetCache(a.reference_dir/'targets.pt',c['r1_reference_fingerprint'])
    result=dict(protocol='r1_fixed_views_v1',splits=['train','val'],scale=target.scale,
                encoder_checkpoint_sha256=file_sha(a.checkpoint),encoder_selection=a.encoder_selection,
                config=c,reference_fingerprint=c['r1_reference_fingerprint'])
    for split in result['splits']:
        loader=build_loader(root,c,split,processor,shuffle=split=='train')
        reference={str(r['sample_id']):r for r in torch.load(a.reference_dir/'val_probabilities.pt',map_location='cpu',weights_only=False)['rows']} if split=='val' else None
        values={k:[] for k in ('z','availability','labels','q','ids')}
        with torch.no_grad():
            for batch in loader:
                inputs={k:batch[k].to(device) for k in ('text_input_ids','text_attention_mask','pixel_values','image_owner')}
                with autocast_context(device,c['train']['precision']):z=model.encode_views(inputs)
                q=(target.lookup(batch['ids'],batch['augmentation_ids'],batch['availability'],'cpu') if split=='train' else
                   torch.stack([reference[str(i)]['q'] for i in batch['ids']]))
                values['z'].append(z.float().cpu());values['availability'].append(batch['availability']);values['labels'].append(batch['labels']);values['q'].append(q)
                values['ids'].extend(f'{sid}:{aid}' for sid,aid in zip(batch['ids'],batch['augmentation_ids']))
        result[split]={k:torch.cat(v) if k!='ids' else v for k,v in values.items()}
    if a.output.exists():raise FileExistsError(a.output)
    torch.save(result,a.output);print(a.output)


if __name__=='__main__':main()
