#!/usr/bin/env python3
"""Synthetic end-to-end CPU smoke using real tiny Qwen2 + SigLIP + PEFT APIs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
import torch
from PIL import Image
import numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Qwen2Config,Qwen2Model,SiglipConfig,SiglipModel,SiglipImageProcessor,PreTrainedTokenizerFast
from mmfnd.reference_pipeline import build_reference,build_targets_from_oof
from mmfnd.dataset_contract import bind_dataset_workspace
from mmfnd.utils import DistributedContext


def make_fixture(directory):
    directory.mkdir(parents=True,exist_ok=False)
    qpath=directory/'qwen';vpath=directory/'siglip';images=directory/'images';images.mkdir()
    qc=Qwen2Config(vocab_size=32,hidden_size=32,intermediate_size=48,num_hidden_layers=3,num_attention_heads=4,
                   num_key_value_heads=2,max_position_embeddings=64,attention_dropout=0.,use_cache=False)
    Qwen2Model(qc).save_pretrained(qpath)
    tokenizer=Tokenizer(WordLevel({'[UNK]':0,'[PAD]':1,'news':2,'true':3,'fake':4},unk_token='[UNK]'));tokenizer.pre_tokenizer=Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer,unk_token='[UNK]',pad_token='[PAD]').save_pretrained(qpath)
    vc=SiglipConfig(text_config=dict(vocab_size=32,hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,max_position_embeddings=32),
                    vision_config=dict(hidden_size=32,intermediate_size=48,num_hidden_layers=2,num_attention_heads=4,image_size=16,patch_size=4))
    SiglipModel(vc).save_pretrained(vpath);SiglipImageProcessor(size={'height':16,'width':16}).save_pretrained(vpath)
    rng=np.random.default_rng(13)
    for i in range(4):Image.fromarray(rng.integers(0,256,(20,20,3),dtype=np.uint8)).save(images/f'{i}.png')
    ready=directory/'ready';ready.mkdir()
    records={}
    for split,n in [('train',36),('val',8),('test',8)]:
        records[split]=[dict(id=f'{split}_{i}',text=f'news {split} {i} '+('fake' if i%2==0 else 'true'),label=i%2,
                              dataset='weibo21',images=[f'{i%4}.png',f'{(i+1)%4}.png']) for i in range(n)]
        (ready/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in records[split])+'\n')
    (ready/'dataset_manifest.json').write_text(json.dumps(dict(dataset='weibo21',schema_version='cute_fnd_multimodal_v1',label_semantics={'0':'fake','1':'real'},image_root=str(images))))
    c=json.loads((ROOT/'configs/revisions/v3_masked_r1/weibo21.json').read_text())
    c['seed']=991;c['data'].update(root=str(images),image_root=str(images),processed_dir=str(ready),replay_seed=991,max_images=2,max_text_length=16,num_workers=0)
    c['model'].update(text_backbone=str(qpath),vision_backbone=str(vpath),hidden_dim=16,num_heads=4,qwen_dtype='float32',dropout=0.,lora_rank=2,lora_alpha=4,lora_dropout=0.,unfreeze_vision_last_n=1)
    c['model']['r1'].update(head_hidden_dim=16,gradient_checkpointing=True)
    c['reference'].update(epochs=1,width=16)
    c['train'].update(epochs=2,per_gpu_batch_size=4,grad_accum_steps=2,precision='fp32',top_k_checkpoint_average=2,output_dir=str(directory/'runs'))
    bind_dataset_workspace(ROOT,c,'weibo21',str(ready))
    path=directory/'config.json';path.write_text(json.dumps(c,indent=2))
    return c,path,records,ready


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True,type=Path);p.add_argument('--ddp',type=int,default=0);a=p.parse_args()
    torch.set_num_threads(1);c,path,records,ready=make_fixture(a.output.resolve())
    ref=a.output.resolve()/'reference'
    if a.ddp:
        # Spawn reference directly with fixture manifest instead of production CLI dataset resolution.
        script=a.output.resolve()/'reference_worker.py'
        script.write_text('import sys\nfrom pathlib import Path\nsys.path.insert(0,'+repr(str(ROOT))+')\nfrom mmfnd.utils import init_distributed,cleanup_distributed,load_config\nfrom mmfnd.reference_pipeline import build_reference\nc=init_distributed()\ntry: build_reference(Path('+repr(str(ROOT))+'),load_config('+repr(str(path))+'),c,Path('+repr(str(ref))+'))\nfinally: cleanup_distributed()\n')
        subprocess.run([sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.ddp}',str(script)],check=True,cwd=ROOT)
    else:build_reference(ROOT,c,DistributedContext(0,1,0,torch.device('cpu')),ref)
    target=build_targets_from_oof(ref,records['train'])
    prefix=[sys.executable]
    if a.ddp:prefix+=['-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.ddp}']
    command=prefix+[str(ROOT/'train.py'),'--dataset','weibo21','--config',str(path),'--manifest-dir',str(ready),
                    '--r1-target-cache',str(target),'--run-name','synthetic_smoke']
    subprocess.run(command,check=True,cwd=ROOT)
    run=a.output.resolve()/'runs/synthetic_smoke';summary=json.loads((run/'final_summary.json').read_text())
    assert summary['test'].startswith('NOT RUN')
    assert not (run/'evaluation/final/test').exists()
    # Exact final weight reload and threshold check on val with no reference model loaded.
    subprocess.run([sys.executable,str(ROOT/'evaluate.py'),'--config',str(run/'config.json'),'--checkpoint',summary['checkpoint'],
                    '--manifest-dir',str(ready),'--split','val'],check=True,cwd=ROOT)
    print(json.dumps(dict(status='PASS',fixture='synthetic tiny Qwen2/SigLIP/PEFT',gpu_used=False,ddp_world_size=a.ddp or 1,output=str(run))))


if __name__=='__main__':main()
