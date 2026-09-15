#!/usr/bin/env python3
"""Encoder-level reference fitting and independent offline target construction."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from mmfnd.utils import load_config,init_distributed,cleanup_distributed
from mmfnd.dataset_contract import bind_dataset_workspace
from mmfnd.reference_pipeline import build_reference,build_targets_from_oof,prepare_identity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['weibo21','weibo','gossipcop'])
    p.add_argument('--config',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--seed',required=True,type=int);p.add_argument('--stage',choices=['reference','targets','preflight'],default='reference')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];c=load_config(a.config)
    c['seed']=a.seed;c['data']['replay_seed']=a.seed
    bind_dataset_workspace(root,c,a.dataset,c.get('dataset',{}).get('manifest_dir',f'datasets/{a.dataset}/ready'))
    if a.stage=='preflight':
        records,folds,fp=prepare_identity(root,c)
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'preflight.json').write_text(json.dumps(dict(samples=len(records),folds=folds,fingerprint=fp),indent=2))
        print(f'CPU preflight complete: {len(records)} originals; {a.output}')
        return
    if a.stage=='targets':
        records=[json.loads(x) for x in (Path(c['data']['processed_dir'])/'train.jsonl').read_text().splitlines() if x]
        print(build_targets_from_oof(a.output,records));return
    context=init_distributed()
    try:build_reference(root,c,context,a.output)
    finally:cleanup_distributed()


if __name__=='__main__':main()
