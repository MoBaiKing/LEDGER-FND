#!/usr/bin/env python3
"""Unified R1 entry. all = reference, targets, train, validation diagnostics; no test."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True,choices=['weibo21','weibo','gossipcop','gossip'])
    p.add_argument('--stage',required=True,choices=['preflight','reference','targets','train','diagnostics','test','all'])
    p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--seeds',type=int,nargs='+',help='Explicit matched seeds, run sequentially')
    p.add_argument('--nproc',type=int,default=1);p.add_argument('--config',type=Path)
    p.add_argument('--reference-dir',type=Path);p.add_argument('--run-name')
    p.add_argument('--checkpoint',type=Path);p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--frozen',action='store_true');p.add_argument('--smoke-steps',type=int,default=0)
    p.add_argument('--resume',type=Path)
    a=p.parse_args()
    if a.nproc<1:p.error('--nproc must be positive')
    if a.stage=='test' and not a.frozen:p.error('--stage test requires --frozen')
    if a.stage=='diagnostics' and a.split=='test' and not a.frozen:p.error('test diagnostics require --frozen')
    dataset='gossipcop' if a.dataset=='gossip' else a.dataset
    config=(a.config or ROOT/f'configs/revisions/v3_masked_r1/{dataset}.json').resolve()
    for seed in (a.seeds or [a.seed]):
        name=a.run_name or f'r1_{dataset}_seed{seed}'
        ref=(a.reference_dir or ROOT/f'workspaces/r1_reference/{dataset}/seed{seed}').resolve()
        cfg=json.loads(config.read_text())
        out=Path(cfg['train']['output_dir']);out=out if out.is_absolute() else ROOT/out
        run=out/name
        stages=(['train'] if cfg['model'].get('r1',{}).get('baseline') else ['reference','targets','train','diagnostics']) if a.stage=='all' else [a.stage]
        for stage in stages:
            prefix=[sys.executable]
            if a.nproc>1 and stage in ('reference','train'):
                prefix += ['-m','torch.distributed.run','--standalone',f'--nproc_per_node={a.nproc}']
            if stage in ('reference','targets','preflight'):
                command=prefix+[str(ROOT/'scripts/build_masked_r1_targets.py'),'--dataset',dataset,'--config',str(config),
                                '--output',str(ref),'--seed',str(seed),'--stage',stage]
            elif stage=='train':
                command=prefix+[str(ROOT/'train.py'),'--dataset',dataset,'--config',str(config),'--manifest-dir',
                                str(cfg.get('dataset',{}).get('manifest_dir',ROOT/f'datasets/{dataset}/ready')),'--seed',str(seed),'--run-name',name,
                                '--r1-target-cache',str(ref/'targets.pt')]
                if a.smoke_steps:command+=['--smoke-steps',str(a.smoke_steps)]
                if a.resume:command+=['--resume',str(a.resume)]
            else:
                checkpoint=a.checkpoint or run/'checkpoints/final_averaged.pth'
                script='evaluate.py' if stage=='test' else 'scripts/evaluate_masked_r1_mechanisms.py'
                command=prefix+[str(ROOT/script),'--config',str(run/'config.json'),'--checkpoint',str(checkpoint),
                                '--split','test' if stage=='test' else a.split]
                if stage=='diagnostics':command+=['--reference-dir',str(ref)]
                else:command+=['--manifest-dir',str(cfg.get('dataset',{}).get('manifest_dir',ROOT/f'datasets/{dataset}/ready'))]
                if a.frozen:command+=['--frozen']
            print('RUN',json.dumps(command),flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
            if a.smoke_steps and stage=='train':break


if __name__=='__main__':main()
