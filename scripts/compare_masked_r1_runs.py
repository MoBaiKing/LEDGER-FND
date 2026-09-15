#!/usr/bin/env python3
"""Paired seeds and sample/group bootstrap; do not flatten seeds into new news."""
import argparse,json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score


def paired_comparison(left,right,groups=None,seed=20260916,repetitions=2000,comparisons=1):
    if len(left)!=len(right) or not left:raise ValueError('paired seed lists required')
    ids=sorted(str(r['id']) for r in left[0]);ys=[];lp=[];rp=[]
    for l,r in zip(left,right):
        l={str(x['id']):x for x in l};r={str(x['id']):x for x in r}
        if sorted(l)!=ids or sorted(r)!=ids:raise ValueError('same news required for all paired seeds')
        y=np.array([l[i]['label'] for i in ids])
        if not np.array_equal(y,[r[i]['label'] for i in ids]):raise ValueError('labels disagree')
        ys.append(y);lp.append([l[i]['prediction'] for i in ids]);rp.append([r[i]['prediction'] for i in ids])
    if any(not np.array_equal(y,ys[0]) for y in ys):raise ValueError('same evaluation labels across seeds required')
    y=ys[0];lp=np.array(lp);rp=np.array(rp)
    def score(indices):
        return np.array([f1_score(y[indices],a[indices],labels=[0,1],average='macro',zero_division=0)-
                         f1_score(y[indices],b[indices],labels=[0,1],average='macro',zero_division=0) for a,b in zip(lp,rp)])
    keys=[str(groups[i]) if groups else i for i in ids];unique=sorted(set(keys));members={k:np.flatnonzero(np.array(keys)==k) for k in unique}
    rng=np.random.default_rng(seed);samples=[]
    for _ in range(repetitions):
        indices=np.concatenate([members[k] for k in rng.choice(unique,len(unique),replace=True)])
        # Reuse the same resampled news across every seed; seeds remain paired.
        samples.append(float(score(indices).mean()))
    alpha=.05/max(1,comparisons);delta=score(np.arange(len(ids)))
    return dict(paired_seed_differences=delta.tolist(),mean_difference=float(delta.mean()),
                seed_std=float(delta.std(ddof=1)) if len(delta)>1 else None,
                paired_bootstrap_interval=np.quantile(samples,[alpha/2,1-alpha/2]).tolist(),
                resampling_unit='provided group' if groups else 'news sample',seed=seed,repetitions=repetitions,
                multiple_comparisons='Bonferroni interval',comparisons=comparisons,
                limitation='Bootstrap conditional on these paired trained seeds; not independent news replicated by seed')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--left',nargs='+',required=True,type=Path);p.add_argument('--right',nargs='+',required=True,type=Path)
    p.add_argument('--groups',type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--seed',type=int,default=20260916)
    p.add_argument('--repetitions',type=int,default=2000);p.add_argument('--comparisons',type=int,default=1);a=p.parse_args()
    read=lambda path:[json.loads(l) for l in path.read_text().splitlines() if l]
    report=paired_comparison([read(x) for x in a.left],[read(x) for x in a.right],json.loads(a.groups.read_text()) if a.groups else None,a.seed,a.repetitions,a.comparisons)
    a.output.write_text(json.dumps(report,indent=2));print(a.output)


if __name__=='__main__':main()
