#!/usr/bin/env python3
"""Write label-free student predictions first, then run separate labeled diagnostics."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from mmfnd.factory import build_model,build_loader,build_processor
from mmfnd.engine import load_checkpoint,write_jsonl
from mmfnd.evaluation import checkpoint_threshold_selection
from mmfnd.evaluation_masked_r1 import predict_student
from mmfnd.mechanisms_masked_r1 import mechanism_report
from mmfnd.utils import load_config,get_device,dump_json
from mmfnd.dataset_contract import bind_dataset_workspace
from mmfnd.r1_cache import file_sha


def export_test_reference(root,config,ref,loader,device):
    from mmfnd.reference_pipeline import ReferenceModel,predict_reference,label_free_batches
    import gc
    folds=[]
    for path in sorted(ref.glob('fold_*.pth')):
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=ReferenceModel(config).to(device)
        names={n for n,p in model.named_parameters() if p.requires_grad}
        if names!=set(saved['state']):raise ValueError('reference checkpoint parameter contract mismatch')
        state=model.state_dict();state.update(saved['state']);model.load_state_dict(state,strict=True)
        folds.append(predict_reference(model,label_free_batches(loader),device,config['train']['precision'],saved['prior'],saved['temperature']))
        del model,saved,state;gc.collect()
        if device.type=='cuda':torch.cuda.empty_cache()
    if len(folds)!=config['reference']['folds']:raise ValueError('missing reference fold checkpoints')
    rows=folds[0]
    for i,row in enumerate(rows):
        if any(f[i]['sample_id']!=row['sample_id'] for f in folds):raise ValueError('reference ensemble ID mismatch')
        row['q']=torch.stack([f[i]['q'] for f in folds]).mean(0)
    torch.save(dict(rows=rows,split='test',labels_read=False,fingerprint=config['r1_reference_fingerprint']),ref/'test_probabilities.pt')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--reference-dir',required=True,type=Path);p.add_argument('--split',choices=['val','test'],default='val')
    p.add_argument('--frozen',action='store_true');p.add_argument('--output',type=Path)
    p.add_argument('--robustness',choices=['none','gaussian','typo'],default='none')
    p.add_argument('--gaussian-sigma',type=float,default=0.);p.add_argument('--typo-rate',type=float,default=0.)
    p.add_argument('--corruption-seed',type=int,default=2027)
    a=p.parse_args()
    if a.split=='test' and not a.frozen:p.error('test requires --frozen')
    root=Path(__file__).resolve().parents[1];c=load_config(a.config);ds=c['dataset']['name']
    bind_dataset_workspace(root,c,ds,c.get('dataset',{}).get('manifest_dir',f'datasets/{ds}/ready'))
    from mmfnd.r1_cache import verify_backbone_identity
    verify_backbone_identity(root,c)
    device=get_device();processor=build_processor(root,c);loader=build_loader(root,c,a.split,processor)
    if a.robustness!='none':
        from robustness.pipeline import build_robustness_loader
        from robustness.corruptions import CorruptionSpec
        if a.split!='test':p.error('existing robustness protocol is restricted to frozen test')
        loader=build_robustness_loader(root,c,a.split,processor,CorruptionSpec(a.robustness,a.gaussian_sigma,a.typo_rate,a.corruption_seed))
    model=build_model(c).to(device);checkpoint=load_checkpoint(a.checkpoint,model,device)
    selection=checkpoint_threshold_selection(checkpoint)
    rows,timing=predict_student(model,loader,device,c['train']['precision'],deletions=True)
    output=a.output or a.checkpoint.parent.parent/'mechanisms'/a.split/a.robustness
    output.mkdir(parents=True,exist_ok=False)
    write_jsonl(rows,output/'student_predictions_unlabeled.jsonl')
    # Student GPU memory must be released before loading any reference.
    del model,checkpoint
    import gc;gc.collect()
    if device.type=='cuda':torch.cuda.empty_cache()
    ref_path=a.reference_dir/f'{a.split}_probabilities.pt'
    if not ref_path.exists() and a.split=='test':
        export_test_reference(root,c,a.reference_dir,build_loader(root,c,'test',processor),device)
    reference=torch.load(ref_path,map_location='cpu',weights_only=False)
    if reference.get('fingerprint') != c.get('r1_reference_fingerprint'):
        raise ValueError('diagnostic reference and student input/seed fingerprint mismatch')
    cache_info=json.loads((a.reference_dir/'targets.json').read_text())
    labels={str(r['id']):int(r['label']) for r in loader.dataset.records}
    report=mechanism_report(rows,reference['rows'],labels,selection['threshold'],cache_info['scale'])
    report.update(**timing,checkpoint_sha256=file_sha(a.checkpoint),threshold=selection['threshold'],split=a.split,
                  robustness=a.robustness,gaussian_sigma=a.gaussian_sigma,typo_rate=a.typo_rate,corruption_seed=a.corruption_seed,
                  reference_grouping='clean independent reference; corruption results are sensitivity only',
                  typo_limitations='Existing token-insertion protocol does not guarantee entity or label preservation; raw inputs are re-encoded including E/X')
    dump_json(report,output/'report.json');print(output/'report.json')


if __name__=='__main__':main()
