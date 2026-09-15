"""No GPU calls in scheduler preparation; test real FIFO launch/refill logic with mocks."""
import copy
from datetime import datetime,timedelta
import importlib.util
import json
from pathlib import Path
import signal
from unittest.mock import patch
import pytest

path=Path(__file__).parents[1]/'scripts/schedule_masked_r1.py'
spec=importlib.util.spec_from_file_location('r1_scheduler',path);scheduler=importlib.util.module_from_spec(spec);spec.loader.exec_module(scheduler)


def test_prepare_new_four_seeds_no_gpu_usage(tmp_path):
    root=tmp_path/'root';root.mkdir();(root/'workspaces/scheduled/old').mkdir(parents=True)
    (root/'workspaces/scheduled/old/plan.json').write_text(json.dumps({'seeds':[1,2,3,4]}))
    (root/'configs/revisions/v3_masked_r1').mkdir(parents=True)
    actual=Path(__file__).parents[1]
    for ds in ('gossipcop','weibo21','weibo'):
        (root/f'configs/revisions/v3_masked_r1/{ds}.json').write_text((actual/f'configs/revisions/v3_masked_r1/{ds}.json').read_text())
        ready=root/f'datasets/{ds}/ready';ready.mkdir(parents=True)
        for name in ('train.jsonl','val.jsonl','test.jsonl','dataset_manifest.json'):(ready/name).write_text('{}')
    with patch.object(scheduler,'ROOT',root),patch.object(scheduler,'code_files',return_value={}),patch.object(scheduler,'free_gpus',side_effect=AssertionError('GPU query before 01:00')):
        scheduler.prepare(tmp_path/'queue',(datetime.now(scheduler.TZ)+timedelta(hours=1)).isoformat())
    plan=json.loads((tmp_path/'queue/plan.json').read_text())
    assert len(plan['jobs'])==12 and len(set(plan['seeds']))==4 and not set(plan['seeds'])&{1,2,3,4}
    assert all(j['commands'][0][-2:]==['--stage','all'] and j['commands'][1][-3:]==['--stage','test','--frozen'] for j in plan['jobs'])


def test_queue_refills_and_does_not_claim_failed_seed(tmp_path):
    jobs=[dict(dataset='weibo',seed=i,commands=[],run_dir=str(tmp_path/f'run{i}'),log_path=str(tmp_path/f'{i}.log'),status='pending') for i in range(5)]
    plan=dict(start_at=(datetime.now(scheduler.TZ)-timedelta(seconds=1)).isoformat(),python='fake',datasets=['weibo'],
              code_sha256={},data_sha256={},config_sha256={},jobs=jobs)
    path=tmp_path/'plan.json';path.write_text(json.dumps(plan));(tmp_path/'status.json').write_text(json.dumps(dict(status='scheduled',jobs=jobs)))
    tick=[0];starts=[]
    class Process:
        def __init__(self,command,**kwargs):
            self.index=int(command[-1]);self.pid=900+self.index;self.started=tick[0];starts.append((self.index,tick[0],kwargs['env']['CUDA_VISIBLE_DEVICES']))
        def poll(self):
            # Seed zero frees its card while others remain active. Fifth must refill immediately.
            lifetime=1 if self.index==0 else 3
            return 1 if tick[0]-self.started>=lifetime else None
    def sleep(seconds):tick[0]+=1
    with patch.object(scheduler,'code_files',return_value={}),patch.object(scheduler,'free_gpus',return_value={i:f'GPU-{i}' for i in range(4)}),patch.object(scheduler.subprocess,'Popen',Process),patch.object(scheduler.time,'sleep',sleep):
        scheduler.run(path)
    assert [i for i,t,g in starts[:4]]==[0,1,2,3]
    assert starts[4]==(4,1,'GPU-0')
    state=json.loads((tmp_path/'status.json').read_text())
    assert state['status']=='completed_with_errors' and state['aggregates']['weibo']['status']=='INCOMPLETE'
