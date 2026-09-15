#!/usr/bin/env python3
"""One-shot 01:00 FIFO, one complete reference/student seed pipeline per GPU."""
import argparse
from datetime import datetime,timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
ROOT=Path(__file__).resolve().parents[1]
TZ=ZoneInfo('Asia/Shanghai')


def now():return datetime.now(TZ).isoformat(timespec='seconds')


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def code_files():
    files=[ROOT/'train.py',ROOT/'evaluate.py']
    for folder in ('mmfnd','scripts','configs/revisions/v3_masked_r1'):
        files += [p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.py','.sh','.json')]
    return {str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}


def prepare(directory,start_at):
    start=datetime.fromisoformat(start_at)
    if start.tzinfo is None:start=start.replace(tzinfo=TZ)
    if start<=datetime.now(TZ):raise ValueError('start time must be in the future')
    directory.mkdir(parents=True,exist_ok=False);(directory/'configs').mkdir()
    used=set()
    for p in (ROOT/'workspaces/scheduled').glob('*/plan.json'):
        try:used.update(json.loads(p.read_text()).get('seeds',[]))
        except (ValueError,OSError):pass
    seeds=[]
    while len(seeds)<4:
        value=secrets.randbelow(2**31-1)+1
        if value not in used and value not in seeds:seeds.append(value)
    plan=dict(created_at=now(),start_at=start.isoformat(),project=str(ROOT),python=str(ROOT/'.venv/bin/python'),
              seeds=seeds,seed_source='OS secrets; distinct from previous local suite seeds',
              gpu_indices=[0,1,2,3],policy='one seed per free GPU, FIFO refill; three folds sequential within each job',
              status='prepared',datasets=['gossipcop','weibo21','weibo'],jobs=[],code_sha256=code_files(),data_sha256={})
    for ds in plan['datasets']:
        c=json.loads((ROOT/f'configs/revisions/v3_masked_r1/{ds}.json').read_text())
        # Preserve the default four-rank effective batch on a single GPU/seed.
        c['train']['grad_accum_steps']*=4
        c['protocol_notes']={'scheduled_world_size':1,'matched_reference_world_size':4,
                            'effective_batch_preserved_from_original_four_rank_config':True}
        cp=directory/'configs'/f'{ds}.json';dump(cp,c)
        for file in ('train.jsonl','val.jsonl','test.jsonl','dataset_manifest.json'):
            p=ROOT/f'datasets/{ds}/ready/{file}';plan['data_sha256'][str(p)]=sha(p)
        for seed in seeds:
            name=f'r1_strict_{directory.name}_{ds}_seed{seed}'
            run=ROOT/c['train']['output_dir']/name
            ref=directory/'references'/ds/f'seed{seed}'
            common=['bash',str(ROOT/'scripts/run_masked_r1.sh'),'--dataset',ds,'--seed',str(seed),'--nproc','1','--config',str(cp),'--run-name',name,'--reference-dir',str(ref)]
            commands=[common+['--stage','all'],common+['--stage','test','--frozen']]
            plan['jobs'].append(dict(dataset=ds,seed=seed,run_name=name,run_dir=str(run),reference_dir=str(ref),
                                      commands=commands,log_path=str(directory/f'{ds}_seed{seed}.log'),status='pending'))
    plan['config_sha256']={str(p):sha(p) for p in (directory/'configs').glob('*.json')}
    dump(directory/'plan.json',plan)
    dump(directory/'status.json',dict(status='scheduled',start_at=plan['start_at'],jobs=plan['jobs'],updated_at=now()))
    print(directory/'plan.json')


def free_gpus():
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
    ids={int(row.split(',')[0]):row.split(',')[1].strip() for row in inventory.splitlines() if ',' in row and row.split(',')[0].strip().isdigit()}
    if not all(i in ids for i in range(4)):raise RuntimeError('four expected GPU indices are unavailable')
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if line.startswith('GPU-')}
    return {i:ids[i] for i in range(4) if ids[i] not in busy}


def pipeline(job,env):
    # Child supervisor preserves exact stage exit status; no shell interpolation.
    for command in job['commands']:
        result=subprocess.run(command,cwd=ROOT,env=env)
        if result.returncode:return result.returncode
    run=Path(job['run_dir'])
    summary=json.loads((run/'final_summary.json').read_text())
    candidates=list((run.parent/'manual_evaluation').glob('*/final_averaged/test/metrics.json'))
    expected=summary['val']['threshold_selection']['model_hash']
    matched=[p for p in candidates if json.loads(p.read_text())['threshold_selection'].get('model_hash')==expected]
    if len(matched)!=1:raise ValueError('frozen final test result must match exactly one model hash')
    dump(run/'scheduled_completion.json',dict(status='completed',finished_at=now(),test_metrics_path=str(matched[0]),
                                              test=json.loads(matched[0].read_text()),checkpoint_sha256=summary['checkpoint_sha256']))
    return 0


def run(path):
    path=path.resolve();directory=path.parent;plan=json.loads(path.read_text())
    with (directory/'scheduler.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status_path=directory/'status.json';state=json.loads(status_path.read_text())
        if state['status']!='scheduled':raise ValueError('one-shot scheduler already started')
        state.update(scheduler_pid=os.getpid(),scheduler_started_at=now());dump(status_path,state)
        launch=datetime.fromisoformat(plan['start_at']);active={};stopped=False
        def stop(signum,frame):
            nonlocal stopped;stopped=True
        for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
        try:
            while datetime.now(TZ)<launch and not stopped:
                state.update(updated_at=now(),active_gpu_jobs=0,seconds_to_start=int((launch-datetime.now(TZ)).total_seconds()))
                dump(status_path,state);time.sleep(min(30,max(.1,(launch-datetime.now(TZ)).total_seconds())))
            if stopped:raise InterruptedError('timer cancelled')
            if plan['code_sha256']!=code_files():raise ValueError('code changed after scheduling; fail closed')
            for file,expected in {**plan['data_sha256'],**plan['config_sha256']}.items():
                if sha(Path(file))!=expected:raise ValueError(f'input changed after scheduling: {file}')
            state.update(status='running',started_at=now())
            while not stopped:
                for gpu,(process,job,handle) in list(active.items()):
                    code=process.poll()
                    if code is None:continue
                    handle.close();job.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=now());del active[gpu]
                pending=[j for j in state['jobs'] if j['status']=='pending']
                if not pending and not active:break
                try:free=free_gpus();state.pop('probe_error',None)
                except Exception as e:free={};state['probe_error']=str(e)
                for gpu,uuid in free.items():
                    if gpu in active or not pending:continue
                    job=pending.pop(0)
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',
                             OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',
                             TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED=str(job['seed']))
                    handle=open(job['log_path'],'x')
                    child=[plan['python'],str(Path(__file__).resolve()),'--run-job',str(path),'--job-index',str(state['jobs'].index(job))]
                    process=subprocess.Popen(child,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                    job.update(status='running',gpu=gpu,gpu_uuid=uuid,pid=process.pid,started_at=now());active[gpu]=(process,job,handle)
                state.update(updated_at=now(),active_gpu_jobs=len(active),pending_jobs=sum(j['status']=='pending' for j in state['jobs']))
                dump(status_path,state);time.sleep(5)
            if stopped:raise InterruptedError('queue cancelled')
            aggregates={}
            for ds in plan['datasets']:
                jobs=[j for j in state['jobs'] if j['dataset']==ds]
                if any(j['status']!='completed' for j in jobs):aggregates[ds]={'status':'INCOMPLETE'};continue
                import statistics
                results=[json.loads((Path(j['run_dir'])/'scheduled_completion.json').read_text())['test'] for j in jobs]
                metrics={k:{'mean':statistics.mean(r[k] for r in results),'sample_std':statistics.stdev(r[k] for r in results)} for k in ('macro_f1','accuracy','auc','nll','brier','ece')}
                aggregates[ds]=dict(status='completed',seeds=plan['seeds'],metrics=metrics)
            state.update(status='completed' if all(j['status']=='completed' for j in state['jobs']) else 'completed_with_errors',aggregates=aggregates,finished_at=now())
            dump(status_path,state)
        except BaseException as error:
            for process,job,handle in active.values():
                if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
                job['status']='cancelled';handle.close()
            state.update(status='cancelled' if stopped else 'failed',error=str(error),finished_at=now());dump(status_path,state);raise


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',type=Path);g.add_argument('--run-plan',type=Path);g.add_argument('--run-job',type=Path)
    p.add_argument('--start-at');p.add_argument('--job-index',type=int);a=p.parse_args()
    if a.prepare:
        if not a.start_at:p.error('--prepare requires --start-at')
        prepare(a.prepare.resolve(),a.start_at)
    elif a.run_plan:run(a.run_plan)
    else:sys.exit(pipeline(json.loads(a.run_job.read_text())['jobs'][a.job_index],dict(os.environ)))


if __name__=='__main__':main()
