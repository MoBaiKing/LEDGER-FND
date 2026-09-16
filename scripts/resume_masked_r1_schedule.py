#!/usr/bin/env python3
"""Resume the paused R1 queue without repeating completed reference optimization."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
TZ=ZoneInfo('Asia/Shanghai')
STUDENT_BATCH=8
STUDENT_ACCUMULATION=4


def now():return datetime.now(TZ).isoformat(timespec='seconds')


def dump(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def free_gpus():
    inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits'],text=True,timeout=15)
    ids={int(row.split(',')[0]):row.split(',')[1].strip() for row in inventory.splitlines() if ',' in row and row.split(',')[0].strip().isdigit()}
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=15)
    busy={line.split(',')[0].strip() for line in apps.splitlines() if line.startswith('GPU-')}
    return {i:ids[i] for i in range(4) if i in ids and ids[i] not in busy}


def common(job):
    plan_dir=Path(job['reference_dir']).parents[2]
    return ['bash',str(ROOT/'scripts/run_masked_r1.sh'),'--dataset',job['dataset'],'--seed',str(job['seed']),
            '--nproc','1','--config',str(plan_dir/'configs'/f"{job['dataset']}.json"),'--run-name',job['run_name'],
            '--reference-dir',job['reference_dir'],'--per-gpu-batch-size',str(STUDENT_BATCH),
            '--grad-accum-steps',str(STUDENT_ACCUMULATION)]


def commands(job):
    base=common(job);ref=Path(job['reference_dir']);run=Path(job['run_dir'])
    result=[]
    if not (ref/'targets.pt').is_file():
        if not all((ref/f'fold_{i}.pth').is_file() for i in range(3)):
            result.append(base+['--stage','all'])
            return result+[base+['--stage','test','--frozen']]
        result.extend([[sys.executable,str(ROOT/'scripts/resume_masked_r1_reference.py'),'--dataset',job['dataset'],
                        '--config',base[base.index('--config')+1],'--output',str(ref),'--seed',str(job['seed'])],
                       base+['--stage','targets']])
    if not (run/'final_summary.json').is_file():
        train=base+['--stage','train']
        last=run/'checkpoints/last.pth'
        if last.is_file():train+=['--resume',str(last)]
        result.append(train)
    result.append(base+['--stage','diagnostics'])
    result.append(base+['--stage','test','--frozen'])
    return result


def pipeline(job):
    for command in commands(job):
        print('RESUME_RUN',json.dumps(command),flush=True)
        code=subprocess.run(command,cwd=ROOT).returncode
        if code:return code
    return 0


def run_job(status_path,index):
    state=json.loads(status_path.read_text());sys.exit(pipeline(state['jobs'][index]))


def run(status_path,start_at):
    state=json.loads(status_path.read_text())
    if state['status']!='paused':raise ValueError('resume scheduler requires paused queue status')
    launch=datetime.fromisoformat(start_at)
    if launch.tzinfo is None:launch=launch.replace(tzinfo=TZ)
    stopped=False;active={}
    def stop(signum,frame):
        nonlocal stopped;stopped=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,stop)
    state.update(status='resume_scheduled',resume_start_at=launch.isoformat(),resume_scheduler_pid=os.getpid(),updated_at=now())
    dump(status_path,state)
    while datetime.now(TZ)<launch and not stopped:time.sleep(min(5,max(.1,(launch-datetime.now(TZ)).total_seconds())))
    if stopped:return
    for job in state['jobs']:
        if job['status']=='paused':job['status']='pending'
    state.update(status='running',resumed_at=now(),error=None,finished_at=None)
    try:
        while not stopped:
            for gpu,(process,job,handle) in list(active.items()):
                code=process.poll()
                if code is None:continue
                handle.close();job.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=now());del active[gpu]
            pending=[j for j in state['jobs'] if j['status']=='pending']
            if not pending and not active:break
            try:free=free_gpus();state.pop('probe_error',None)
            except Exception as error:free={};state['probe_error']=str(error)
            for gpu,uuid in free.items():
                if gpu in active or not pending:continue
                job=pending.pop(0);index=state['jobs'].index(job)
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,WORLD_SIZE='1',RANK='0',LOCAL_RANK='0',OMP_NUM_THREADS='4',
                         MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1',TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',
                         TRANSFORMERS_OFFLINE='1',PYTHONHASHSEED=str(job['seed']))
                handle=open(job['log_path'],'a')
                child=[sys.executable,str(Path(__file__).resolve()),'--status',str(status_path),'--run-job','--job-index',str(index)]
                process=subprocess.Popen(child,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=handle,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(status='running',gpu=gpu,gpu_uuid=uuid,pid=process.pid,resumed_job_at=now());active[gpu]=(process,job,handle)
            state.update(updated_at=now(),active_gpu_jobs=len(active),pending_jobs=sum(j['status']=='pending' for j in state['jobs']))
            dump(status_path,state);time.sleep(5)
        if stopped:raise InterruptedError('resumed queue paused by signal')
        state.update(status='completed' if all(j['status']=='completed' for j in state['jobs']) else 'completed_with_errors',finished_at=now())
        dump(status_path,state)
    except BaseException as error:
        for process,job,handle in active.values():
            if process.poll() is None:os.killpg(process.pid,signal.SIGTERM)
            job['status']='paused';handle.close()
        state.update(status='paused' if stopped else 'failed',error=str(error),active_gpu_jobs=0,finished_at=now())
        dump(status_path,state);raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--status',required=True,type=Path);parser.add_argument('--start-at')
    parser.add_argument('--run-job',action='store_true');parser.add_argument('--job-index',type=int)
    args=parser.parse_args();status=args.status.resolve()
    if args.run_job:
        if args.job_index is None:parser.error('--run-job requires --job-index')
        run_job(status,args.job_index)
    else:
        if not args.start_at:parser.error('--start-at is required')
        run(status,args.start_at)


if __name__=='__main__':main()
