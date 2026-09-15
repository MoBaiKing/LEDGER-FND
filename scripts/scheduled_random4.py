#!/usr/bin/env python3
"""One-shot server timer and FIFO GPU queue for v3 experiments (stdlib only).

Prepare a manifest now, then run it inside a detached tmux session. Four seeds
share a single FIFO: a completed job immediately frees its GPU for the next
dataset, even if other seeds of the previous dataset are still running.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import signal
import subprocess
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TZ = ZoneInfo("Asia/Shanghai")
DATASETS = ("weibo21", "gossipcop", "weibo")
BATCHES = {"weibo21": 16, "gossipcop": 32, "weibo": 16}
PYTHON = "/data/dyl/sotamodelv5/.venv/bin/python"


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def log(message):
    print(f"[{now()}] {message}", flush=True)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def query_gpus():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    return {int(fields[0]): fields[1].strip() for line in result.stdout.splitlines()
            if (fields := line.split(",")) and len(fields) == 2 and fields[0].strip().isdigit()}


def busy_gpus(gpu_uuids):
    """Use GPU UUIDs/processes; HAMI aggregate memory counters can be misleading."""
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True, timeout=15,
    )
    occupied = {line.split(",")[0].strip() for line in result.stdout.splitlines()
                if line.startswith("GPU-") and "," in line}
    return {gpu for gpu, uuid in gpu_uuids.items() if uuid in occupied}


def prepare(start_at, suite, python):
    launch_time = datetime.fromisoformat(start_at)
    if launch_time.tzinfo is None:
        launch_time = launch_time.replace(tzinfo=TZ)
    if launch_time <= datetime.now(TZ):
        raise ValueError("The scheduled time must be in the future")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", suite):
        raise ValueError("Invalid experiment name")
    python = str(Path(python).absolute())
    if not os.access(python, os.X_OK):
        raise FileNotFoundError(python)
    inventory = query_gpus()
    if not all(gpu in inventory for gpu in range(4)):
        raise RuntimeError("This experiment requires GPU indices 0, 1, 2, 3")
    configs = {}
    hashes = {}
    for dataset in DATASETS:
        config = json.loads((ROOT / "configs/datasets" / f"{dataset}.json").read_text())
        for filename in ("dataset_manifest.json", "train.jsonl", "val.jsonl", "test.jsonl"):
            source = ROOT / "datasets" / dataset / "ready" / filename
            if not source.is_file():
                raise FileNotFoundError(source)
            hashes[f"{dataset}/{filename}"] = hashlib.sha256(source.read_bytes()).hexdigest()
        for key in ("text_backbone", "vision_backbone"):
            path = Path(config["model"][key])
            path = path if path.is_absolute() else ROOT / path
            if not (path / "config.json").is_file():
                raise FileNotFoundError(path)
            index_path = path / "model.safetensors.index.json"
            shards = (set(json.loads(index_path.read_text())["weight_map"].values())
                      if index_path.is_file() else {"model.safetensors"})
            if any(not (path / shard).is_file() for shard in shards):
                raise FileNotFoundError(f"Incomplete backbone: {path}")
        config["train"].update(epochs=30, early_stop_patience=8, monitor="macro_f1",
                               per_gpu_batch_size=BATCHES[dataset], grad_accum_steps=1)
        configs[dataset] = config
    directory = ROOT / "workspaces/scheduled" / suite
    directory.mkdir(parents=True, exist_ok=False)
    (directory / "configs").mkdir()
    seeds = secrets.SystemRandom().sample(range(1, 2**31), 4)
    plan = {
        "suite": suite, "created_at": now(), "start_at": launch_time.isoformat(),
        "timezone": "Asia/Shanghai", "project": str(ROOT), "python": python,
        "epochs": 30, "patience": 8, "datasets": list(DATASETS), "seeds": seeds,
        "seed_source": "OS CSPRNG via secrets.SystemRandom", "matched_across_datasets": True,
        "gpu_uuids": {str(g): inventory[g] for g in range(4)},
        "batch_sizes": BATCHES, "gradient_accumulation": 1,
        "dispatch": "FIFO, refill any free GPU immediately; dataset boundaries may overlap",
        "aggregation": "four completed seeds per dataset; mean and sample std (n-1)",
        "input_manifest_sha256": hashes, "jobs": [],
    }
    for dataset in DATASETS:
        config_path = directory / "configs" / f"{dataset}.json"
        write_json(config_path, configs[dataset])
        group = f"{suite}_{dataset}"
        group_dir = ROOT / "workspaces" / dataset / "runs/multiseed" / group
        group_dir.mkdir(parents=True, exist_ok=False)
        write_json(group_dir / "seeds.json", {
            "dataset": dataset, "seeds": seeds, "seed_source": plan["seed_source"],
            "epochs": 30, "early_stop_patience": 8, "suite": suite,
        })
        for seed in seeds:
            name = f"{group}_seed{seed}"
            output_root = Path(configs[dataset]["train"]["output_dir"])
            if not output_root.is_absolute():
                output_root = ROOT / output_root
            run_dir = output_root / name
            if run_dir.exists():
                raise FileExistsError(run_dir)
            plan["jobs"].append({
                "dataset": dataset, "seed": seed, "run_name": name,
                "run_dir": str(run_dir), "group_dir": str(group_dir),
                "log_path": str(group_dir / f"seed_{seed}.log"),
                "command": [python, "-u", str(ROOT / "train.py"), "--dataset", dataset,
                            "--config", str(config_path), "--manifest-dir",
                            str(ROOT / "datasets" / dataset / "ready"), "--seed", str(seed),
                            "--run-name", name, "--epochs", "30", "--early-stop-patience", "8",
                            "--per-gpu-batch-size", str(BATCHES[dataset]), "--grad-accum-steps", "1"],
            })
    write_json(directory / "plan.json", plan)
    write_json(directory / "status.json", {
        "status": "scheduled", "start_at": plan["start_at"], "updated_at": now(),
        "jobs": [dict(job, status="pending") for job in plan["jobs"]], "aggregates": {},
    })
    print(directory / "plan.json")


def run_queue(plan, state, save, probe=busy_gpus, popen=subprocess.Popen, sleep=time.sleep):
    """Own all children; launch FIFO work as soon as a GPU becomes available."""
    gpu_uuids = {int(gpu): uuid for gpu, uuid in plan["gpu_uuids"].items()}
    pending = deque(job for job in state["jobs"] if job["status"] == "pending")
    active, aggregators = {}, {}
    stop = False
    last_probe_error = None
    directory = Path(plan["directory"])

    def request_stop(signum, frame):
        nonlocal stop
        stop = True
        log(f"Stop requested (signal {signum}); cancelling this queue")

    previous_handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    try:
        while pending or active or aggregators:
            if stop:
                raise InterruptedError("Experiment queue cancelled")
            for gpu, (process, job, handle) in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                complete = code == 0 and all((Path(job["run_dir"]) / item).is_file() for item in (
                    "final_summary.json", "evaluation/final/test/metrics.json"))
                job.update(status="completed" if complete else "failed", exit_code=code, ended_at=now())
                del active[gpu]
                log(f"FINISH GPU={gpu} dataset={job['dataset']} seed={job['seed']} status={job['status']} exit={code}")
            # Refill first; CPU aggregation must not hold a GPU idle.
            if pending:
                try:
                    occupied = probe(gpu_uuids)
                    last_probe_error = None
                except (subprocess.SubprocessError, OSError) as error:
                    occupied = set(gpu_uuids)
                    if str(error) != last_probe_error:
                        log(f"GPU availability check failed; waiting: {error}")
                        last_probe_error = str(error)
                state["externally_busy_gpus"] = sorted(occupied - set(active))
                for gpu in gpu_uuids:
                    if not pending:
                        break
                    if gpu in active or gpu in occupied:
                        continue
                    job = pending.popleft()
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu_uuids[gpu], CUDA_DEVICE_ORDER="PCI_BUS_ID",
                               PYTHONHASHSEED=str(job["seed"]), PYTHONUNBUFFERED="1", OMP_NUM_THREADS="8",
                               MKL_NUM_THREADS="8", TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                               TRANSFORMERS_OFFLINE="1", WORLD_SIZE="1", RANK="0", LOCAL_RANK="0")
                    handle = open(job["log_path"], "x")
                    try:
                        process = popen(job["command"], cwd=plan["project"], env=env,
                                        stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                    except Exception as error:
                        handle.close()
                        job.update(status="failed", error=str(error), ended_at=now())
                        raise
                    job.update(status="running", gpu=gpu, pid=process.pid, started_at=now())
                    active[gpu] = (process, job, handle)
                    log(f"START GPU={gpu} dataset={job['dataset']} seed={job['seed']} pid={process.pid} log={job['log_path']}")
            for dataset in plan["datasets"]:
                jobs = [job for job in state["jobs"] if job["dataset"] == dataset]
                if dataset in state["aggregates"] or not all(job["status"] in {"completed", "failed"} for job in jobs):
                    continue
                if any(job["status"] != "completed" for job in jobs):
                    state["aggregates"][dataset] = {"status": "incomplete", "reason": "At least one seed failed"}
                else:
                    output = str(Path(jobs[0]["group_dir"]) / "aggregate.json")
                    command = [plan["python"], str(Path(plan["project"]) / "aggregate_seeds.py"),
                               "--output", output, *[job["run_dir"] for job in jobs]]
                    handle = open(directory / f"aggregate_{dataset}.log", "x")
                    try:
                        process = popen(command, cwd=plan["project"], stdin=subprocess.DEVNULL,
                                        stdout=handle, stderr=subprocess.STDOUT, start_new_session=True)
                    except Exception:
                        handle.close()
                        raise
                    aggregators[dataset] = (process, handle)
                    state["aggregates"][dataset] = {"status": "running", "output": output}
                    log(f"AGGREGATE dataset={dataset} output={output}")
            for dataset, (process, handle) in list(aggregators.items()):
                code = process.poll()
                if code is None:
                    continue
                handle.close()
                complete = code == 0 and Path(state["aggregates"][dataset]["output"]).is_file()
                state["aggregates"][dataset].update(status="completed" if complete else "failed", exit_code=code)
                del aggregators[dataset]
                log(f"AGGREGATE_FINISHED dataset={dataset} exit={code}")
            state.update(updated_at=now(), pending_count=len(pending), active_count=len(active))
            save(state)
            if pending or active or aggregators:
                sleep(2)
        state.update(status="completed" if all(item["status"] == "completed" for item in state["aggregates"].values())
                     else "completed_with_errors", ended_at=now())
        save(state)
        log(f"QUEUE_FINISHED status={state['status']}")
    except BaseException as error:
        processes = [item[0] for item in active.values()] + [item[0] for item in aggregators.values()]
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        for _, job, handle in active.values():
            job.update(status="cancelled", ended_at=now())
            handle.close()
        for _, handle in aggregators.values():
            handle.close()
        for item in state["aggregates"].values():
            if item["status"] == "running":
                item["status"] = "cancelled"
        for job in pending:
            job["status"] = "cancelled"
        state.update(status="cancelled" if isinstance(error, InterruptedError) else "failed",
                     error=str(error), ended_at=now(), active_count=0, pending_count=0)
        save(state)
        raise
    finally:
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)


def run_plan(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text())
    plan["directory"] = str(path.parent)
    lock_path = path.parent / "scheduler.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = path.parent / "status.json"
        state = json.loads(state_path.read_text())
        if state["status"] != "scheduled":
            raise RuntimeError(f"This one-shot plan already ran: {state['status']}")
        state.update(scheduler_pid=os.getpid(), scheduler_started_at=now())
        save = lambda value: write_json(state_path, value)
        save(state)
        def cancel_wait(signum, frame):
            raise InterruptedError(f"Timer cancelled (signal {signum})")

        previous_handlers = {sig: signal.signal(sig, cancel_wait)
                             for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            launch_time = datetime.fromisoformat(plan["start_at"])
            log(f"SCHEDULED start_at={plan['start_at']} jobs={len(plan['jobs'])} GPUs=0,1,2,3")
            while datetime.now(TZ) < launch_time:
                state.update(updated_at=now(), active_count=0, pending_count=len(plan["jobs"]),
                             seconds_to_start=max(0, int((launch_time - datetime.now(TZ)).total_seconds())))
                save(state)
                time.sleep(min(30, max(0.1, (launch_time - datetime.now(TZ)).total_seconds())))
            inventory = query_gpus()
            if any(inventory.get(int(gpu)) != uuid for gpu, uuid in plan["gpu_uuids"].items()):
                raise RuntimeError("GPU index/UUID mapping changed since this task was scheduled")
            for source, expected in plan["input_manifest_sha256"].items():
                dataset, filename = source.split("/", 1)
                actual = hashlib.sha256((ROOT / "datasets" / dataset / "ready" / filename).read_bytes()).hexdigest()
                if actual != expected:
                    raise RuntimeError(f"Dataset manifest changed since scheduling: {source}")
        except BaseException as error:
            state.update(status="cancelled" if isinstance(error, InterruptedError) else "failed",
                         error=str(error), ended_at=now(), active_count=0, pending_count=0)
            for job in state["jobs"]:
                job["status"] = "cancelled"
            save(state)
            log(f"TIMER_STOPPED status={state['status']} error={error}")
            raise
        finally:
            for sig, handler in previous_handlers.items():
                signal.signal(sig, handler)
        state.update(status="running", started_at=now(), seconds_to_start=0)
        save(state)
        run_queue(plan, state, save)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run-plan", type=Path)
    parser.add_argument("--start-at")
    parser.add_argument("--suite")
    parser.add_argument("--python", default=PYTHON)
    args = parser.parse_args()
    if args.prepare:
        if not args.start_at or not args.suite:
            parser.error("--prepare requires --start-at and --suite")
        prepare(args.start_at, args.suite, args.python)
    else:
        run_plan(args.run_plan)


if __name__ == "__main__":
    main()
