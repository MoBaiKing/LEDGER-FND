#!/usr/bin/env python3
"""Stop one scheduled suite at a deadline and build a resumable plan."""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import time
from zoneinfo import ZoneInfo


TZ = ZoneInfo("Asia/Shanghai")


def now() -> str:
    return datetime.now(TZ).isoformat(timespec="seconds")


def read_json(path: Path) -> dict:
    for attempt in range(5):
        try:
            return json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            if attempt == 4:
                raise
            time.sleep(0.1)
    raise AssertionError("unreachable")


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def process_command(pid: int) -> list[str]:
    raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    return [item.decode(errors="replace") for item in raw.split(b"\0") if item]


def scheduler_matches(pid: int, plan_path: Path) -> bool:
    try:
        command = process_command(pid)
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return False
    return (
        any(Path(item).name == "scheduled_random4.py" for item in command)
        and str(plan_path.resolve()) in command
    )


def checkpoint_metadata(path: Path) -> dict:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("optimizer_state_dict") is None:
        raise ValueError("checkpoint has no optimizer state")
    if checkpoint.get("scheduler_state_dict") is None:
        raise ValueError("checkpoint has no scheduler state")
    epoch = int(checkpoint["epoch"])
    state = checkpoint.get("training_state", {})
    result = {
        "path": str(path.resolve()),
        "epoch": epoch,
        "global_step": int(state.get("global_step", 0)),
        "bad_epochs": int(state.get("bad_epochs", 0)),
        "best_score": float(state.get("best_score", float("nan"))),
        "size_bytes": path.stat().st_size,
        "mtime_ns": path.stat().st_mtime_ns,
        "validated_at": now(),
        "has_optimizer_state": True,
        "has_scheduler_state": True,
    }
    del checkpoint
    return result


def snapshot_checkpoint(job: dict, stamp: str, previous: dict | None) -> dict | None:
    source = Path(job["run_dir"]) / "checkpoints" / "last.pth"
    if not source.is_file():
        return previous
    source_stat = source.stat()
    signature = (source_stat.st_mtime_ns, source_stat.st_size)
    if previous and tuple(previous.get("source_signature", ())) == signature:
        return previous

    target = source.parent / f"cutoff_{stamp}.pth"
    temporary = target.with_suffix(f".{os.getpid()}.tmp")
    try:
        shutil.copy2(source, temporary)
        after = source.stat()
        if (after.st_mtime_ns, after.st_size) != signature:
            raise RuntimeError("source changed while it was copied")
        metadata = checkpoint_metadata(temporary)
        temporary.replace(target)
        metadata.update(
            path=str(target.resolve()),
            source=str(source.resolve()),
            source_signature=list(signature),
            dataset=job["dataset"],
            seed=int(job["seed"]),
        )
        print(
            f"[{now()}] SNAPSHOT dataset={job['dataset']} seed={job['seed']} "
            f"epoch={metadata['epoch']} path={target}",
            flush=True,
        )
        return metadata
    except Exception as error:
        temporary.unlink(missing_ok=True)
        print(
            f"[{now()}] SNAPSHOT_RETRY dataset={job['dataset']} "
            f"seed={job['seed']} error={error}",
            flush=True,
        )
        return previous


def is_completed(job: dict) -> bool:
    run_dir = Path(job["run_dir"])
    return (
        (run_dir / "final_summary.json").is_file()
        and (run_dir / "evaluation/final/test/metrics.json").is_file()
    )


def replace_option(command: list[str], option: str, value: str) -> list[str]:
    result = list(command)
    if option in result:
        index = result.index(option)
        result[index + 1] = value
    else:
        result.extend([option, value])
    return result


def remove_option(command: list[str], option: str) -> list[str]:
    result = list(command)
    while option in result:
        index = result.index(option)
        del result[index:index + 2]
    return result


def build_continuation(plan: dict, state: dict, output_dir: Path,
                       snapshots: dict[str, dict], cutoff_at: str) -> dict:
    continuation = output_dir / "continuation"
    continuation.mkdir(parents=True, exist_ok=False)
    resume_logs = continuation / "logs"
    resume_logs.mkdir()

    state_jobs = {
        (job["dataset"], int(job["seed"])): job for job in state["jobs"]
    }
    resumed_jobs, continuation_jobs = [], []
    completed_jobs = []
    for original in plan["jobs"]:
        key = (original["dataset"], int(original["seed"]))
        old_state = state_jobs[key]
        job = dict(original)
        if is_completed(job):
            completed_jobs.append({"dataset": key[0], "seed": key[1],
                                   "run_dir": job["run_dir"]})
            continuation_jobs.append(dict(old_state, status="completed"))
            continue

        snapshot = snapshots.get(f"{key[0]}:{key[1]}")
        command = remove_option(job["command"], "--resume")
        resume_from = None
        checkpoint_epoch = None
        if snapshot and Path(snapshot["path"]).is_file():
            resume_from = snapshot["path"]
            checkpoint_epoch = int(snapshot["epoch"])
            command.extend(["--resume", resume_from])
        elif Path(job["run_dir"]).exists():
            # Preserve an initialized but checkpoint-free run and restart this
            # seed under a new name rather than deleting partial artifacts.
            old_name = job["run_name"]
            new_name = f"{old_name}_fresh_after_{cutoff_at[11:16].replace(':', '')}"
            command = replace_option(command, "--run-name", new_name)
            run_dir = Path(job["run_dir"])
            job["run_name"] = new_name
            job["run_dir"] = str(run_dir.parent / new_name)

        job["command"] = command
        job["log_path"] = str(
            resume_logs / f"{key[0]}_seed{key[1]}.log"
        )
        continuation_jobs.append(dict(
            job, status="pending", resume_from=resume_from,
            checkpoint_epoch=checkpoint_epoch,
        ))
        resumed_jobs.append({
            "dataset": key[0], "seed": key[1], "run_dir": job["run_dir"],
            "resume_from": resume_from, "checkpoint_epoch": checkpoint_epoch,
            "command": command,
        })

    resume_plan = dict(plan)
    resume_plan.update(
        suite=f"{plan['suite']}_continuation_1025",
        created_at=now(), start_at=cutoff_at,
        continuation_of=plan["suite"], jobs=[
            {key: value for key, value in job.items()
             if key not in {"status", "resume_from", "checkpoint_epoch"}}
            for job in continuation_jobs
        ],
    )
    completed_aggregates = {
        dataset: value for dataset, value in state.get("aggregates", {}).items()
        if value.get("status") == "completed"
    }
    resume_status = {
        "status": "scheduled" if resumed_jobs else "completed",
        "start_at": cutoff_at,
        "updated_at": now(),
        "continuation_of": plan["suite"],
        "pending_count": len(resumed_jobs), "active_count": 0,
        "jobs": continuation_jobs, "aggregates": completed_aggregates,
    }
    plan_path = continuation / "plan.json"
    status_path = continuation / "status.json"
    write_json(plan_path, resume_plan)
    write_json(status_path, resume_status)
    resume_command = (
        f"cd {shlex.quote(plan['project'])} && "
        f"{shlex.quote(plan['python'])} -u scripts/scheduled_random4.py "
        f"--run-plan {shlex.quote(str(plan_path.resolve()))}"
    )
    (continuation / "resume_command.txt").write_text(
        (resume_command + "\n") if resumed_jobs else "No resume is required.\n"
    )
    manifest = {
        "cutoff_at": cutoff_at, "created_at": now(),
        "original_suite": plan["suite"],
        "original_final_status": state.get("status"),
        "resume_required": bool(resumed_jobs),
        "completed_jobs": completed_jobs,
        "unfinished_jobs": resumed_jobs,
        "validated_snapshots": snapshots,
        "resume_plan": str(plan_path.resolve()),
        "resume_status": str(status_path.resolve()),
        "resume_command_file": str((continuation / "resume_command.txt").resolve()),
    }
    write_json(output_dir / "cutoff_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--cutoff-at", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--snapshot-interval", type=float, default=30.0)
    args = parser.parse_args()

    plan_path = args.plan.resolve()
    status_path = args.status.resolve()
    plan = read_json(plan_path)
    cutoff = datetime.fromisoformat(args.cutoff_at)
    if cutoff.tzinfo is None:
        cutoff = cutoff.replace(tzinfo=TZ)
    if cutoff <= datetime.now(TZ):
        raise ValueError("cutoff must be in the future")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = cutoff.strftime("%Y%m%d_%H%M%S")
    snapshots: dict[str, dict] = {}
    print(f"[{now()}] CUTOFF_SCHEDULED cutoff_at={cutoff.isoformat()}", flush=True)

    while datetime.now(TZ) < cutoff:
        state = read_json(status_path)
        for job in state["jobs"]:
            if not is_completed(job):
                key = f"{job['dataset']}:{int(job['seed'])}"
                snapshots[key] = snapshot_checkpoint(
                    job, stamp, snapshots.get(key)
                ) or snapshots.get(key)
        write_json(output_dir / "snapshot_index.json", snapshots)
        if state.get("status") in {"completed", "completed_with_errors", "failed"}:
            break
        delay = min(args.snapshot_interval,
                    max(0.1, (cutoff - datetime.now(TZ)).total_seconds()))
        time.sleep(delay)

    state = read_json(status_path)
    pid = int(state.get("scheduler_pid", 0) or 0)
    if state.get("status") in {"scheduled", "running"}:
        if not scheduler_matches(pid, plan_path):
            raise RuntimeError(f"refusing to signal unverified scheduler PID {pid}")
        print(f"[{now()}] CUTOFF_SIGNAL scheduler_pid={pid}", flush=True)
        os.kill(pid, signal.SIGTERM)
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            time.sleep(1)
            state = read_json(status_path)
            if state.get("status") not in {"scheduled", "running"}:
                break
        else:
            raise TimeoutError("scheduler did not stop within 60 seconds")

    state = read_json(status_path)
    for job in state["jobs"]:
        if not is_completed(job):
            key = f"{job['dataset']}:{int(job['seed'])}"
            snapshots[key] = snapshot_checkpoint(
                job, stamp, snapshots.get(key)
            ) or snapshots.get(key)
    write_json(output_dir / "snapshot_index.json", snapshots)
    manifest = build_continuation(
        plan, state, output_dir, snapshots, cutoff.isoformat()
    )
    print(
        f"[{now()}] CUTOFF_COMPLETE resume_required={manifest['resume_required']} "
        f"unfinished={len(manifest['unfinished_jobs'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
