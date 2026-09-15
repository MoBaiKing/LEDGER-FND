#!/usr/bin/env python3
"""Export reproducible experiment results without model checkpoints."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any


RUN_METADATA = ("config.json", "run_info.json", "history.json", "final_summary.json")
EVALUATION_METADATA = ("evaluation_info.json", "metrics.json", "predictions.jsonl")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_artifact(source: Path, destination: Path, repo_root: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_size = source.stat().st_size
    source_digest = sha256(source)
    if source.name == "predictions.jsonl":
        destination = destination.with_suffix(destination.suffix + ".gz")
        with source.open("rb") as src, destination.open("wb") as raw_dst:
            with gzip.GzipFile(filename="", mode="wb", compresslevel=9, mtime=0, fileobj=raw_dst) as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)
        record = {
            "path": destination.relative_to(repo_root).as_posix(),
            "size_bytes": destination.stat().st_size,
            "sha256": sha256(destination),
            "compression": "gzip",
            "original_name": source.name,
            "original_size_bytes": source_size,
            "original_sha256": source_digest,
        }
    else:
        shutil.copy2(source, destination)
        record = {
            "path": destination.relative_to(repo_root).as_posix(),
            "size_bytes": source_size,
            "sha256": source_digest,
        }
    return record


def export_run(
    repo_root: Path,
    output_dir: Path,
    dataset: str,
    experiment: str,
    run_name: str,
    seed: int,
    status: str,
    require_final: bool,
) -> dict[str, Any]:
    source_dir = repo_root / "workspaces" / dataset / "runs" / "training" / run_name
    if not source_dir.is_dir():
        raise FileNotFoundError(source_dir)
    destination_dir = output_dir / dataset / experiment / "runs" / run_name
    files: list[dict[str, Any]] = []
    for filename in RUN_METADATA:
        source = source_dir / filename
        if source.is_file():
            files.append(copy_artifact(source, destination_dir / filename, repo_root))
    for split in ("val", "test"):
        for filename in EVALUATION_METADATA:
            source = source_dir / "evaluation" / "final" / split / filename
            if source.is_file():
                files.append(
                    copy_artifact(
                        source,
                        destination_dir / "evaluation" / split / filename,
                        repo_root,
                    )
                )
            elif require_final:
                raise FileNotFoundError(source)
    return {
        "run_name": run_name,
        "seed": int(seed),
        "status": status,
        "source": source_dir.relative_to(repo_root).as_posix(),
        "files": files,
    }


def metric_mean(aggregate: dict[str, Any], name: str) -> float | None:
    value = aggregate.get("aggregate", {}).get(name)
    if isinstance(value, dict) and "mean" in value:
        return float(value["mean"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--variant", required=True)
    parser.add_argument("--experiment-prefix")
    parser.add_argument("--suite-status", type=Path)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = (repo_root / args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to replace existing export: {output_dir}")
    output_dir.mkdir(parents=True)
    (output_dir / ".gitattributes").write_text("*.jsonl.gz binary\n", encoding="utf-8")

    experiments: dict[tuple[str, str], dict[str, Any]] = {}
    exported_runs: dict[str, dict[str, Any]] = {}

    aggregate_paths = sorted(
        (repo_root / "workspaces").glob("*/runs/multiseed/*/aggregate.json")
    )
    for aggregate_path in aggregate_paths:
        experiment = aggregate_path.parent.name
        if args.experiment_prefix and not experiment.startswith(args.experiment_prefix):
            continue
        aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
        dataset = str(aggregate["dataset"])
        runs = int(aggregate["runs"])
        destination = output_dir / dataset / experiment / "aggregate.json"
        aggregate_file = copy_artifact(aggregate_path, destination, repo_root)
        item = {
            "dataset": dataset,
            "experiment": experiment,
            "kind": f"{runs}-seed",
            "status": "completed",
            "planned_runs": runs,
            "completed_runs": runs,
            "epochs": aggregate.get("epochs"),
            "early_stop_patience": aggregate.get("early_stop_patience"),
            "seeds": [int(seed) for seed in aggregate.get("seeds", [])],
            "metrics": {
                name: metric_mean(aggregate, name)
                for name in ("accuracy", "macro_f1", "auc")
            },
            "aggregate_file": aggregate_file,
            "runs": [],
        }
        for per_seed in aggregate["per_seed"]:
            run_name = str(per_seed["run"])
            run = export_run(
                repo_root,
                output_dir,
                dataset,
                experiment,
                run_name,
                int(per_seed["seed"]),
                "completed",
                require_final=True,
            )
            item["runs"].append(run)
            exported_runs[run_name] = run
        experiments[(dataset, experiment)] = item

    if args.suite_status:
        suite_status_path = args.suite_status.resolve()
        suite = json.loads(suite_status_path.read_text(encoding="utf-8"))
        jobs = [
            job
            for job in suite["jobs"]
            if not args.experiment_prefix
            or str(job["run_name"]).startswith(args.experiment_prefix)
        ]
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for job in jobs:
            dataset = str(job["dataset"])
            experiment = Path(job["group_dir"]).name
            grouped.setdefault((dataset, experiment), []).append(job)

        for key, group_jobs in grouped.items():
            dataset, experiment = key
            destination = output_dir / dataset / experiment
            public_status = {
                "suite_status": suite.get("status"),
                "suite_updated_at": suite.get("updated_at"),
                "dataset": dataset,
                "experiment": experiment,
                "jobs": [
                    {
                        field: job.get(field)
                        for field in (
                            "dataset",
                            "seed",
                            "run_name",
                            "status",
                            "started_at",
                            "ended_at",
                            "exit_code",
                        )
                        if field in job
                    }
                    for job in group_jobs
                ],
            }
            write_json(destination / "status.json", public_status)
            status_file = {
                "path": (destination / "status.json").relative_to(repo_root).as_posix(),
                "size_bytes": (destination / "status.json").stat().st_size,
                "sha256": sha256(destination / "status.json"),
            }
            item = experiments.get(key)
            if item is None:
                completed = sum(job["status"] == "completed" for job in group_jobs)
                item = {
                    "dataset": dataset,
                    "experiment": experiment,
                    "kind": f"{len(group_jobs)}-seed",
                    "status": "completed" if completed == len(group_jobs) else "partial",
                    "planned_runs": len(group_jobs),
                    "completed_runs": completed,
                    "epochs": None,
                    "early_stop_patience": None,
                    "seeds": [int(job["seed"]) for job in group_jobs],
                    "metrics": {"accuracy": None, "macro_f1": None, "auc": None},
                    "aggregate_file": None,
                    "runs": [],
                }
                experiments[key] = item
            item["suite_status_file"] = status_file
            item["planned_runs"] = len(group_jobs)
            item["completed_runs"] = sum(job["status"] == "completed" for job in group_jobs)
            if item["completed_runs"] != item["planned_runs"]:
                item["status"] = "partial"

            known = {run["run_name"] for run in item["runs"]}
            for job in group_jobs:
                run_name = str(job["run_name"])
                if run_name in known:
                    continue
                run = export_run(
                    repo_root,
                    output_dir,
                    dataset,
                    experiment,
                    run_name,
                    int(job["seed"]),
                    str(job["status"]),
                    require_final=job["status"] == "completed",
                )
                item["runs"].append(run)
                exported_runs[run_name] = run

        suite_root = suite_status_path.parent
        suite_output = output_dir / "suites" / suite_root.name
        for source in sorted(suite_root.rglob("*")):
            if not source.is_file() or source.suffix not in {".json", ".txt"}:
                continue
            if source.name == "snapshot_index.json":
                continue
            relative = source.relative_to(suite_root)
            copy_artifact(source, suite_output / relative, repo_root)

    experiment_list = sorted(
        experiments.values(), key=lambda item: (item["dataset"], item["experiment"])
    )
    manifest = {
        "schema_version": "sotamodel_results_export_v1",
        "model_variant": args.variant,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "label_semantics": {"0": "fake", "1": "real"},
        "prediction_storage": {
            "format": "JSON Lines",
            "compression": "gzip",
            "deterministic_gzip_mtime": 0,
            "integrity": "manifest records SHA-256 for compressed and original bytes",
        },
        "experiments": experiment_list,
    }
    write_json(output_dir / "manifest.json", manifest)

    lines = [
        "# Experiment results",
        "",
        f"Model variant: **{args.variant}**.",
        "",
        "This directory contains per-sample validation/test predictions, metrics,",
        "threshold metadata, run configuration, epoch history, final summaries and",
        "multi-seed aggregates. Checkpoints, optimizer state, pretrained weights and",
        "training logs are intentionally excluded.",
        "",
        "Prediction files are the original JSONL bytes stored with deterministic gzip",
        "compression. Decompress with gzip -dc predictions.jsonl.gz; SHA-256 values",
        "for both compressed and original bytes are recorded in manifest.json.",
        "",
        "Labels use 0=fake and 1=real.",
        "",
        "| Dataset | Experiment | Seeds | Status | Accuracy | Macro-F1 | AUC |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for item in experiment_list:
        def pct(name: str) -> str:
            value = item["metrics"].get(name)
            return "—" if value is None else f"{100.0 * value:.4f}%"

        status = (
            "complete"
            if item["status"] == "completed"
            else f"partial ({item['completed_runs']}/{item['planned_runs']})"
        )
        lines.append(
            f"| {item['dataset']} | {item['experiment']} | "
            f"{item['planned_runs']} | {status} | {pct('accuracy')} | "
            f"{pct('macro_f1')} | {pct('auc')} |"
        )
    lines.extend(
        [
            "",
            "A partial experiment has no multi-seed aggregate. Its completed runs retain",
            "their final predictions and metrics; interrupted runs retain configuration",
            "and epoch history only.",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
