#!/usr/bin/env python3
"""Aggregate frozen final-test metrics with the v2 reporting contract."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev


REPORT_METRICS = (
    "macro_f1", "accuracy",
    "fake_precision", "fake_recall", "fake_f1",
    "real_precision", "real_recall", "real_f1",
    "auc", "brier", "nll", "decision_threshold",
)
CONFUSION_FIELDS = ("tp", "tn", "fp", "fn")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report mean and sample standard deviation over seed runs"
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.run_dirs) < 2:
        raise ValueError("at least two seed runs are required")

    rows = []
    for run_dir in args.run_dirs:
        info_path = run_dir / "run_info.json"
        metrics_path = run_dir / "evaluation/final/test/metrics.json"
        summary_path = run_dir / "final_summary.json"
        for path in (info_path, metrics_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(f"incomplete seed run: missing {path}")
        info = json.loads(info_path.read_text(encoding="utf-8"))
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        missing = [name for name in REPORT_METRICS if metrics.get(name) is None]
        if missing:
            raise ValueError(f"{metrics_path} missing metrics: {missing}")
        rows.append({
            "run": run_dir.name,
            "dataset": str(info["dataset"]),
            "seed": int(info["seed"]),
            "epochs": int(info["epochs"]),
            "early_stop_patience": int(info["early_stop_patience"]),
            **{name: float(metrics[name]) for name in REPORT_METRICS},
            **{name: int(metrics[name]) for name in CONFUSION_FIELDS},
        })

    seeds = [row["seed"] for row in rows]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds are not independent runs: {seeds}")
    for field in ("dataset", "epochs", "early_stop_patience"):
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"seed runs use different {field}: {sorted(values)}")

    aggregate = {
        name: {
            "mean": mean(row[name] for row in rows),
            "sample_std": stdev(row[name] for row in rows),
        }
        for name in REPORT_METRICS
    }
    payload = {
        "runs": len(rows),
        "dataset": rows[0]["dataset"],
        "epochs": rows[0]["epochs"],
        "early_stop_patience": rows[0]["early_stop_patience"],
        "seeds": seeds,
        "standard_deviation": "sample (n-1)",
        "per_seed": rows,
        "aggregate": aggregate,
        "confusion_matrix_sum": {
            name: sum(row[name] for row in rows) for name in CONFUSION_FIELDS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with args.output.with_name("per_seed_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with args.output.with_name("aggregate_metrics.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=("metric", "mean", "sample_std"))
        writer.writeheader()
        for name, values in aggregate.items():
            writer.writerow({"metric": name, **values})

    for name in REPORT_METRICS:
        values = aggregate[name]
        print(f"{name}: {values['mean']:.6f} ± {values['sample_std']:.6f}")
    print(f"summary={args.output}")


if __name__ == "__main__":
    main()
