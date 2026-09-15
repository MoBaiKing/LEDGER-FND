#!/usr/bin/env python3
"""Paper curves from per-seed CSVs, including future baseline methods."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from statistics import mean, stdev


def plot_results(csv_paths, output_dir, *, dataset=None, uncertainty="shade", allow_subset=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for path in csv_paths:
        with Path(path).open(encoding="utf-8-sig", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    if dataset:
        rows = [r for r in rows if r["dataset"] == dataset]
    if not rows:
        raise ValueError("No rows match the requested dataset")
    datasets = sorted({r["dataset"] for r in rows})
    if len(datasets) > 1:
        for name in datasets:
            plot_results(csv_paths, Path(output_dir) / name, dataset=name, uncertainty=uncertainty, allow_subset=allow_subset)
        return
    scopes = {r.get("evaluation_scope", "full_test") for r in rows}
    if len(scopes) != 1 or (scopes != {"full_test"} and not allow_subset):
        raise ValueError("Paper plots require full_test rows; use --allow-subset for diagnostic plots only")
    if len({r.get("corruption_seed", "2027") for r in rows}) != 1:
        raise ValueError("Do not pool different corruption seeds into checkpoint standard deviation")
    for row in rows:
        row["severity"] = float(row["severity"])
        row["macro_f1"] = float(row["macro_f1"])
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42, "ps.fonttype": 42})
    for corruption, label in (("gaussian", r"Gaussian noise $\sigma$"), ("typo", "Typo injection rate")):
        fig, ax = plt.subplots(figsize=(4.7, 3.4), constrained_layout=True)
        plotted = False
        for method in sorted({r["method"] for r in rows}):
            member = [r for r in rows if r["method"] == method and r["corruption"] in {"clean", corruption}]
            if not any(r["corruption"] == corruption and r["severity"] > 0 for r in member):
                continue
            groups = {}
            for row in member:
                groups.setdefault(row["severity"], []).append(row)
            if 0.0 not in groups:
                raise ValueError(f"{method}: missing clean reference")
            seed_set = None
            for severity, group in groups.items():
                seeds = [r["checkpoint_seed"] for r in group]
                if len(set(seeds)) != len(seeds):
                    raise ValueError(f"Duplicate seed/clean rows for {method} severity={severity}")
                if seed_set is not None and set(seeds) != seed_set:
                    raise ValueError(f"Incomplete seed cohort for {method}")
                seed_set = set(seeds)
            xs = sorted(groups)
            ys = [mean(r["macro_f1"] for r in groups[x]) for x in xs]
            stds = [stdev(r["macro_f1"] for r in groups[x]) if len(groups[x]) > 1 else 0 for x in xs]
            if uncertainty == "errorbar" and len(seed_set) > 1:
                ax.errorbar(xs, ys, yerr=stds, marker="o", markersize=4, capsize=3, label=method, linewidth=1.5)
            else:
                line, = ax.plot(xs, ys, marker="o", markersize=4, label=method, linewidth=1.5)
                if uncertainty == "shade" and len(seed_set) > 1:
                    ax.fill_between(xs, [y-s for y, s in zip(ys, stds)], [y+s for y, s in zip(ys, stds)],
                                    color=line.get_color(), alpha=.15, linewidth=0)
            plotted = True
        if plotted:
            ax.set(xlabel=label, ylabel="Macro-F1", title=datasets[0] + (" (diagnostic subset)" if scopes != {"full_test"} else ""))
            ax.set_xticks(sorted({r["severity"] for r in rows if r["corruption"] in {"clean", corruption}}))
            ax.grid(axis="y", alpha=.22)
            ax.legend(frameon=False)
            for extension in ("pdf", "png"):
                fig.savefig(output_dir / f"{corruption}_robustness.{extension}", dpi=300)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", nargs="+", required=True, type=Path)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--dataset")
    p.add_argument("--uncertainty", choices=("shade", "errorbar", "none"), default="shade")
    p.add_argument("--allow-subset", action="store_true")
    args = p.parse_args()
    plot_results(args.csv, args.output_dir, dataset=args.dataset, uncertainty=args.uncertainty, allow_subset=args.allow_subset)


if __name__ == "__main__":
    main()
