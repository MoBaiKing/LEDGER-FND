"""Measured tables, paired deltas, module states and explicit incomplete-run lists."""
import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from ablation.registry import VARIANTS, CONTROLS, states
from ablation.run import write_json
from mmfnd.evaluation import compute_classification_metrics, validate_threshold_selection

METRICS = ("macro_f1", "accuracy", "fake_precision", "fake_recall", "fake_f1",
           "real_precision", "real_recall", "real_f1", "auc", "nll", "brier", "ece")


def read_rows(path):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    result = {str(row["id"]): row for row in rows}
    if not rows or len(result) != len(rows):
        raise ValueError(f"Empty predictions or duplicate sample IDs: {path}")
    return result


def freeze_subsets(val_rows, test_rows, reference_threshold, fraction):
    conflict = lambda row: float(row["lgled"]["sample_disagreement"])
    margin = lambda row: abs(float(row["fake_probability"]) - reference_threshold)
    if not 0 < fraction < 1:
        raise ValueError("Invalid subset fraction")
    conflicts = [conflict(row) for row in val_rows.values()]
    margins = [margin(row) for row in val_rows.values()]
    if not np.isfinite(conflicts + margins).all():
        raise ValueError("Nonfinite validation subset scores")
    c = float(np.quantile(conflicts, 1 - fraction))
    m = float(np.quantile(margins, fraction))
    subsets = {"all": sorted(test_rows),
               "conflict": sorted(key for key, row in test_rows.items() if conflict(row) >= c),
               "hard": sorted(key for key, row in test_rows.items() if margin(row) <= m)}
    return subsets, {"validation_conflict_min": c, "validation_boundary_distance_max": m,
                     "fraction": fraction, "reference_threshold": reference_threshold}


def stats(values):
    if not values:
        return {"n": 0, "mean": None, "std": None, "reason": "no completed observations"}
    if any(value is None or not math.isfinite(value) for value in values):
        return {"n": len(values), "mean": None, "std": None, "reason": "undefined metric in at least one run; not silently dropped"}
    return {"n": len(values), "mean": mean(values), "std": stdev(values) if len(values) > 1 else None,
            "reason": None if len(values) > 1 else "sample std requires at least two seeds"}


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_run(run, expected_variant, seed, protocol):
    config = json.loads((run / "config.json").read_text())
    if config["seed"] != seed or config["ablation"]["name"] != expected_variant:
        raise ValueError("Run identity mismatch")
    if config.get("ablation_provenance") != protocol:
        raise ValueError("Training protocol mismatch")
    summary = json.loads((run / "final_summary.json").read_text())
    for path in (run / "history.json", Path(summary["checkpoint"]), Path(summary["best_checkpoint"])):
        if not path.is_file():
            raise FileNotFoundError(f"Incomplete run: {path}")
    folder = run / "evaluation/final"
    val = json.loads((folder / "val/metrics.json").read_text())
    test = json.loads((folder / "test/metrics.json").read_text())
    selection = test["threshold_selection"]
    threshold = validate_threshold_selection(selection)
    if val["threshold_selection"] != selection or summary["threshold_selection"] != selection:
        raise ValueError("Final validation/test/checkpoint thresholds disagree")
    if val["macro_f1"] != selection["val_best_macro_f1"] or test["split"] != "test" or val["split"] != "val":
        raise ValueError("Invalid final evaluation pairing")
    rows = read_rows(folder / "test/predictions.jsonl")
    probabilities = []
    for row in rows.values():
        if (row["dataset"] != protocol["dataset"] or row["seed"] != seed or row["variant"] != expected_variant):
            raise ValueError("Prediction dataset/seed/variant does not match the run")
        p = float(row["fake_probability"])
        logits = np.array(row["logits_final"], dtype=float)
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        if not np.isclose(probs[0], p, atol=1e-6):
            raise ValueError("Final probability does not match exported logits")
        if row["prediction"] != (0 if p >= threshold else 1) or row["threshold"] != threshold:
            raise ValueError("Wrong label direction/threshold in predictions")
        probabilities.append([p, 1 - p])
    measured = compute_classification_metrics([r["label"] for r in rows.values()], probabilities, threshold)
    for key in METRICS:
        if not np.isclose(measured[key], test[key], equal_nan=True, atol=1e-6):
            raise ValueError(f"Stored {key} does not match final probabilities")
    return {"config": config, "summary": summary, "test": test, "rows": rows, "threshold": threshold,
            "val_rows": read_rows(folder / "val/predictions.jsonl"), "run": run}


def summarize_suite(suite, output=None):
    protocol = json.loads((suite / "protocol.json").read_text())
    planned = json.loads((suite / "runs.json").read_text())
    if protocol["smoke_steps"]:
        raise ValueError("Smoke runs are not research results")
    output = output or suite / "report"
    output.mkdir(parents=True, exist_ok=True)
    loaded, failures = {}, []
    variants = list(dict.fromkeys(record["variant"] for record in planned.values()))
    for key, record in planned.items():
        target = record.get("alias_of", key)
        variant, seed_text = target.split("/")
        seed = int(seed_text.removeprefix("seed"))
        run = suite / variant / seed_text
        if not (run / "final_summary.json").is_file():
            failures.append({"dataset": protocol["dataset"], "variant": record["variant"], "seed": seed,
                             "status": record.get("status", "missing"), "reason": record.get("reason", "Final result not available"), "alias_of": record.get("alias_of")})
            continue
        try:
            loaded[(record["variant"], seed)] = load_run(run, variant, seed, protocol)
        except (ValueError, KeyError, FileNotFoundError) as error:
            failures.append({"dataset": protocol["dataset"], "variant": record["variant"], "seed": seed,
                             "status": "invalid_result", "reason": str(error)})
    reference_seed = protocol["subset_definition"]["reference_seed"]
    reference = loaded.get(("full", reference_seed))
    subsets, cutoffs, subset_status = {}, {}, "pending_definition: reference Full has not completed"
    if reference:
        subsets, cutoffs = freeze_subsets(reference["val_rows"], reference["rows"], reference["threshold"], protocol["subset_definition"]["fraction"])
        subset_status = "frozen Full-reference validation-quantile diagnostic subsets; not annotated ground truth"
    # Compare all completed predictions on the exact same test IDs, even if Full is missing.
    reference_rows = next(iter(loaded.values()))["rows"] if loaded else None
    valid = {}
    for key, data in loaded.items():
        rows = data["rows"]
        if rows.keys() != reference_rows.keys() or any(rows[k]["label"] != reference_rows[k]["label"] for k in rows):
            failures.append({"dataset": protocol["dataset"], "variant": key[0], "seed": key[1],
                             "status": "invalid_result", "reason": "Test IDs/labels do not match other runs"})
        else:
            valid[key] = data
    loaded = valid
    per_seed, lookup = [], {}
    for (variant, seed), data in loaded.items():
        rows = data["rows"]
        groups = {"all": list(rows), **{k: v for k, v in subsets.items() if k != "all"}}
        for subset, ids in groups.items():
            if not ids:
                continue
            measured = compute_classification_metrics([rows[k]["label"] for k in ids],
                [[rows[k]["fake_probability"], 1 - rows[k]["fake_probability"]] for k in ids], data["threshold"])
            metrics = {key: float(measured[key]) if math.isfinite(measured[key]) else None for key in METRICS}
            record = {"dataset": protocol["dataset"], "variant": variant, "seed": seed, "subset": subset,
                      "samples": len(ids), "fake_support": sum(rows[k]["label"] == 0 for k in ids),
                      "real_support": sum(rows[k]["label"] == 1 for k in ids), "threshold": data["threshold"], **metrics,
                      "auc_reason": "single-class subset" if metrics["auc"] is None else None,
                      "fixed_mix_coefficient": data["config"]["ablation"]["fixed_mix_coefficient"],
                      "best_checkpoint": data["summary"]["best_checkpoint"], "selection_metric": "validation macro_f1",
                      "final_checkpoint": data["summary"]["checkpoint"], "actual_epochs": len(json.loads((data["run"] / "history.json").read_text())),
                      "config_hash": data["config"]["ablation_resolved_config_hash"],
                      **data["config"].get("ablation_parameter_counts", {})}
            runtime_files = list(data["run"].glob("runtime_rank*.json"))
            runtime = [json.loads(p.read_text()) for p in runtime_files]
            record["wall_seconds"] = max((r["wall_seconds"] for r in runtime), default=None)
            memory = [r["peak_memory_allocated_bytes"] for r in runtime if r["peak_memory_allocated_bytes"] is not None]
            record["peak_memory_bytes"] = max(memory, default=None)
            record["cost_reason"] = "see runtime records; CPU memory not measured" if runtime else "runtime record missing"
            per_seed.append(record)
            lookup[(variant, seed, subset)] = record
    paired = []
    for row in per_seed:
        baseline = "dual_path_reference" if row["variant"] in CONTROLS else "full"
        ref = lookup.get((baseline, row["seed"], row["subset"]))
        if ref is None:
            continue
        for metric in METRICS:
            delta = ref[metric] - row[metric] if ref[metric] is not None and row[metric] is not None else None
            paired.append({"dataset": protocol["dataset"], "variant": row["variant"], "seed": row["seed"],
                           "subset": row["subset"], "reference": baseline, "metric": metric, "delta_raw": delta,
                           "delta_macro_f1": delta if metric == "macro_f1" else None,
                           "delta_macro_f1_pp": 100 * delta if metric == "macro_f1" and delta is not None else None})
    aggregate = []
    for variant in variants:
        for subset in ("all", "conflict", "hard"):
            for metric in METRICS:
                values = [r[metric] for r in per_seed if r["variant"] == variant and r["subset"] == subset]
                aggregate.append({"dataset": protocol["dataset"], "variant": variant, "subset": subset, "metric": metric,
                                  "expected_seeds": len(protocol["seeds"]), **stats(values)})
    modules = [states(name) for name in variants]
    paired_aggregate = []
    for variant in variants:
        for subset in ("all", "conflict", "hard"):
            for metric in METRICS:
                values = [row["delta_raw"] for row in paired if row["variant"] == variant and row["subset"] == subset and row["metric"] == metric]
                summary = stats(values)
                paired_aggregate.append({"dataset": protocol["dataset"], "variant": variant, "subset": subset,
                    "metric": metric, "reference": "dual_path_reference" if variant in CONTROLS else "full", **summary,
                    "delta_macro_f1_pp_mean": 100 * summary["mean"] if metric == "macro_f1" and summary["mean"] is not None else None})
    write_csv(output / "per_seed_metrics.csv", per_seed, list(per_seed[0]) if per_seed else ["dataset", "variant", "seed", *METRICS])
    write_csv(output / "mean_std.csv", aggregate, list(aggregate[0]))
    write_csv(output / "paired_deltas.csv", paired, ["dataset", "variant", "seed", "subset", "reference", "metric", "delta_raw", "delta_macro_f1", "delta_macro_f1_pp"])
    write_csv(output / "paired_mean_std.csv", paired_aggregate, list(paired_aggregate[0]))
    write_csv(output / "module_states.csv", modules, list(modules[0]))
    write_csv(output / "missing_failed.csv", failures, ["dataset", "variant", "seed", "status", "reason", "alias_of"])
    write_json(output / "results.json", {"protocol": protocol, "standard_deviation": "sample (n-1)",
        "subset_status": subset_status, "subset_cutoffs": cutoffs, "subset_ids": subsets,
        "per_seed": per_seed, "aggregate": aggregate, "paired": paired, "paired_aggregate": paired_aggregate, "missing_failed": failures})
    lines = [f"# {protocol['dataset']} 消融结果", "", f"计划 seed 数：{len(protocol['seeds'])}；下表仅列实测值，未完成用 —。", "",
             "| Variant | 完成 seed | Macro-F1 (%) | Accuracy (%) | AUC | NLL | Brier | ECE-15 | 配对ΔF1(pp) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for variant in variants:
        cells = []
        for metric in ("macro_f1", "accuracy", "auc", "nll", "brier", "ece"):
            item = next(r for r in aggregate if r["variant"] == variant and r["subset"] == "all" and r["metric"] == metric)
            scale = 100 if metric in {"macro_f1", "accuracy"} else 1
            cells.append("—" if item["mean"] is None else f"{scale * item['mean']:.4f} ± " + (f"{scale * item['std']:.4f}" if item["std"] is not None else "NA"))
        n = sum(r["variant"] == variant and r["subset"] == "all" for r in per_seed)
        delta = next(row for row in paired_aggregate if row["variant"] == variant and row["subset"] == "all" and row["metric"] == "macro_f1")
        cells.append("—" if delta["mean"] is None else f"{100 * delta['mean']:.4f} (n={delta['n']})")
        lines.append(f"| {variant} | {n}/{len(protocol['seeds'])} | " + " | ".join(cells) + " |")
    lines += ["", f"子集状态：{subset_status}", "", "完整类别 P/R/F1、配对原始差值/百分点差值、模块状态和失败原因见同目录 CSV。"]
    (output / "results.md").write_text("\n".join(lines) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dirs", "--suite-dir", nargs="+", type=Path, required=True)
    args = parser.parse_args()
    for suite in args.suite_dirs:
        print(summarize_suite(suite.resolve()))


if __name__ == "__main__":
    main()
