"""Serialize shared evaluator metrics and paired drops; no metric reimplementation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev

METRICS = ("accuracy", "macro_f1", "auc", "nll", "brier", "ece")
DROP_NAMES = {"accuracy": "acc", "macro_f1": "f1", "auc": "auc"}


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def paired_drops(clean, corrupted):
    result = {}
    for metric, short in DROP_NAMES.items():
        baseline, value = clean.get(metric), corrupted.get(metric)
        valid = baseline is not None and value is not None and math.isfinite(baseline) and math.isfinite(value)
        drop = baseline - value if valid else None
        result[f"{short}_drop"] = drop
        result[f"relative_{short}_drop_pct"] = drop / baseline * 100 if valid and baseline != 0 else None
    return result


def flatten_diagnostics(metrics, predictions):
    result = {}
    def visit(prefix, value):
        if isinstance(value, dict):
            for key, item in value.items():
                visit(f"{prefix}_{key}" if prefix else key, item)
        elif isinstance(value, (int, float)):
            result[prefix] = value
    visit("", metrics.get("lgled", {}))
    if predictions:
        result["mean_uncertainty"] = mean(row["uncertainty"] for row in predictions)
    return result


def result_payload(metadata, spec, metrics, predictions, clean):
    diagnostics = flatten_diagnostics(metrics, predictions)
    return {
        **metadata, "robustness": spec.robustness, "corruption": "clean" if spec.severity == 0 else spec.robustness,
        "severity": spec.severity, "corruption_seed": spec.corruption_seed,
        **{key: metrics[key] for key in METRICS if key in metrics},
        **paired_drops(clean, metrics), "diagnostics": diagnostics,
        "metrics": metrics, "clean_metrics": {key: clean[key] for key in METRICS if key in clean},
        "undefined_metrics": [key for key in METRICS if key in metrics and not math.isfinite(metrics[key])],
    }


def severity_path(base, spec):
    if spec.robustness == "none":
        return Path(base) / "clean.json"
    prefix = "sigma" if spec.robustness == "gaussian" else "rate"
    # Keep the requested paper names without colliding for custom severities.
    value = f"{spec.severity:.2f}" if round(spec.severity, 2) == spec.severity else str(spec.severity)
    return Path(base) / spec.robustness / f"{prefix}_{value}.json"


def summary_row(payload):
    return {
        **{k: payload[k] for k in ("dataset", "method", "checkpoint", "checkpoint_seed", "corruption_seed",
                                  "corruption", "severity", "evaluation_scope", "samples")},
        "acc": payload.get("accuracy"),
        **{k: payload.get(k) for k in METRICS},
        **{k: payload.get(k) for short in DROP_NAMES.values()
           for k in (f"{short}_drop", f"relative_{short}_drop_pct")},
        **payload["diagnostics"],
    }


def write_csv(path, rows):
    if not rows:
        raise ValueError("No result rows to write")
    path = Path(path)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(json_safe(rows))
    temporary.replace(path)


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in ("dataset", "method", "corruption", "severity", "corruption_seed", "evaluation_scope"))
        groups.setdefault(key, []).append(row)
    result = []
    expected_seeds = {}
    for key, members in groups.items():
        seeds = [m["checkpoint_seed"] for m in members]
        if len(seeds) != len(set(seeds)):
            raise ValueError(f"Duplicate checkpoint seeds for {key}")
        cohort = (key[0], key[1], key[4], key[5])
        if cohort in expected_seeds and set(seeds) != expected_seeds[cohort]:
            raise ValueError("Every severity must include the same checkpoint seeds")
        expected_seeds[cohort] = set(seeds)
        if len({m["samples"] for m in members}) != 1:
            raise ValueError("Cannot aggregate different test sample counts")
        row = dict(zip(("dataset", "method", "corruption", "severity", "corruption_seed", "evaluation_scope"), key))
        row.update(n_seeds=len(seeds), checkpoint_seeds=",".join(map(str, seeds)),
                   samples=members[0]["samples"], standard_deviation="sample (n-1)")
        names = list(METRICS) + [f"{s}_drop" for s in DROP_NAMES.values()] + [f"relative_{s}_drop_pct" for s in DROP_NAMES.values()]
        names += [name for name in members[0] if name.startswith("mean_")]
        for name in names:
            values = [m.get(name) for m in members]
            valid = all(v is not None and math.isfinite(v) for v in values)
            row[f"{name}_mean"] = mean(values) if valid else None
            # Sample std is undefined for a single checkpoint, never fake zero.
            row[f"{name}_std"] = stdev(values) if valid and len(values) > 1 else None
        result.append(row)
    return result
