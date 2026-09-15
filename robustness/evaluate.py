#!/usr/bin/env python3
"""Frozen-checkpoint orchestration around mmfnd.engine.evaluate."""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import evaluate, load_checkpoint, write_jsonl
from mmfnd.evaluation import checkpoint_threshold_selection, log_evaluation
from mmfnd.factory import build_processor
from mmfnd.model import ExplainableMMFND
from mmfnd.utils import load_config, resolve_path, seed_everything
from robustness.corruptions import CorruptionSpec
from robustness.pipeline import build_robustness_loader
from robustness.results import aggregate_rows, result_payload, severity_path, summary_row, write_csv, write_json

DEFAULT_SEEDS = "42,3407,2024,2025,200408"


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default=os.environ.get("DATASET"))
    p.add_argument("--config", help="Override config; default uses each checkpoint's saved config")
    p.add_argument("--manifest-dir")
    source = p.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", action="append", type=Path, help="Repeat for multiple checkpoints")
    source.add_argument("--checkpoint-template", help="Path with {seed} and optionally {dataset}")
    source.add_argument("--group", help="Existing run_5seeds group; loads final_averaged.pth per seed")
    p.add_argument("--seeds", default=os.environ.get("SEEDS", DEFAULT_SEEDS))
    p.add_argument("--robustness", choices=("none", "gaussian", "typo"), default="none")
    p.add_argument("--gaussian-sigma", type=float, default=0.0)
    p.add_argument("--typo-rate", type=float, default=0.0)
    p.add_argument("--corruption-seed", type=int, default=int(os.environ.get("CORRUPTION_SEED", "2027")))
    p.add_argument("--suite", action="store_true", help="Clean once + Gaussian/typo at .05, .10, .20")
    p.add_argument("--output-dir", type=Path, help="New result directory; refuses existing run files")
    p.add_argument("--method", default="LEDGER", help="CSV method label, supports future baselines")
    p.add_argument("--per-gpu-batch-size", "--batch-size", dest="batch_size", type=int)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--device", default="auto", help="auto, cpu, cuda:0, ...; CPU evaluates in FP32")
    p.add_argument("--limit-samples", type=int, help="Diagnostic first-N subset, labeled separately from full test")
    p.add_argument("--check-only", action="store_true", help="Validate all checkpoint/manifest contracts without model inference")
    p.add_argument("--verify-clean-equivalence", action="store_true", help="Also compare zero-severity logits/probabilities/metrics")
    p.add_argument("--save-predictions", action="store_true")
    p.add_argument("--plot", action="store_true")
    return p


def checkpoint_paths(args):
    if args.checkpoint:
        paths = args.checkpoint
    elif args.checkpoint_template or args.group:
        if not args.dataset:
            raise ValueError("--dataset is required for --group/--checkpoint-template")
        seeds = [int(s) for s in args.seeds.split(",")]
        if len(set(seeds)) != len(seeds):
            raise ValueError("Duplicate --seeds")
        if args.group:
            config = load_config(resolve_path(ROOT, args.config or f"configs/datasets/{args.dataset}.json"))
            base = resolve_path(ROOT, config["train"]["output_dir"])
            paths = [base / f"{args.group}_seed{s}" / "checkpoints/final_averaged.pth" for s in seeds]
        else:
            if "{seed}" not in args.checkpoint_template:
                raise ValueError("--checkpoint-template must contain {seed}")
            paths = [Path(args.checkpoint_template.format(seed=s, dataset=args.dataset)) for s in seeds]
    elif os.environ.get("CHECKPOINT"):
        paths = [Path(os.environ["CHECKPOINT"])]
    else:
        raise ValueError("Provide --checkpoint, --checkpoint-template, --group, or CHECKPOINT")
    paths = [resolve_path(ROOT, str(p)).resolve() for p in paths]
    if len(set(paths)) != len(paths):
        raise ValueError("Duplicate checkpoint paths")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def prepare_run(path, args):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    selection = checkpoint_threshold_selection(checkpoint)  # fail before loading 7B
    config = deepcopy(load_config(resolve_path(ROOT, args.config)) if args.config else checkpoint["config"])
    config["seed"] = int(checkpoint["config"]["seed"])
    dataset = args.dataset or str(config["dataset"]["name"])
    if checkpoint["config"]["dataset"]["name"] != dataset:
        raise ValueError("Checkpoint dataset mismatch")
    manifest = args.manifest_dir or config["dataset"].get("manifest_dir") or config["data"].get("processed_dir")
    manifest_dir = bind_dataset_workspace(ROOT, config, dataset, manifest or f"datasets/{dataset}/ready")
    if args.batch_size is not None:
        if args.batch_size <= 0:
            raise ValueError("batch size must be positive")
        config["train"]["per_gpu_batch_size"] = args.batch_size
    if args.num_workers is not None:
        if args.num_workers < 0:
            raise ValueError("num_workers must be nonnegative")
        config["data"]["num_workers"] = args.num_workers
    metadata = {
        "dataset": dataset, "method": args.method, "checkpoint": str(path),
        "checkpoint_seed": config["seed"], "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_sha256": file_sha256(path), "manifest_dir": str(manifest_dir),
        "test_manifest_sha256": file_sha256(manifest_dir / "test.jsonl"),
        "evaluation_protocol": selection["evaluation_protocol"], "threshold_selection": selection,
        "evaluation_scope": f"first_{args.limit_samples}_test_samples" if args.limit_samples else "full_test",
        "config": config,
    }
    del checkpoint
    return config, metadata


def run_evaluation(model, config, processor, device, selection, spec, args, *, capture_logits=False):
    loader = build_robustness_loader(ROOT, config, "test", processor, spec, limit_samples=args.limit_samples)
    logits = []
    hook = model.register_forward_hook(lambda module, inputs, outputs: logits.append(outputs["logits"].detach().float().cpu())) if capture_logits else None
    seed_everything(int(config["seed"]))
    positive, names = validate_dataset_semantics(config)
    try:
        metrics, predictions, _ = evaluate(model, loader, device, positive, names,
                                          precision=config["train"].get("precision", "fp32"),
                                          split="test", threshold_selection=selection)
    finally:
        if hook is not None:
            hook.remove()
    return metrics, predictions, torch.cat(logits) if logits else None


def main(argv=None):
    args = parser().parse_args(argv)
    selected = CorruptionSpec(args.robustness, args.gaussian_sigma, args.typo_rate, args.corruption_seed)
    if args.suite and (args.robustness != "none" or args.gaussian_sigma or args.typo_rate):
        raise ValueError("--suite cannot be combined with individual corruption parameters")
    if args.limit_samples is not None and args.limit_samples <= 0:
        raise ValueError("--limit-samples must be positive")
    paths = checkpoint_paths(args)
    prepared = [prepare_run(path, args) for path in paths]
    seeds = [metadata["checkpoint_seed"] for _, metadata in prepared]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Multiple checkpoints must be independently trained seeds")
    # Same test cohort and preprocessing/architecture across paired seed runs.
    reference = prepared[0][0]
    for config, metadata in prepared:
        if any(config[k] != reference[k] for k in ("dataset", "data", "model")):
            raise ValueError("Seed checkpoints have different model/data configurations")
        if any(config["train"].get(k) != reference["train"].get(k) for k in ("precision", "per_gpu_batch_size")):
            raise ValueError("Seed checkpoints require identical evaluation precision and batch size")
        if metadata["test_manifest_sha256"] != prepared[0][1]["test_manifest_sha256"]:
            raise ValueError("Seed checkpoints use different test manifests")
    output = (args.output_dir or ROOT / "robustness/robustness_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")).resolve()
    # A result directory is one cohort/invocation; never mix stale partial runs.
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"Output directory must be empty: {output}")
    write_json(output / "run_manifest.json", {"arguments": vars(args) | {"checkpoint": [str(p) for p in paths],
                                                                      "output_dir": str(output)},
                                               "runs": [m for _, m in prepared], "status": "preflight_passed"})
    if args.check_only:
        print(f"Validated {len(paths)} checkpoints; manifest={output / 'run_manifest.json'}")
        return
    specs = ([CorruptionSpec("gaussian", gaussian_sigma=s, corruption_seed=args.corruption_seed) for s in (.05, .10, .20)] +
             [CorruptionSpec("typo", typo_rate=s, corruption_seed=args.corruption_seed) for s in (.05, .10, .20)]
             if args.suite else ([selected] if selected.active else []))
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    rows = []
    for path, (config, metadata) in zip(paths, prepared):
        print(f"checkpoint={path}; device={device}; seed={config['seed']}", flush=True)
        seed_everything(config["seed"])
        processor = build_processor(ROOT, config)
        model = ExplainableMMFND(config)
        loaded = load_checkpoint(path, model, torch.device("cpu"))
        selection = checkpoint_threshold_selection(loaded)
        if selection != metadata["threshold_selection"]:
            raise ValueError("Checkpoint changed since preflight")
        del loaded
        model = model.to(device)
        if device.type == "cpu":
            model.float()  # CPU shared evaluator has no autocast; mixed BF16/FP32 linear is invalid
        metadata = {**metadata, "device": str(device),
                    "execution_precision": "fp32" if device.type == "cpu" else config["train"].get("precision", "fp32")}
        clean_spec = CorruptionSpec(corruption_seed=args.corruption_seed)
        clean, predictions, clean_logits = run_evaluation(model, config, processor, device, selection, clean_spec, args,
                                                          capture_logits=args.verify_clean_equivalence)
        metadata["samples"] = clean["samples"]
        base = output / metadata["dataset"]
        if len(paths) > 1:
            base = base / f"seed_{config['seed']}"
        payload = result_payload(metadata, clean_spec, clean, predictions, clean)
        write_json(severity_path(base, clean_spec), payload)
        rows.append(summary_row(payload))
        if args.save_predictions:
            write_jsonl(predictions, base / "clean_predictions.jsonl")
        # Reuse the ONE clean evaluation for both zero-severity file names.
        zero_specs = [CorruptionSpec("gaussian", corruption_seed=args.corruption_seed),
                      CorruptionSpec("typo", corruption_seed=args.corruption_seed)] if args.suite or args.verify_clean_equivalence else [selected]
        for zero in zero_specs:
            if zero.severity == 0 and zero.robustness != "none":
                zero_payload = result_payload(metadata, zero, clean, predictions, clean)
                zero_payload["reused_clean_result"] = str(severity_path(base, clean_spec))
                write_json(severity_path(base, zero), zero_payload)
        if args.verify_clean_equivalence:
            from robustness.verify import assert_clean_equivalence
            comparisons = []
            for zero in zero_specs:
                metrics, zero_predictions, zero_logits = run_evaluation(model, config, processor, device, selection, zero, args,
                                                                        capture_logits=True)
                comparisons.append(assert_clean_equivalence(clean, predictions, clean_logits, metrics, zero_predictions,
                                                            zero_logits, zero.robustness))
            write_json(base / "clean_equivalence.json", {"checkpoint": str(path), "samples": clean["samples"],
                                                        "evaluation_scope": metadata["evaluation_scope"], "comparisons": comparisons})
        log_evaluation(clean)
        for spec in specs:
            metrics, predictions, _ = run_evaluation(model, config, processor, device, selection, spec, args)
            payload = result_payload(metadata, spec, metrics, predictions, clean)
            target = severity_path(base, spec)
            write_json(target, payload)
            if args.save_predictions:
                write_jsonl(predictions, target.with_name(target.stem + "_predictions.jsonl"))
            rows.append(summary_row(payload))
            log_evaluation(metrics)
            print(f"robustness={spec.robustness}; severity={spec.severity}; results={target}", flush=True)
        if file_sha256(path) != metadata["checkpoint_sha256"]:
            raise RuntimeError("Checkpoint file changed during evaluation")
        del model, processor, predictions, clean_logits
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        # Save per-seed progress even if a later checkpoint cannot finish.
        write_csv(output / "robustness_summary.csv", rows)
    aggregate = aggregate_rows(rows)
    write_csv(output / "robustness_aggregate.csv", aggregate)
    write_json(output / "robustness_aggregate.json", aggregate)
    for row in aggregate:
        score, std = row["macro_f1_mean"], row["macro_f1_std"]
        print(f"{row['corruption']} {row['severity']:.2f}: Macro-F1={score:.6f}" +
              (f" ± {std:.6f}" if std is not None else " (single checkpoint)"))
    if args.plot:
        from robustness.plot_robustness import plot_results
        plot_results([output / "robustness_summary.csv"], output / "figures", allow_subset=args.limit_samples is not None)
    write_json(output / "completion.json", {"status": "complete", "checkpoints": len(paths),
                                             "evaluation_scope": prepared[0][1]["evaluation_scope"]})
    print(f"results={output}")


if __name__ == "__main__":
    main()
