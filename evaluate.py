#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

import torch

from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import evaluate, load_checkpoint, save_checkpoint, write_jsonl
from mmfnd.evaluation import checkpoint_threshold_selection, evaluation_info, log_evaluation
from mmfnd.factory import build_loader, build_processor, build_model
from mmfnd.model import ExplainableMMFND
from mmfnd.utils import dump_json, get_device, load_config, resolve_path, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen checkpoint and threshold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--manifest-dir")
    parser.add_argument("--frozen", action="store_true")
    parser.add_argument("--recalibrate-on-val", action="store_true",
                        help="Explicitly tune THIS checkpoint on validation before frozen evaluation (for legacy checkpoints)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    dataset = str(config["dataset"]["name"])
    bind_dataset_workspace(
        root, config, dataset,
        args.manifest_dir or f"datasets/{dataset}/ready",
    )
    positive_label, class_names = validate_dataset_semantics(config)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    # Fail closed before constructing a large model: never silently use 0.5 or
    # a legacy Fake-F1 threshold when a Macro-F1 protocol is requested.
    if not args.recalibrate_on_val:
        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_threshold_selection(metadata)
        del metadata
    if config["model"]["architecture_version"] == "qwen_lora_lgled_masked_r1" and args.split == "test" and not args.frozen:
        raise ValueError("R1 final test requires explicit --frozen")
    if config["model"]["architecture_version"] == "qwen_lora_lgled_masked_r1":
        from mmfnd.r1_cache import verify_backbone_identity
        verify_backbone_identity(root, config)
    seed_everything(int(config["seed"]))
    device = get_device()
    processor = build_processor(root, config)
    model = build_model(config).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    output_dir = (
        resolve_path(root, config["train"]["output_dir"])
        / "manual_evaluation" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        / checkpoint_path.stem
    )
    precision = str(config["train"].get("precision", "fp32"))
    if args.recalibrate_on_val:
        val_loader = build_loader(root, config, "val", processor)
        val_metrics, val_predictions, _ = evaluate(
            model, val_loader, device, positive_label, class_names, precision=precision,
            split="val", tune_threshold=True,
            checkpoint_reference={"kind": "recalibrated_checkpoint", "source_checkpoint": str(checkpoint_path.resolve()),
                                  "source_epoch": checkpoint.get("epoch")},
        )
        calibrated_path = output_dir / "validation_calibrated.pth"
        save_checkpoint(calibrated_path, model, None, None, checkpoint["epoch"], val_metrics, config)
        selection = checkpoint_threshold_selection(load_checkpoint(calibrated_path, model, device))
        dump_json(val_metrics, output_dir / "val/metrics.json")
        dump_json(evaluation_info(val_metrics), output_dir / "val/evaluation_info.json")
        write_jsonl(val_predictions, output_dir / "val/predictions.jsonl")
        log_evaluation(val_metrics)
    else:
        selection = checkpoint_threshold_selection(checkpoint)
    if args.recalibrate_on_val and args.split == "val":
        metrics, predictions = val_metrics, val_predictions
    else:
        loader = build_loader(root, config, args.split, processor)
        metrics, predictions, _ = evaluate(
            model, loader, device, positive_label, class_names, precision=precision,
            split=args.split, threshold_selection=selection,
        )
        log_evaluation(metrics)
    dump_json(metrics, output_dir / args.split / "metrics.json")
    write_jsonl(predictions, output_dir / args.split / "predictions.jsonl")
    dump_json(evaluation_info(metrics), output_dir / args.split / "evaluation_info.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"checkpoint_epoch={checkpoint.get('epoch')}; results={output_dir}")


if __name__ == "__main__":
    main()
