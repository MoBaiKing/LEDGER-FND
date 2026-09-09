#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import evaluate, load_checkpoint, write_jsonl
from mmfnd.factory import build_loader, build_processor
from mmfnd.model import ExplainableMMFND
from mmfnd.utils import dump_json, get_device, load_config, resolve_path, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one frozen checkpoint and threshold")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--manifest-dir")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    dataset = str(config["dataset"]["name"])
    bind_dataset_workspace(
        root, config, dataset,
        args.manifest_dir or f"datasets/{dataset}/ready",
    )
    positive_label, class_names = validate_dataset_semantics(config)
    seed_everything(int(config["seed"]))
    device = get_device()
    processor = build_processor(root, config)
    loader = build_loader(root, config, args.split, processor)
    model = ExplainableMMFND(config).to(device)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    threshold = float(checkpoint.get("decision_threshold", 0.5))
    metrics, predictions, _ = evaluate(
        model, loader, device, positive_label, class_names, threshold,
        str(config["train"].get("precision", "fp32")),
    )
    output_dir = (
        resolve_path(root, config["train"]["output_dir"])
        / "manual_evaluation" / datetime.now().strftime("%Y%m%d_%H%M%S")
        / checkpoint_path.stem / args.split
    )
    dump_json(metrics, output_dir / "metrics.json")
    write_jsonl(predictions, output_dir / "predictions.jsonl")
    dump_json({"decision_threshold": threshold}, output_dir / "evaluation_info.json")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"checkpoint_epoch={checkpoint.get('epoch')}; results={output_dir}")


if __name__ == "__main__":
    main()
