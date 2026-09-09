#!/usr/bin/env python3
"""Export sample-level LG-LED case-study diagnostics as JSONL."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm

from mmfnd.data import move_batch
from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import autocast_context, lgled_sample_diagnostics, load_checkpoint
from mmfnd.factory import build_loader, build_processor
from mmfnd.model import ExplainableMMFND
from mmfnd.utils import get_device, load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Export LG-LED diagnostics")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest-dir")
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    dataset = str(config["dataset"]["name"])
    bind_dataset_workspace(
        root, config, dataset,
        args.manifest_dir or f"datasets/{dataset}/ready",
    )
    positive_label, _ = validate_dataset_semantics(config)
    negative_label = 1 - positive_label
    device = get_device()
    precision = str(config["train"].get("precision", "bf16"))
    processor = build_processor(root, config)
    loader = build_loader(root, config, args.split, processor)
    model = ExplainableMMFND(config).to(device)
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    checkpoint = load_checkpoint(checkpoint_path, model, device)
    threshold = float(checkpoint.get("decision_threshold", 0.5))
    model.eval()
    output = args.output or (
        checkpoint_path.resolve().parent.parent / "lgled_diagnostics"
        / f"{args.split}.jsonl"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output.open("w", encoding="utf-8") as file:
        for raw_batch in tqdm(loader, desc="export LG-LED"):
            batch = move_batch(raw_batch, device)
            with torch.no_grad(), autocast_context(device, precision):
                outputs = model(batch)
            probabilities = outputs["logits"].float().softmax(dim=-1)
            for index, sample_id in enumerate(batch["ids"]):
                positive_probability = float(probabilities[index, positive_label])
                prediction = positive_label if positive_probability >= threshold else negative_label
                row = {
                    "sample_id": sample_id,
                    "label": int(batch["labels"][index]),
                    "prediction": int(prediction),
                    "positive_probability": positive_probability,
                    "lgled": lgled_sample_diagnostics(outputs, index),
                }
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
                if args.max_samples and written >= args.max_samples:
                    print(f"written={written}; output={output.resolve()}")
                    return
    print(f"written={written}; output={output.resolve()}")


if __name__ == "__main__":
    main()
