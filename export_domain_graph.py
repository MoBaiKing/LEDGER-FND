#!/usr/bin/env python3
"""Export evidence graphs grouped by category metadata (never used as input)."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from mmfnd.data import move_batch
from mmfnd.dataset_contract import normalize_runtime_paths
from mmfnd.engine import load_checkpoint
from mmfnd.factory import build_loader, build_processor
from mmfnd.model import ExplainableMMFND
from mmfnd.utils import get_device, load_config, resolve_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export category-averaged internal evidence graphs")
    parser.add_argument("--config", default="configs/datasets/weibo21.json")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config); normalize_runtime_paths(root, config)
    device = get_device(); processor = build_processor(root, config)
    loader = build_loader(root, config, args.split, processor)
    model = ExplainableMMFND(config).to(device); load_checkpoint(args.checkpoint, model, device); model.eval()

    adjacency_sum: dict[str, torch.Tensor] = {}
    node_weight_sum: dict[str, torch.Tensor] = {}
    counts: dict[str, int] = defaultdict(int)
    with torch.no_grad():
        for raw_batch in tqdm(loader, desc="category evidence graph"):
            batch = move_batch(raw_batch, device); outputs = model(batch)
            for row, raw_category in enumerate(batch["categories"]):
                category = raw_category or "overall"
                if category not in adjacency_sum:
                    adjacency_sum[category] = torch.zeros_like(outputs["graph_adjacency"][row])
                    node_weight_sum[category] = torch.zeros_like(outputs["graph_node_weights"][row])
                adjacency_sum[category] += outputs["graph_adjacency"][row]
                node_weight_sum[category] += outputs["graph_node_weights"][row]
                counts[category] += 1

    categories = sorted(counts)
    adjacency = [(adjacency_sum[name] / counts[name]).cpu().tolist() for name in categories]
    node_weights = [(node_weight_sum[name] / counts[name]).cpu().tolist() for name in categories]
    output_dir = (args.checkpoint.resolve().parent.parent / "category_evidence_graph"
                  if args.checkpoint.parent.name == "checkpoints"
                  else resolve_path(root, config["train"]["output_dir"]) / "manual_category_evidence_graph")
    output_dir.mkdir(parents=True, exist_ok=True)
    node_names = list(model.evidence_graph.node_names)
    payload = {"grouping": "category_metadata_only", "categories": categories,
               "node_names": node_names, "split": args.split, "samples_per_category": dict(counts),
               "category_mean_adjacency": adjacency, "category_mean_node_weights": node_weights}
    (output_dir / "category_evidence_graph.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    total = sum(counts.values())
    overall = sum(adjacency_sum.values()) / max(total, 1)
    fig, axis = plt.subplots(figsize=(7, 6)); image = axis.imshow(overall.cpu().numpy(), cmap="Blues", vmin=0.0, vmax=1.0)
    axis.set_xticks(range(len(node_names)), node_names, rotation=30, ha="right")
    axis.set_yticks(range(len(node_names)), node_names); axis.set_xlabel("Source evidence node")
    axis.set_ylabel("Updated evidence node"); axis.set_title("Mean Internal Evidence-Graph Adjacency")
    fig.colorbar(image, ax=axis, label="attention weight"); fig.tight_layout()
    fig.savefig(output_dir / "evidence_graph_heatmap.png", dpi=200); plt.close(fig)
    print(f"saved to {output_dir}")


if __name__ == "__main__":
    main()
