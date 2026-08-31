#!/usr/bin/env python3
"""One real-batch smoke test for the intrinsic-evidence architecture."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile

import torch
from torch.optim import AdamW

from mmfnd.data import move_batch
from mmfnd.dataset_contract import normalize_runtime_paths, validate_dataset_semantics
from mmfnd.engine import load_checkpoint, save_checkpoint
from mmfnd.factory import build_loader, build_processor
from mmfnd.model import ExplainableMMFND, multimodal_loss
from mmfnd.utils import get_device, load_config, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/weibo21.json")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    normalize_runtime_paths(root, config)
    positive_label, _ = validate_dataset_semantics(config)
    seed_everything(int(config["seed"]))
    device = get_device()
    processor = build_processor(root, config)
    # Exercise the pairwise ranking branch with a normally mixed-label batch.
    loader = build_loader(root, config, "train", processor, shuffle=True)
    batch = move_batch(next(iter(loader)), device)
    forbidden = sorted(key for key in batch if "evidence" in key.lower())
    if forbidden:
        raise RuntimeError(f"External-evidence fields remain in batch: {forbidden}")

    model = ExplainableMMFND(config).to(device)
    model.train()
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(
        trainable,
        lr=float(config["train"]["head_learning_rate"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    outputs = model(batch)
    loss, components = multimodal_loss(
        outputs, batch["labels"], config["loss"],
        float(config["train"]["label_smoothing"]),
        positive_label=positive_label,
    )
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {loss}; components={components}")
    loss.backward()
    invalid_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None
        and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    if invalid_gradients:
        raise FloatingPointError(
            f"Non-finite gradients: {invalid_gradients[:10]}"
        )
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    if not torch.isfinite(gradient_norm):
        raise FloatingPointError(f"Non-finite gradient norm: {gradient_norm}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # Verify that the compact trainable-only checkpoint restores exact weights.
    first_name, first_parameter = next(
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    expected = first_parameter.detach().clone()
    with tempfile.TemporaryDirectory(prefix="cute_fnd_smoke_") as directory:
        checkpoint_path = Path(directory) / "roundtrip.pth"
        save_checkpoint(
            checkpoint_path, model, optimizer, None, 0, {"smoke": True}, config,
            decision_threshold=0.5,
        )
        with torch.no_grad():
            first_parameter.zero_()
        restored_checkpoint = load_checkpoint(checkpoint_path, model, device)
        optimizer.load_state_dict(restored_checkpoint["optimizer_state_dict"])
        restored = dict(model.named_parameters())[first_name].detach()
        if not torch.equal(expected, restored):
            raise RuntimeError(f"Checkpoint round-trip mismatch: {first_name}")

    model.eval()
    with torch.no_grad():
        ablated = {
            name: model(batch, ablate_component=name)["logits"].shape
            for name in ("text", "image", "intrinsic_event")
        }
    report = {
        "device": str(device),
        "batch_size": int(batch["labels"].size(0)),
        "images": int(batch["pixel_values"].size(0)),
        "logits_shape": list(outputs["logits"].shape),
        "modality_weights_shape": list(outputs["modality_weights"].shape),
        "mean_modality_weights": {
            name: float(outputs["modality_weights"][:, index].mean().detach())
            for index, name in enumerate(("text", "image", "intrinsic_event"))
        },
        "mean_uncertainty_analysis": {
            "overall_score": float(outputs["uncertainty"].mean().detach()),
            "decision_margin": float(
                outputs["uncertainty_components"][:, 0].mean().detach()
            ),
            "view_conflict": float(
                outputs["uncertainty_components"][:, 1].mean().detach()
            ),
            "intervention_sensitivity": float(
                outputs["uncertainty_components"][:, 2].mean().detach()
            ),
            "graph_ambiguity": float(
                outputs["uncertainty_components"][:, 3].mean().detach()
            ),
            "component_weights": outputs[
                "uncertainty_component_weights"
            ].detach().cpu().tolist(),
            "mean_temperature": float(
                outputs["uncertainty_temperature"].mean().detach()
            ),
            "mean_residual_correction_norm": float(
                outputs["uncertainty_correction_norm"].mean().detach()
            ),
        },
        "alignment_attention_shape": list(
            outputs["intrinsic_alignment_attention"].shape
        ),
        "graph_adjacency_shape": list(outputs["graph_adjacency"].shape),
        "mean_graph_node_weights": {
            name: float(outputs["graph_node_weights"][:, index].mean().detach())
            for index, name in enumerate(model.evidence_graph.node_names)
        },
        "graph_residual_scale": float(
            outputs["graph_residual_scale"].detach()
        ),
        "ablation_shapes": {name: list(shape) for name, shape in ablated.items()},
        "loss": float(loss.detach()),
        "loss_components": components,
        "gradient_norm_before_clip": float(gradient_norm),
        "optimizer_step": "passed",
        "checkpoint_roundtrip": "passed",
        "forbidden_input_fields": forbidden,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
