#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch

from mmfnd.data import move_batch
from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import autocast_context, lgled_sample_diagnostics, load_checkpoint
from mmfnd.factory import build_loader, build_processor
from mmfnd.image_preprocessing import load_preprocessed_image
from mmfnd.model import ExplainableMMFND
from mmfnd.utils import get_device, load_config, resolve_path


def configure_chinese_font() -> str:
    """Select an installed CJK font and avoid missing Chinese glyphs in figures."""
    candidates = (
        "PingFang SC",          # macOS
        "Heiti SC",            # macOS
        "STHeiti",             # macOS
        "Arial Unicode MS",    # macOS / Windows
        "Noto Sans CJK SC",    # Linux
        "Source Han Sans SC",  # Linux / Adobe Source Han Sans
        "Microsoft YaHei",     # Windows
        "SimHei",              # Windows
        "WenQuanYi Micro Hei", # Linux
    )
    installed = {font.name for font in font_manager.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")

    # Keep DejaVu Sans as the fallback for Latin characters and mathematical symbols.
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [selected, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    return selected


def image_occlusion_map(
    model, batch, sample_index: int, grid: int = 7,
    target_class: int | None = None,
) -> np.ndarray:
    """Model-agnostic faithfulness map: probability drop after patch occlusion."""
    device = next(model.parameters()).device
    qwen_dtype = next(model.encoder.shared_qwen_model().parameters()).dtype
    precision = "bf16" if qwen_dtype == torch.bfloat16 else (
        "fp16" if qwen_dtype == torch.float16 else "fp32"
    )
    with torch.no_grad(), autocast_context(device, precision):
        base = model(batch)["logits"].softmax(-1)
        predicted = (
            int(base[sample_index].argmax())
            if target_class is None else int(target_class)
        )
        base_score = float(base[sample_index, predicted])
    owners = batch["image_owner"]
    image_indices = torch.where(owners == sample_index)[0]
    if len(image_indices) == 0:
        return np.zeros((grid, grid), dtype=np.float32)
    target_index = int(image_indices[0])
    pixels = batch["pixel_values"]
    height, width = pixels.shape[-2:]
    heat = np.zeros((grid, grid), dtype=np.float32)
    for row in range(grid):
        for col in range(grid):
            modified = pixels.clone()
            y0, y1 = row * height // grid, (row + 1) * height // grid
            x0, x1 = col * width // grid, (col + 1) * width // grid
            modified[target_index, :, y0:y1, x0:x1] = 0
            changed = dict(batch)
            changed["pixel_values"] = modified
            with torch.no_grad(), autocast_context(device, precision):
                score = float(model(changed)["logits"].softmax(-1)[sample_index, predicted])
            heat[row, col] = max(0.0, base_score - score)
    return heat / max(float(heat.max()), 1e-8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate faithful visual explanations")
    parser.add_argument("--config", default="configs/datasets/weibo21.json")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--grid", type=int, default=7)
    parser.add_argument("--manifest-dir")
    args = parser.parse_args()
    chinese_font = configure_chinese_font()
    print(f"matplotlib_chinese_font={chinese_font}")
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    dataset = str(config["dataset"]["name"])
    bind_dataset_workspace(
        root, config, dataset,
        args.manifest_dir or f"datasets/{dataset}/ready",
    )
    positive_label, class_names = validate_dataset_semantics(config)
    negative_label = 1 - positive_label
    device = get_device()
    processor = build_processor(root, config)
    loader = build_loader(root, config, args.split, processor)
    model = ExplainableMMFND(config).to(device)
    checkpoint = load_checkpoint(args.checkpoint, model, device)
    decision_threshold = float(checkpoint.get("decision_threshold", 0.5))
    model.eval()
    precision = str(config["train"].get("precision", "bf16"))
    output_dir = (
        args.checkpoint.resolve().parent.parent / "explanations"
        if args.checkpoint.parent.name == "checkpoints"
        else resolve_path(root, config["train"]["output_dir"]) / "manual_explanations"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = resolve_path(
        root, config["data"].get("image_root", config["data"]["root"])
    )

    produced = 0
    for raw_batch in loader:
        batch = move_batch(raw_batch, device)
        with torch.no_grad(), autocast_context(device, precision):
            outputs = model(batch)
            probs = outputs["logits"].softmax(-1)
            ablated_probs = {
                component: model(
                    batch, ablate_component=component
                )["logits"].softmax(-1)
                for component in ("text", "image", "intrinsic_event")
            }
        for index, sample_id in enumerate(batch["ids"]):
            if produced >= args.samples:
                return
            predicted_class = (
                positive_label
                if float(probs[index, positive_label]) >= decision_threshold
                else negative_label
            )
            heat = image_occlusion_map(
                model, batch, index, args.grid, target_class=predicted_class
            )
            image_path = data_root / batch["image_paths"][index][0]
            image = load_preprocessed_image(image_path, {}, train=False)
            try:
                fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
                axes[0].imshow(image)
                axes[0].axis("off")
                axes[0].set_title("原始图片")
                axes[1].imshow(image)
                axes[1].imshow(heat, cmap="jet", alpha=0.45, interpolation="bilinear",
                               extent=(0, image.width, image.height, 0))
                axes[1].axis("off")
                axes[1].set_title("遮挡忠实性热力图")
                fig.tight_layout()
                fig.savefig(output_dir / f"{sample_id}.png", dpi=180, bbox_inches="tight")
                plt.close(fig)
            finally:
                image.close()
            explanation = {
                "id": sample_id,
                "label": int(batch["labels"][index]),
                "prediction": predicted_class,
                "decision_threshold": decision_threshold,
                "positive_label": positive_label,
                "positive_class": class_names[positive_label],
                "positive_probability": float(probs[index, positive_label]),
                "class_probabilities": {
                    class_names[label]: float(probs[index, label])
                    for label in (0, 1)
                },
                "uncertainty": float(outputs["uncertainty"][index]),
                "uncertainty_analysis": {
                    "overall_score": float(outputs["uncertainty"][index].item()),
                    "decision_margin": float(
                        outputs["uncertainty_components"][index, 0].item()
                    ),
                    "view_conflict": float(
                        outputs["uncertainty_components"][index, 1].item()
                    ),
                    "intervention_sensitivity": float(
                        outputs["uncertainty_components"][index, 2].item()
                    ),
                    "latent_relation_uncertainty": float(
                        outputs["uncertainty_components"][index, 3].item()
                    ),
                    "component_weights": outputs[
                        "uncertainty_component_weights"
                    ].cpu().tolist(),
                    "temperature": float(
                        outputs["uncertainty_temperature"][index].item()
                    ),
                    "residual_correction_norm": float(
                        outputs["uncertainty_correction_norm"][index].item()
                    ),
                },
                "modality_weights": dict(zip(
                    ("text", "image", "intrinsic_event"),
                    outputs["modality_weights"][index].cpu().tolist(),
                )),
                "intrinsic_evidence": {
                    "text_relation_ambiguity": float(
                        outputs["text_relation_ambiguity"][index].item()
                    ),
                    "event_contradiction": float(
                        outputs["event_contradiction"][index].item()
                    ),
                    "visual_evidence_inconsistency": float(
                        outputs["visual_evidence_inconsistency"][index].item()
                    ),
                    "multi_image_dispersion": float(
                        outputs["multi_image_dispersion"][index].item()
                    ),
                    "multi_image_available": bool(
                        outputs["multi_image_available"][index].item()
                    ),
                    "image_count": len(batch["image_paths"][index]),
                },
                "causal_gate_effects": dict(zip(
                    ("text", "image", "intrinsic_event"),
                    outputs["causal_gate_effects"][index].cpu().tolist(),
                )),
                "lgled": lgled_sample_diagnostics(outputs, index),
                "causal_intervention_probability_drop": {
                    component: float(
                        probs[index, predicted_class].item()
                        - values[index, predicted_class].item()
                    )
                    for component, values in ablated_probs.items()
                },
                "category": batch["categories"][index],
                "image": str(image_path),
                "text": batch["texts"][index],
                "method": "patch occlusion; higher heat means larger predicted-probability drop",
            }
            (output_dir / f"{sample_id}.json").write_text(json.dumps(explanation, ensure_ascii=False, indent=2), encoding="utf-8")
            produced += 1


if __name__ == "__main__":
    main()
