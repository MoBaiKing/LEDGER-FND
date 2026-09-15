#!/usr/bin/env python3
"""Render one prediction and faithful model-internal explanations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from explain import configure_chinese_font, image_occlusion_map
from mmfnd.data import move_batch
from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import autocast_context, lgled_sample_diagnostics, load_checkpoint
from mmfnd.evaluation import checkpoint_threshold_selection
from mmfnd.factory import build_loader, build_processor
from mmfnd.image_preprocessing import load_preprocessed_image
from mmfnd.model import ExplainableMMFND
from mmfnd.utils import get_device, load_config, resolve_path, seed_everything


EVIDENCE_ZH = {
    "text": "文本", "vision": "图像",
    "intrinsic_feature": "内生事件", "interaction": "图文交互",
}


def find_latest_final(root: Path, config: dict) -> Path:
    output_root = resolve_path(root, config["train"]["output_dir"])
    checkpoints = list(output_root.glob("*/checkpoints/final_averaged.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"在 {output_root} 中找不到 final_averaged.pth，请通过 --checkpoint 指定。")
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def select_sample(loader, sample_id: str | None, sample_index: int) -> tuple[dict, int]:
    seen = 0
    for batch in loader:
        if sample_id is not None and sample_id in batch["ids"]:
            return batch, batch["ids"].index(sample_id)
        if sample_id is None and seen <= sample_index < seen + len(batch["ids"]):
            return batch, sample_index - seen
        seen += len(batch["ids"])
    target = f"id={sample_id}" if sample_id is not None else f"index={sample_index}"
    raise IndexError(f"数据分区中找不到样本 {target}")


def uncertainty_level(value: float) -> tuple[str, str]:
    if value < 0.2:
        return "较低", "low"
    if value < 0.5:
        return "中等", "moderate"
    return "较高", "high"


def format_explanation(info: dict, chinese: bool) -> str:
    names = info["class_names"]
    lgled = info["lgled"]
    evidence_weights = ("、" if chinese else ", ").join(
        f"{EVIDENCE_ZH.get(name, name) if chinese else name}={values['final_weight']:.3f}"
        for name, values in lgled["evidence"].items()
    )
    probabilities = ("，" if chinese else ", ").join(
        f"{names[str(label)]}={info['class_probabilities'][names[str(label)]]:.4f}" for label in (0, 1)
    )
    if chinese:
        return "\n".join([
            "========== CUTE-FND 中文可解释性结果 ==========", f"样本ID：{info['id']}",
            f"新闻文本：{info['text']}", f"真实标签：{names[str(info['label'])]}（{info['label']}）",
            f"预测标签：{names[str(info['prediction'])]}（{info['prediction']}）", f"分类概率：{probabilities}",
            f"正类：{info['positive_class']}（label={info['positive_label']}），判定阈值={info['decision_threshold']:.4f}",
            f"预测不确定性：{info['uncertainty']:.4f}（{info['uncertainty_level_zh']}）",
            f"不确定性分量：{info['uncertainty_analysis']}",
            f"三路可靠性探针权重：{info['modality_weights']}",
            f"LG-LED 最终四路证据权重：{evidence_weights}",
            f"内生证据：{info['intrinsic_evidence']}",
            f"因果门控效应：{info['causal_gate_effects']}",
            f"LG-LED disagreement={lgled['sample_disagreement']:.4f}，routing gate={lgled['routing_gate']:.4f}",
            f"数据子群（仅元数据）：{info['category']}",
            "视觉解释：热力图越红，遮挡该区域后当前预测类别概率下降越明显。",
            "注意：删除干预只衡量模型内部依赖，不代表现实世界因果关系。",
        ])
    return "\n".join([
        "========== CUTE-FND English Explanation ==========", f"Sample ID: {info['id']}",
        f"News text: {info['text']}", f"Ground truth: {names[str(info['label'])]} ({info['label']})",
        f"Prediction: {names[str(info['prediction'])]} ({info['prediction']})", f"Class probabilities: {probabilities}",
        f"Positive class: {info['positive_class']} (label={info['positive_label']}), threshold={info['decision_threshold']:.4f}",
        f"Predictive uncertainty: {info['uncertainty']:.4f} ({info['uncertainty_level_en']})",
        f"Uncertainty components: {info['uncertainty_analysis']}",
        f"Three-view reliability-probe weights: {info['modality_weights']}",
        f"LG-LED final evidence weights: {evidence_weights}",
        f"Intrinsic evidence: {info['intrinsic_evidence']}",
        f"Causal gate effects: {info['causal_gate_effects']}",
        f"LG-LED disagreement={lgled['sample_disagreement']:.4f}, routing gate={lgled['routing_gate']:.4f}",
        f"Metadata subgroup: {info['category']}",
        "Red regions produce a larger current-class probability drop when occluded.",
        "Deletion interventions measure internal dependence, not real-world causality.",
    ])


def save_plot(image_path: Path, heat: np.ndarray, info: dict, output_path: Path) -> None:
    configure_chinese_font()
    image = load_preprocessed_image(image_path, {}, train=False)
    try:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        axes[0].imshow(image); axes[0].set_title("原始图片 / Original image"); axes[0].axis("off")
        axes[1].imshow(image)
        overlay = axes[1].imshow(heat, cmap="jet", alpha=0.45, interpolation="bilinear",
                                 extent=(0, image.width, image.height, 0), vmin=0.0, vmax=1.0)
        axes[1].set_title("遮挡忠实性热力图 / Occlusion map"); axes[1].axis("off")
        fig.colorbar(overlay, ax=axes[1], fraction=0.046, pad=0.04)
        names = info["class_names"]
        fig.suptitle(f"{info['id']} | True: {names[str(info['label'])]} | Pred: {names[str(info['prediction'])]} | P({info['positive_class']})={info['positive_probability']:.4f}")
        fig.tight_layout(); fig.savefig(output_path, dpi=220, bbox_inches="tight"); plt.close(fig)
    finally:
        image.close()


def save_lgled_plot(info: dict, output_path: Path) -> None:
    configure_chinese_font()
    lgled = info["lgled"]
    labels = list(lgled["pairs"])
    relation_names = lgled["relation_order"]
    beliefs = np.asarray([
        [lgled["pairs"][pair][name] for name in relation_names]
        for pair in labels
    ], dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    rendered = ax.imshow(beliefs, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(relation_names)), relation_names, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_title("LG-LED 潜在关系信念 / Latent relation beliefs")
    fig.colorbar(rendered, ax=ax, label="Dirichlet mean belief")
    fig.tight_layout(); fig.savefig(output_path, dpi=220, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Display one explainable prediction")
    parser.add_argument("--config", default="configs/datasets/weibo21.json")
    parser.add_argument("--checkpoint", type=Path, help="默认选择最近的 final_averaged.pth")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--sample-id"); parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--grid", type=int, default=7); parser.add_argument("--output-dir", default="outputs/display")
    parser.add_argument("--manifest-dir")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    dataset = str(config["dataset"]["name"])
    bind_dataset_workspace(
        root, config, dataset,
        args.manifest_dir or f"datasets/{dataset}/ready",
    )
    positive_label, class_names_int = validate_dataset_semantics(config)
    class_names = {str(key): value for key, value in class_names_int.items()}
    seed_everything(int(config["seed"])); device = get_device()
    checkpoint_path = args.checkpoint or find_latest_final(root, config)
    checkpoint_path = checkpoint_path if checkpoint_path.is_absolute() else root / checkpoint_path
    output_dir = resolve_path(root, args.output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    processor = build_processor(root, config); loader = build_loader(root, config, args.split, processor)
    raw_batch, index = select_sample(loader, args.sample_id, args.sample_index); batch = move_batch(raw_batch, device)
    model = ExplainableMMFND(config).to(device); checkpoint = load_checkpoint(checkpoint_path, model, device)
    threshold_selection = checkpoint_threshold_selection(checkpoint)
    decision_threshold = threshold_selection["threshold"]; model.eval()
    precision = str(config["train"].get("precision", "bf16"))
    with torch.no_grad(), autocast_context(device, precision):
        outputs = model(batch); probabilities = outputs["logits"].float().softmax(dim=-1)
        ablated = {component: model(batch, ablate_component=component)["logits"].float().softmax(dim=-1)
                   for component in ("text", "image", "intrinsic_event")}
    prediction = positive_label if float(probabilities[index, positive_label]) >= decision_threshold else 1 - positive_label
    label = int(batch["labels"][index].item()); uncertainty = float(outputs["uncertainty"][index].item())
    level_zh, level_en = uncertainty_level(uncertainty)
    image_root = resolve_path(root, config["data"].get("image_root", config["data"]["root"]))
    image_path = image_root / batch["image_paths"][index][0]
    lgled = lgled_sample_diagnostics(outputs, index)
    info = {"id": batch["ids"][index], "split": args.split, "checkpoint": str(checkpoint_path.resolve()),
            "checkpoint_epoch": int(checkpoint.get("epoch", -1)), "text": batch["texts"][index], "image": str(image_path),
            "label": label, "prediction": prediction, "correct": prediction == label, "class_names": class_names,
            "positive_label": positive_label, "positive_class": class_names_int[positive_label],
            "positive_probability": float(probabilities[index, positive_label]),
            "class_probabilities": {class_names_int[item]: float(probabilities[index, item]) for item in (0, 1)},
            "decision_threshold": decision_threshold, "uncertainty": uncertainty,
            "threshold": decision_threshold, "threshold_source": "validation", "threshold_objective": "macro_f1",
            "evaluation_protocol": threshold_selection["evaluation_protocol"], "threshold_selection": threshold_selection,
            "uncertainty_level_zh": level_zh, "uncertainty_level_en": level_en,
            "uncertainty_analysis": {
                name: float(outputs["uncertainty_components"][index, position])
                for position, name in enumerate(("decision_margin", "view_conflict", "intervention_sensitivity", "latent_relation_uncertainty"))
            } | {
                "component_weights": outputs["uncertainty_component_weights"].float().cpu().tolist(),
                "temperature": float(outputs["uncertainty_temperature"][index]),
                "residual_correction_norm": float(outputs["uncertainty_correction_norm"][index]),
            },
            "modality_weights": {name: float(outputs["modality_weights"][index, position])
                                 for position, name in enumerate(("text", "image", "intrinsic_event"))},
            "intrinsic_evidence": {
                name: float(outputs[name][index]) for name in (
                    "text_relation_ambiguity", "event_contradiction",
                    "visual_evidence_inconsistency", "multi_image_dispersion",
                )
            } | {
                "multi_image_available": bool(
                    outputs["multi_image_available"][index].item()
                ),
                "image_count": len(batch["image_paths"][index]),
            },
            "causal_gate_effects": {
                name: float(outputs["causal_gate_effects"][index, position])
                for position, name in enumerate(("text", "image", "intrinsic_event"))
            },
            "lgled": lgled,
            "causal_intervention_probability_drop": {component: float(probabilities[index, prediction] - values[index, prediction])
                                                       for component, values in ablated.items()},
            "category": batch["categories"][index]}
    heat = image_occlusion_map(model, batch, index, args.grid, target_class=prediction)
    stem = info["id"]; save_plot(image_path, heat, info, output_dir / f"{stem}_explanation.png")
    save_lgled_plot(info, output_dir / f"{stem}_lgled_relations.png")
    zh_text = format_explanation(info, True); en_text = format_explanation(info, False)
    (output_dir / f"{stem}_explanation_zh.txt").write_text(zh_text + "\n", encoding="utf-8")
    (output_dir / f"{stem}_explanation_en.txt").write_text(en_text + "\n", encoding="utf-8")
    (output_dir / f"{stem}_explanation.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(zh_text); print(f"\noutput_dir={output_dir}")


if __name__ == "__main__":
    main()
