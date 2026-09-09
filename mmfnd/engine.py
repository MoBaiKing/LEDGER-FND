from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score
from tqdm import tqdm

from mmfnd.data import move_batch
from mmfnd.latent_evidence_deliberation import (
    EVIDENCE_ORDER,
    PAIR_ORDER,
    RELATION_ORDER,
)


def lgled_sample_diagnostics(outputs: dict, index: int) -> dict:
    """Convert one sample's LG-LED tensors to a JSON-serializable record."""
    relation_probabilities = outputs["relation_probs"][index].float().cpu()
    relation_uncertainty = outputs["relation_uncertainty"][index].float().cpu()
    relation_strength = outputs["relation_strength"][index].float().cpu()
    pairs = {}
    for pair_index, pair in enumerate(PAIR_ORDER):
        name = "-".join(pair)
        pairs[name] = {
            relation: float(relation_probabilities[pair_index, relation_index])
            for relation_index, relation in enumerate(RELATION_ORDER)
        } | {
            "uncertainty": float(relation_uncertainty[pair_index]),
            "strength": float(relation_strength[pair_index]),
        }
    evidence = {}
    for evidence_index, name in enumerate(EVIDENCE_ORDER):
        evidence[name] = {
            "confidence": float(outputs["evidence_confidence"][index, evidence_index]),
            "deviation": float(outputs["evidence_deviation"][index, evidence_index]),
            "minority_score": float(outputs["minority_score"][index, evidence_index]),
            "global_judge_weight": float(outputs["global_judge_weights"][index, evidence_index]),
            "deliberative_weight": float(outputs["deliberative_weights"][index, evidence_index]),
            "direct_weight": float(outputs["direct_weights"][index, evidence_index]),
            "final_weight": float(outputs["final_evidence_weights"][index, evidence_index]),
        }
    return {
        "relation_order": list(RELATION_ORDER),
        "pairs": pairs,
        "evidence": evidence,
        "sample_disagreement": float(outputs["sample_disagreement"][index, 0]),
        "routing_gate": float(outputs["routing_gate"][index, 0]),
    }


def _update_lgled_totals(totals: dict[str, torch.Tensor | float],
                         outputs: dict) -> int:
    batch_size = int(outputs["relation_probs"].size(0))
    values = {
        "relations": outputs["relation_probs"].float().mean(dim=1).sum(dim=0),
        "relation_uncertainty": outputs["relation_uncertainty"].float().mean(dim=1).sum(),
        "relation_strength": outputs["relation_strength"].float().mean(dim=1).sum(),
        "confidence": outputs["evidence_confidence"].float().sum(dim=0),
        "deviation": outputs["evidence_deviation"].float().sum(dim=0),
        "minority": outputs["minority_score"].float().sum(dim=0),
        "global_weights": outputs["global_judge_weights"].float().sum(dim=0),
        "deliberative_weights": outputs["deliberative_weights"].float().sum(dim=0),
        "direct_weights": outputs["direct_weights"].float().sum(dim=0),
        "final_weights": outputs["final_evidence_weights"].float().sum(dim=0),
        "disagreement": outputs["sample_disagreement"].float().sum(),
        "routing": outputs["routing_gate"].float().sum(),
    }
    for name, value in values.items():
        detached = value.detach().cpu()
        totals[name] = totals.get(name, torch.zeros_like(detached)) + detached
    return batch_size


def _finalize_lgled_totals(totals: dict, count: int) -> dict:
    divisor = max(count, 1)
    relations = totals["relations"] / divisor
    result = {
        f"mean_{name}": float(relations[index])
        for index, name in enumerate(RELATION_ORDER)
    }
    result.update({
        "mean_relation_uncertainty": float(totals["relation_uncertainty"] / divisor),
        "mean_relation_strength": float(totals["relation_strength"] / divisor),
        "mean_disagreement": float(totals["disagreement"] / divisor),
        "mean_routing_gate": float(totals["routing"] / divisor),
    })
    for metric in (
        "confidence", "deviation", "minority", "global_weights",
        "deliberative_weights", "direct_weights", "final_weights",
    ):
        result[f"mean_evidence_{metric}"] = {
            name: float(totals[metric][index] / divisor)
            for index, name in enumerate(EVIDENCE_ORDER)
        }
    return result


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "fp32":
        return nullcontext()
    return torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16 if precision == "bf16" else torch.float16,
    )


def select_robust_threshold(
    labels, positive_probabilities, positive_label: int,
    threshold_min: float = 0.20, threshold_max: float = 0.80,
    threshold_step: float = 0.01, plateau_delta: float = 0.002,
) -> tuple[float, float]:
    """Select the threshold nearest 0.5 on the near-optimal fixed-grid plateau."""
    if threshold_step <= 0 or threshold_min > threshold_max:
        raise ValueError("invalid threshold grid")
    binary = (np.asarray(labels, dtype=np.int64) == positive_label).astype(int)
    probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    grid = np.arange(
        threshold_min, threshold_max + threshold_step * 0.5, threshold_step,
        dtype=np.float64,
    )
    scores = np.asarray([
        f1_score(binary, probabilities >= threshold, zero_division=0)
        for threshold in grid
    ])
    maximum = float(scores.max())
    plateau = grid[scores >= maximum - plateau_delta - 1e-12]
    selected = min(plateau.tolist(), key=lambda value: (abs(value - 0.5), value))
    return float(round(selected, 10)), maximum


def classification_metrics(
    labels, positive_probabilities, decision_threshold: float,
    positive_label: int, class_names: dict[int, str], argmax_predictions=None,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    probabilities = np.asarray(positive_probabilities, dtype=np.float64)
    negative_label = 1 - positive_label
    predictions = np.where(
        probabilities >= decision_threshold, positive_label, negative_label
    ).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0
    )
    binary = (labels == positive_label).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        binary, predictions == positive_label, labels=[0, 1]
    ).ravel()
    clipped = np.clip(probabilities, 1e-7, 1.0 - 1e-7)
    threshold_accuracy = float(accuracy_score(labels, predictions))
    argmax_accuracy = (
        float(accuracy_score(labels, argmax_predictions))
        if argmax_predictions is not None else threshold_accuracy
    )
    metrics = {
        "samples": int(labels.size),
        "decision_threshold": float(decision_threshold),
        "macro_f1": float(f1.mean()),
        # Accuracy follows the original two-logit argmax contract. Thresholded
        # accuracy is retained separately for calibration diagnostics.
        "accuracy": argmax_accuracy,
        "threshold_accuracy": threshold_accuracy,
        "auc": float(roc_auc_score(binary, probabilities)) if len(set(binary.tolist())) > 1 else None,
        "brier": float(np.mean((probabilities - binary) ** 2)),
        "nll": float(np.mean(-(binary * np.log(clipped) + (1 - binary) * np.log(1 - clipped)))),
        "positive_precision": float(precision[positive_label]),
        "positive_recall": float(recall[positive_label]),
        "positive_f1": float(f1[positive_label]),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }
    for label in (0, 1):
        name = class_names[label].strip().lower().replace(" ", "_")
        metrics.update({
            f"{name}_precision": float(precision[label]),
            f"{name}_recall": float(recall[label]),
            f"{name}_f1": float(f1[label]),
            f"{name}_support": int(support[label]),
        })
    metrics["argmax_accuracy"] = argmax_accuracy
    return metrics


@torch.no_grad()
def evaluate(
    model, loader, device: torch.device, positive_label: int,
    class_names: dict[int, str], decision_threshold: float = 0.5,
    precision: str = "fp32", show_progress: bool = True,
) -> tuple[dict, list[dict], float]:
    raw_model = unwrap_model(model)
    raw_model.eval()
    labels, positive_probabilities, argmax_predictions, rows = [], [], [], []
    lgled_totals: dict[str, torch.Tensor | float] = {}
    lgled_count = 0
    export_diagnostics = bool(
        raw_model.runtime_config["model"].get("lgled", {}).get(
            "export_diagnostics", False
        )
    )
    iterator = tqdm(loader, desc="evaluate", leave=False, disable=not show_progress)
    for batch in iterator:
        batch = move_batch(batch, device)
        with autocast_context(device, precision):
            outputs = raw_model(batch)
        lgled_count += _update_lgled_totals(lgled_totals, outputs)
        probs = outputs["logits"].float().softmax(-1)
        batch_labels = batch["labels"].cpu().tolist()
        batch_positive = probs[:, positive_label].cpu().tolist()
        batch_argmax = probs.argmax(dim=-1).cpu().tolist()
        labels.extend(batch_labels)
        positive_probabilities.extend(batch_positive)
        argmax_predictions.extend(batch_argmax)
        for index, sample_id in enumerate(batch["ids"]):
            rows.append({
                "id": sample_id,
                "label": int(batch_labels[index]),
                "positive_label": int(positive_label),
                "positive_class": class_names[positive_label],
                "positive_probability": float(batch_positive[index]),
                "class_probabilities": {
                    class_names[label]: float(probs[index, label].item())
                    for label in (0, 1)
                },
                "uncertainty": float(outputs["uncertainty"][index].item()),
                "uncertainty_analysis": {
                    "overall_score": float(outputs["uncertainty"][index].item()),
                    "decision_margin": float(outputs["uncertainty_components"][index, 0].item()),
                    "view_conflict": float(outputs["uncertainty_components"][index, 1].item()),
                    "intervention_sensitivity": float(outputs["uncertainty_components"][index, 2].item()),
                    "latent_relation_uncertainty": float(outputs["uncertainty_components"][index, 3].item()),
                    "component_weights": outputs["uncertainty_component_weights"].float().cpu().tolist(),
                    "temperature": float(outputs["uncertainty_temperature"][index].item()),
                    "residual_correction_norm": float(outputs["uncertainty_correction_norm"][index].item()),
                },
                "modality_weights": {
                    name: float(outputs["modality_weights"][index, position].item())
                    for position, name in enumerate(("text", "image", "intrinsic_event"))
                },
                "intrinsic_evidence": {
                    key: float(outputs[key][index].item()) for key in (
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
                    name: float(outputs["causal_gate_effects"][index, position].item())
                    for position, name in enumerate(("text", "image", "intrinsic_event"))
                },
                "image_paths": batch["image_paths"][index],
                "text": batch["texts"][index],
                "category": batch["categories"][index],
            })
            if export_diagnostics:
                rows[-1]["lgled"] = lgled_sample_diagnostics(outputs, index)
    metrics = classification_metrics(
        labels, positive_probabilities, decision_threshold, positive_label,
        class_names, argmax_predictions,
    )
    metrics["lgled"] = _finalize_lgled_totals(lgled_totals, lgled_count)
    negative_label = 1 - positive_label
    predictions = [
        positive_label if probability >= decision_threshold else negative_label
        for probability in positive_probabilities
    ]
    for row, prediction in zip(rows, predictions):
        row["prediction"] = int(prediction)
        row["decision_threshold"] = float(decision_threshold)
    return metrics, rows, float(decision_threshold)


def _checkpoint_contract(config: dict) -> dict:
    lgled = config["model"].get("lgled", {})
    return {
        "architecture_version": config["model"]["architecture_version"],
        "dataset": config["dataset"]["name"],
        "positive_label": int(config["dataset"]["positive_label"]),
        # Keep relocation possible while still rejecting a different backbone.
        "text_backbone": Path(str(config["model"]["text_backbone"])).name,
        "vision_backbone": Path(str(config["model"]["vision_backbone"])).name,
        "projector_type": lgled.get("projector_type", "shared"),
        "judge_type": lgled.get("judge_type", "qwen_latent"),
        "latent_judge_num_layers": int(lgled.get("latent_judge_num_layers", 2)),
    }


def save_checkpoint(
    path: Path, model, optimizer, scheduler, epoch: int, metrics: dict,
    config: dict, decision_threshold: float = 0.5, scaler=None,
    training_state: dict | None = None,
) -> None:
    raw_model = unwrap_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    trainable_names = {name for name, parameter in raw_model.named_parameters() if parameter.requires_grad}
    trainable_state = {
        name: value.detach().cpu() for name, value in raw_model.state_dict().items()
        if name in trainable_names
    }
    torch.save({
        "model_state_dict": trainable_state,
        "model_state_dict_format": "trainable_only",
        "trainable_parameter_names": sorted(trainable_names),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": int(epoch), "metrics": metrics, "config": config,
        "checkpoint_contract": _checkpoint_contract(config),
        "decision_threshold": float(decision_threshold),
        "training_state": training_state or {},
    }, path)


def load_checkpoint(path: Path, model, device, strict: bool = True) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    raw_model = unwrap_model(model)
    expected_version = getattr(raw_model, "architecture_version", None)
    actual_version = checkpoint.get("config", {}).get("model", {}).get("architecture_version")
    if expected_version is not None and actual_version != expected_version:
        raise ValueError(f"Checkpoint architecture {actual_version!r} is incompatible with {expected_version!r}.")
    current_config = getattr(raw_model, "runtime_config", None)
    if current_config is not None:
        expected = _checkpoint_contract(current_config)
        actual = checkpoint.get("checkpoint_contract", _checkpoint_contract(checkpoint["config"]))
        for key in expected:
            if str(actual.get(key)) != str(expected.get(key)):
                raise ValueError(f"Checkpoint contract mismatch for {key}")
    state = {(name[7:] if name.startswith("module.") else name): value for name, value in checkpoint["model_state_dict"].items()}
    if checkpoint.get("model_state_dict_format", "full") == "trainable_only":
        incompatible = raw_model.load_state_dict(state, strict=False)
        trainable_names = {name for name, parameter in raw_model.named_parameters() if parameter.requires_grad}
        missing_trainable = sorted(set(incompatible.missing_keys) & trainable_names)
        if incompatible.unexpected_keys or missing_trainable:
            raise RuntimeError(
                f"Compact checkpoint mismatch: missing_trainable={missing_trainable[:10]}, "
                f"unexpected={incompatible.unexpected_keys[:10]}"
            )
    else:
        raw_model.load_state_dict(state, strict=strict)
    return checkpoint


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
