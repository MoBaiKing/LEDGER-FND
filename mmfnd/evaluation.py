"""Validation Macro-F1 protocol; preserve v3's original Fake=0 / Real=1.

Select P(Fake) using dataset.positive_label; never relabel records or logits.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.metrics import (
    accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_recall_fscore_support, roc_auc_score,
)

PROTOCOL = "validation_macro_f1_threshold_v1"
LABEL_SEMANTICS = {"0": "fake", "1": "real"}


def validate_binary_inputs(y_true, positive_probs):
    labels = np.asarray(y_true)
    probabilities = np.asarray(positive_probs, dtype=np.float64)
    if labels.ndim != 1 or probabilities.ndim != 1 or not labels.size:
        raise ValueError("Expected nonempty one-dimensional labels and Fake probabilities")
    if labels.shape != probabilities.shape or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary and match probability count")
    if not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Fake probabilities must be finite and in [0, 1]")
    return labels.astype(np.int64), probabilities


def validate_class_names(positive_label, class_names):
    names = {int(k): str(v).strip().lower() for k, v in class_names.items()}
    if positive_label not in (0, 1) or names != {positive_label: "fake", 1 - positive_label: "real"}:
        raise ValueError("Evaluation requires the configured positive class to be Fake")
    return names


def threshold_grid(train_config):
    low = float(train_config.get("threshold_min", 0.01))
    high = float(train_config.get("threshold_max", 0.99))
    step = float(train_config.get("threshold_step", 0.01))
    if not all(math.isfinite(x) for x in (low, high, step)) or not (0 <= low <= high <= 1) or step <= 0:
        raise ValueError("Invalid threshold grid")
    # Integer step count avoids np.arange accidentally extending beyond high.
    count = int(math.floor((high - low) / step + 1e-10)) + 1
    return np.round(low + np.arange(count, dtype=np.float64) * step, 12)


def find_best_macro_f1_threshold(y_true, positive_probs, thresholds=None, *, split, positive_label=0):
    """Exact Macro-F1 maximum; ties: nearest 0.5, then lower threshold.

    `split` is mandatory: the shared API refuses tuning on train/test. Every
    seed/checkpoint must supply its own validation observations.
    """
    if split not in {"val", "validation"}:
        raise ValueError("Threshold tuning is permitted only on validation, never test")
    if positive_label not in (0, 1):
        raise ValueError("positive_label must be 0 or 1")
    labels, probabilities = validate_binary_inputs(y_true, positive_probs)
    grid = np.asarray(thresholds if thresholds is not None else np.arange(1, 100) / 100, dtype=np.float64)
    if grid.ndim != 1 or not grid.size or not np.isfinite(grid).all() or ((grid < 0) | (grid > 1)).any():
        raise ValueError("Thresholds must be a nonempty finite one-dimensional grid in [0, 1]")
    scores = np.array([
        f1_score(labels, np.where(probabilities >= threshold, positive_label, 1 - positive_label),
                 labels=[0, 1], average="macro", zero_division=0)
        for threshold in grid
    ])
    maximum = float(scores.max())
    # No near-optimal plateau: only exact score ties are eligible.
    candidates = grid[scores == maximum]
    selected = min(candidates.tolist(), key=lambda x: (round(abs(x - 0.5), 12), x))
    return {"threshold": float(selected), "macro_f1": maximum,
            "threshold_grid": sorted(set(float(x) for x in grid))}


def compute_classification_metrics(y_true, probs, threshold, *, positive_label=0, class_names=None, ece_bins=15):
    """All discrete metrics share y_pred; probability metrics never use tau."""
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != 2:
        raise ValueError("Expected N x 2 probabilities in the original model label order")
    names = validate_class_names(positive_label, class_names or LABEL_SEMANTICS)
    labels, fake_prob = validate_binary_inputs(y_true, probs[:, positive_label])
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any() or not np.allclose(probs.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probabilities must be finite, in [0, 1], and sum to one")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and in [0, 1]")
    if not isinstance(ece_bins, int) or ece_bins < 1:
        raise ValueError("ece_bins must be a positive integer")
    predictions = np.where(fake_prob >= threshold, positive_label, 1 - positive_label).astype(np.int64)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=[0, 1], zero_division=0,
    )
    binary = (labels == positive_label).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(binary, predictions == positive_label, labels=[0, 1]).ravel()
    if np.unique(labels).size < 2:
        warnings.warn("AUC is undefined for a single-class split; returning NaN", RuntimeWarning, stacklevel=2)
        auc = float("nan")
    else:
        auc = float(roc_auc_score(binary, fake_prob))
    # Standard top-label ECE. Argmax is used ONLY here, never for reported Accuracy.
    confidence = probs.max(axis=1)
    calibration_correct = (probs.argmax(axis=1) == labels).astype(np.float64)
    bins = np.minimum((confidence * ece_bins).astype(np.int64), ece_bins - 1)
    ece = 0.0
    for index in range(ece_bins):
        mask = bins == index
        if mask.any():
            ece += float(mask.mean() * abs(calibration_correct[mask].mean() - confidence[mask].mean()))
    accuracy = float(accuracy_score(labels, predictions))
    # Use the original true-class probability. Float32 softmax rows can differ
    # from 1 by ~1e-7: don't rescale probabilities just to satisfy log_loss's
    # float64 sum check. Clip only for log(0), independently of the threshold.
    true_prob = probs[np.arange(labels.size), labels]
    nll = float(-np.log(np.clip(true_prob, np.finfo(np.float64).eps, 1.0)).mean())
    metrics = {
        "samples": int(labels.size), "threshold": float(threshold),
        "decision_threshold": float(threshold), "accuracy": accuracy,
        "threshold_accuracy": accuracy,  # compatibility alias; exactly the same y_pred
        "macro_f1": float(f1_score(labels, predictions, labels=[0, 1], average="macro", zero_division=0)),
        "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "auc": auc, "nll": nll,
        "brier": float(brier_score_loss(binary, fake_prob)), "ece": float(ece),
        "ece_definition": "top_label_equal_width_15_bins" if ece_bins == 15 else f"top_label_equal_width_{ece_bins}_bins",
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
        "confusion_matrix_labels": [0, 1],
        "label_semantics": {str(k): v for k, v in names.items()},
        "positive_label": positive_label, "positive_class": "fake",
    }
    for label, name in names.items():
        metrics.update({f"{name}_precision": float(precision[label]), f"{name}_recall": float(recall[label]),
                        f"{name}_f1": float(f1[label]), f"{name}_support": int(support[label])})
    metrics.update(positive_precision=metrics["fake_precision"], positive_recall=metrics["fake_recall"],
                   positive_f1=metrics["fake_f1"])
    return metrics


def validate_threshold_selection(selection):
    """Reject legacy/fixed/default thresholds lacking validation provenance."""
    if not isinstance(selection, dict) or selection.get("evaluation_protocol") != PROTOCOL:
        raise ValueError("Checkpoint has no compatible validation Macro-F1 threshold; recalibrate on validation first")
    if (selection.get("threshold_source") != "validation" or selection.get("threshold_objective") != "macro_f1"
            or selection.get("positive_label") not in (0, 1) or selection.get("positive_class") != "fake"):
        raise ValueError("Invalid threshold provenance or class semantics")
    validate_class_names(selection["positive_label"], selection.get("label_semantics", {}))
    threshold = selection.get("threshold")
    if not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid saved validation threshold")
    score = selection.get("val_best_macro_f1")
    if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
        raise ValueError("Missing validation Macro-F1 for threshold selection")
    reference = selection.get("checkpoint_reference")
    if not isinstance(reference, dict) or reference.get("kind") not in {"epoch", "top_k_average", "recalibrated_checkpoint"}:
        raise ValueError("Threshold must identify the checkpoint evaluated on validation")
    if reference["kind"] == "epoch" and not isinstance(reference.get("epoch"), int):
        raise ValueError("Threshold must identify its validation epoch")
    if reference["kind"] == "top_k_average" and not reference.get("epochs"):
        raise ValueError("Threshold must identify the averaged epochs")
    if reference["kind"] == "recalibrated_checkpoint" and not reference.get("source_checkpoint"):
        raise ValueError("Threshold must identify the recalibrated checkpoint")
    if selection.get("decision_threshold") != threshold:
        raise ValueError("Saved threshold aliases disagree")
    return float(threshold)


def checkpoint_threshold_selection(checkpoint):
    selection = checkpoint.get("threshold_selection")
    threshold = validate_threshold_selection(selection)
    if checkpoint.get("decision_threshold") != threshold or checkpoint.get("evaluation_protocol") != PROTOCOL:
        raise ValueError("Checkpoint/threshold protocol mismatch")
    if checkpoint.get("config", {}).get("dataset", {}).get("positive_label") != selection["positive_label"]:
        raise ValueError("Checkpoint and threshold use different positive-label indices")
    metrics = checkpoint.get("metrics", {})
    if (metrics.get("split") != "val" or metrics.get("threshold_selection") != selection
            or metrics.get("macro_f1") != selection["val_best_macro_f1"]
            or metrics.get("threshold") != threshold or metrics.get("decision_threshold") != threshold):
        raise ValueError("Checkpoint metrics are not paired with its validation threshold")
    reference = selection["checkpoint_reference"]
    if reference.get("kind") == "epoch" and reference.get("epoch") != checkpoint.get("epoch"):
        raise ValueError("Checkpoint and threshold come from different epochs")
    if reference.get("kind") == "top_k_average":
        epochs = [item["epoch"] for item in checkpoint.get("training_state", {}).get("averaged_checkpoints", [])]
        if epochs != reference.get("epochs"):
            raise ValueError("Averaged checkpoint and threshold provenance disagree")
    return selection


def evaluation_info(metrics):
    return {key: metrics[key] for key in (
        "evaluation_protocol", "threshold", "decision_threshold", "threshold_source", "threshold_objective",
        "positive_label", "positive_class", "label_semantics", "split",
        "val_best_macro_f1", "threshold_selection",
    )}


def log_evaluation(metrics):
    fields = ["macro_f1", "threshold", "accuracy", "fake_precision", "fake_recall", "fake_f1",
              "real_precision", "real_recall", "real_f1", "auc", "nll", "brier", "ece"]
    print(f"[{metrics['split'].upper()}] Threshold Source: {metrics['threshold_source']}; "
          f"Objective: {metrics['threshold_objective']}; " + "; ".join(f"{key}={metrics[key]:.6f}" for key in fields), flush=True)
