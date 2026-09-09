#!/usr/bin/env python3
"""One real-data, real-Qwen forward/backward audit for CUTE-FND v3."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time

import torch
from torch.optim import AdamW

from mmfnd.data import move_batch
from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import autocast_context, load_checkpoint, save_checkpoint
from mmfnd.factory import build_loader, build_processor
from mmfnd.latent_evidence_deliberation import EVIDENCE_ORDER, LATENT_SEQUENCE_LENGTH
from mmfnd.model import ExplainableMMFND, multimodal_loss
from mmfnd.utils import get_device, load_config, seed_everything


def parameter_count(module: torch.nn.Module, trainable_only: bool = False) -> int:
    return sum(
        parameter.numel() for parameter in module.parameters()
        if not trainable_only or parameter.requires_grad
    )


def gradient_status(model: ExplainableMMFND) -> dict[str, bool]:
    groups = {
        "qwen_lora": lambda name: "encoder.text_encoder" in name and "lora_" in name,
        "shared_projector": lambda name: "lgled.shared_projector" in name,
        "role_embedding": lambda name: "lgled.role_embedding" in name,
        "judge_tokens": lambda name: "lgled.judge_tokens" in name,
        "evidential_head": lambda name: "lgled.relation_head" in name,
        "confidence_head": lambda name: "lgled.confidence_head" in name,
        "minority_head": lambda name: "lgled.minority_head" in name,
        "global_judge_head": lambda name: "lgled.global_judge_head" in name,
        "adjudication_head": lambda name: "lgled.adjudication_head" in name,
        "direct_fusion_head": lambda name: "lgled.direct_fusion_head" in name,
    }
    named = list(model.named_parameters())
    return {
        group: any(
            predicate(name) and parameter.requires_grad
            and parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all().item())
            for name, parameter in named
        )
        for group, predicate in groups.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/datasets/weibo21.json")
    parser.add_argument("--manifest-dir")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    config = load_config(root / args.config)
    dataset = str(config["dataset"]["name"])
    manifest_dir = args.manifest_dir or f"datasets/{dataset}/ready"
    bind_dataset_workspace(root, config, dataset, manifest_dir)
    positive_label, _ = validate_dataset_semantics(config)
    seed_everything(int(config["seed"]))
    device = get_device()
    precision = str(config["train"].get("precision", "bf16"))

    processor = build_processor(root, config)
    loader = build_loader(root, config, "train", processor, shuffle=True)
    batch = move_batch(next(iter(loader)), device)
    forbidden = sorted(key for key in batch if "evidence" in key.lower())
    if forbidden:
        raise RuntimeError(f"External-evidence fields remain in batch: {forbidden}")

    model = ExplainableMMFND(config).to(device)
    model.train()
    qwen = model.encoder.shared_qwen_model()
    shared_layers = model.lgled.selected_qwen_layers(qwen)
    expected_layers = list(qwen.layers)[-model.lgled.latent_judge_num_layers:]
    shared_layer_reference = all(
        actual is expected for actual, expected in zip(shared_layers, expected_layers)
    )
    lora_shared_in_judge_layers = all(
        any("lora_" in name for name, _ in layer.named_parameters())
        for layer in shared_layers
    )
    duplicate_qwen_namespace = any(
        key.startswith("reasoner.lgled.qwen") for key in model.state_dict()
    )

    stage_times: dict[str, float] = {}
    stage_starts: dict[str, float] = {}
    hooks = []
    for name, module in (("text_qwen_encoding", model.encoder.text_encoder),
                         ("latent_judge", model.lgled)):
        hooks.append(module.register_forward_pre_hook(
            lambda _module, _inputs, stage=name: stage_starts.__setitem__(stage, time.perf_counter())
        ))
        hooks.append(module.register_forward_hook(
            lambda _module, _inputs, _output, stage=name: stage_times.__setitem__(
                stage, time.perf_counter() - stage_starts[stage]
            )
        ))

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=float(config["train"]["head_learning_rate"]),
                      weight_decay=float(config["train"]["weight_decay"]))
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        memory_before = torch.cuda.memory_allocated(device)
        torch.cuda.synchronize(device)
    else:
        memory_before = 0
    forward_start = time.perf_counter()
    with autocast_context(device, precision):
        outputs = model(batch)
        loss, components = multimodal_loss(
            outputs, batch["labels"], config["loss"],
            float(config["train"]["label_smoothing"]), positive_label,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    whole_forward_seconds = time.perf_counter() - forward_start
    memory_after_forward = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    if not torch.isfinite(loss):
        raise FloatingPointError(f"Non-finite loss: {loss}; components={components}")
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    memory_after_backward = torch.cuda.memory_allocated(device) if device.type == "cuda" else 0
    peak_memory = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    for hook in hooks:
        hook.remove()

    invalid_gradients = [
        name for name, parameter in model.named_parameters()
        if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
    ]
    if invalid_gradients:
        raise FloatingPointError(f"Non-finite gradients: {invalid_gradients[:10]}")
    gradients = gradient_status(model)
    frozen_qwen_base_has_grad = any(
        parameter.grad is not None
        for name, parameter in model.encoder.text_encoder.named_parameters()
        if "lora_" not in name and not parameter.requires_grad
    )
    gradient_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    first_name, first_parameter = next(
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    expected = first_parameter.detach().clone()
    with tempfile.TemporaryDirectory(prefix="cute_fnd_v3_smoke_") as directory:
        checkpoint_path = Path(directory) / "roundtrip.pth"
        save_checkpoint(checkpoint_path, model, optimizer, None, 0,
                        {"smoke": True}, config)
        with torch.no_grad():
            first_parameter.zero_()
        restored_checkpoint = load_checkpoint(checkpoint_path, model, device)
        optimizer.load_state_dict(restored_checkpoint["optimizer_state_dict"])
        checkpoint_roundtrip = bool(torch.equal(
            expected, dict(model.named_parameters())[first_name].detach()
        ))

    checked_outputs = (
        "projected_evidence_tokens", "latent_output", "relation_alpha",
        "relation_strength", "relation_uncertainty", "evidence_deviation",
        "minority_score", "routing_gate", "fused_feature", "logits",
    )
    finite_outputs = {
        name: bool(torch.isfinite(outputs[name].float()).all().item())
        for name in checked_outputs
    }
    lgled = model.lgled
    qwen_total = parameter_count(qwen)
    qwen_lora = sum(
        parameter.numel() for name, parameter in model.encoder.text_encoder.named_parameters()
        if "lora_" in name and parameter.requires_grad
    )
    component_parameters = {
        "shared_projector": parameter_count(lgled.shared_projector),
        "role_embedding": parameter_count(lgled.role_embedding),
        "judge_tokens": parameter_count(lgled.judge_tokens),
        "evidential_head": parameter_count(lgled.relation_head),
        "confidence_head": parameter_count(lgled.confidence_head),
        "minority_head": parameter_count(lgled.minority_head),
        "global_judge_head": parameter_count(lgled.global_judge_head),
        "adjudication_head": parameter_count(lgled.adjudication_head),
        "direct_fusion_head": parameter_count(lgled.direct_fusion_head),
    }
    shared_projector_parameters = component_parameters["shared_projector"]
    independent_projector_parameters = shared_projector_parameters * len(EVIDENCE_ORDER)
    runtime = model.lgled.runtime(qwen)

    report = {
        "device": str(device), "precision": precision,
        "qwen_class": type(qwen).__name__,
        "qwen_hidden_size": int(qwen.config.hidden_size),
        "qwen_num_hidden_layers": int(qwen.config.num_hidden_layers),
        "qwen_total_parameters": qwen_total,
        "qwen_frozen_parameters": qwen_total - qwen_lora,
        "qwen_lora_trainable_parameters": qwen_lora,
        "shared_layer_reference": shared_layer_reference,
        "lora_shared_in_judge_layers": lora_shared_in_judge_layers,
        "shared_qwen_layer_indices": [runtime.first_shared_layer, runtime.last_shared_layer],
        "duplicate_qwen_state_namespace": duplicate_qwen_namespace,
        "batch_size": int(batch["labels"].size(0)),
        "evidence_order": list(EVIDENCE_ORDER),
        "evidence_shape": list(outputs["evidence_features"].shape),
        "projected_evidence_shape": list(outputs["projected_evidence_tokens"].shape),
        "judge_tokens_shape": list(outputs["judge_token_embeddings"].shape),
        "latent_sequence_shape": list(outputs["latent_sequence"].shape),
        "latent_output_shape": list(outputs["latent_output"].shape),
        "relation_probs_shape": list(outputs["relation_probs"].shape),
        "relation_uncertainty_shape": list(outputs["relation_uncertainty"].shape),
        "evidence_confidence_shape": list(outputs["evidence_confidence"].shape),
        "evidence_deviation_shape": list(outputs["evidence_deviation"].shape),
        "minority_score_shape": list(outputs["minority_score"].shape),
        "global_judge_weights_shape": list(outputs["global_judge_weights"].shape),
        "deliberative_weights_shape": list(outputs["deliberative_weights"].shape),
        "direct_feature_shape": list(outputs["direct_feature"].shape),
        "deliberative_feature_shape": list(outputs["deliberative_feature"].shape),
        "fused_feature_shape": list(outputs["fused_feature"].shape),
        "logits_shape": list(outputs["logits"].shape),
        "latent_sequence_length_expected": LATENT_SEQUENCE_LENGTH,
        "routing": {name: float(getattr(outputs["routing_gate"], name)())
                    for name in ("mean", "min", "max")},
        "relation_means": outputs["relation_probs"].float().mean(dim=(0, 1)).detach().cpu().tolist(),
        "mean_relation_strength": float(outputs["relation_strength"].mean()),
        "mean_relation_uncertainty": float(outputs["relation_uncertainty"].mean()),
        "gradients": gradients,
        "frozen_qwen_base_has_grad": frozen_qwen_base_has_grad,
        "all_checked_outputs_finite": all(finite_outputs.values()),
        "finite_outputs": finite_outputs,
        "loss": float(loss.detach()), "loss_components": components,
        "gradient_norm_before_clip": float(gradient_norm),
        "optimizer_step": "passed",
        "checkpoint_roundtrip": "passed" if checkpoint_roundtrip else "failed",
        "parameters": {
            **component_parameters,
            "four_independent_projectors_hypothetical": independent_projector_parameters,
            "shared_projector_savings": independent_projector_parameters - shared_projector_parameters,
            "model_total": parameter_count(model),
            "model_trainable": parameter_count(model, trainable_only=True),
        },
        "memory_bytes": {
            "before_forward": memory_before, "after_forward": memory_after_forward,
            "after_backward": memory_after_backward, "peak_allocated": peak_memory,
        },
        "runtime_seconds": {**stage_times, "whole_model_forward": whole_forward_seconds},
        "forbidden_input_fields": forbidden,
    }
    if not all(gradients.values()):
        raise RuntimeError(f"Missing LG-LED/LoRA gradients: {gradients}")
    if frozen_qwen_base_has_grad or duplicate_qwen_namespace:
        raise RuntimeError("Qwen freeze/state_dict sharing audit failed")
    if not shared_layer_reference or not lora_shared_in_judge_layers:
        raise RuntimeError("Qwen layer/LoRA sharing audit failed")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
