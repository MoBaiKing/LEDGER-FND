#!/usr/bin/env python3
"""Fast B=1 LG-LED API/gradient test with a tiny in-memory Qwen2 model."""
from __future__ import annotations

import json

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from transformers import Qwen2Config, Qwen2Model

from mmfnd.latent_evidence_deliberation import (
    LATENT_SEQUENCE_LENGTH,
    LLMGuidedLatentEvidenceDeliberation,
)


def main() -> None:
    torch.manual_seed(7)
    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        use_cache=False,
    )
    peft_qwen = get_peft_model(
        Qwen2Model(config),
        LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=8, lora_alpha=16, lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"], bias="none",
        ),
    )
    qwen = peft_qwen.get_base_model()
    lgled = LLMGuidedLatentEvidenceDeliberation(
        evidence_dim=24,
        qwen_hidden_size=qwen.config.hidden_size,
        cfg={
            "projector_type": "shared", "judge_type": "qwen_latent",
            "latent_judge_num_layers": 2, "head_hidden_dim": 16,
            "dropout": 0.0,
        },
    )
    classifier = nn.Linear(24, 2)
    evidence = [torch.randn(1, 24) for _ in range(4)]
    outputs = lgled(evidence, qwen)
    logits = classifier(outputs["fused_feature"])
    loss = nn.functional.cross_entropy(logits, torch.tensor([1]))
    loss.backward()

    required_shapes = {
        "projected_evidence_tokens": [1, 4, 64],
        "judge_token_embeddings": [1, 7, 64],
        "latent_sequence": [1, LATENT_SEQUENCE_LENGTH, 64],
        "latent_output": [1, LATENT_SEQUENCE_LENGTH, 64],
        "relation_probs": [1, 6, 3],
        "relation_uncertainty": [1, 6],
        "evidence_confidence": [1, 4],
        "evidence_deviation": [1, 4],
        "minority_score": [1, 4],
        "global_judge_weights": [1, 4],
        "deliberative_weights": [1, 4],
        "direct_feature": [1, 24],
        "deliberative_feature": [1, 24],
        "fused_feature": [1, 24],
    }
    actual_shapes = {name: list(outputs[name].shape) for name in required_shapes}
    if actual_shapes != required_shapes:
        raise AssertionError({"expected": required_shapes, "actual": actual_shapes})
    finite = {
        name: bool(torch.isfinite(value.float()).all().item())
        for name, value in outputs.items() if isinstance(value, torch.Tensor)
    }
    selected = lgled.selected_qwen_layers(qwen)
    shared_reference = all(
        actual is expected
        for actual, expected in zip(selected, list(qwen.layers)[-2:])
    )
    gradients = {
        "shared_projector": lgled.shared_projector.linear.weight.grad is not None,
        "role_embedding": lgled.role_embedding.embedding.weight.grad is not None,
        "judge_tokens": lgled.judge_tokens.embedding.weight.grad is not None,
        "evidential_head": lgled.relation_head.output.weight.grad is not None,
        "confidence_head": next(lgled.confidence_head.parameters()).grad is not None,
        "minority_head": next(lgled.minority_head.parameters()).grad is not None,
        "global_judge_head": lgled.global_judge_head.output.weight.grad is not None,
        "adjudication_head": next(lgled.adjudication_head.parameters()).grad is not None,
        "direct_fusion_head": lgled.direct_fusion_head.output.weight.grad is not None,
        "qwen_lora": any(
            "lora_" in name and parameter.grad is not None
            for name, parameter in peft_qwen.named_parameters()
        ),
        "frozen_qwen_base": all(
            parameter.grad is None
            for name, parameter in peft_qwen.named_parameters()
            if "lora_" not in name
        ),
    }
    if not all(finite.values()) or not all(gradients.values()) or not shared_reference:
        raise AssertionError({
            "finite": finite, "gradients": gradients,
            "shared_layer_reference": shared_reference,
        })
    state_has_duplicate_qwen = any(
        key.startswith("qwen") for key in lgled.state_dict()
    )
    report = {
        "status": "passed", "batch_size": 1,
        "shapes": actual_shapes, "finite": finite,
        "gradients": gradients,
        "shared_layer_reference": shared_reference,
        "lgled_state_contains_qwen": state_has_duplicate_qwen,
        "loss": float(loss.detach()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
