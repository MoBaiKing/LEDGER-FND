"""Isolated v2 ablations. Full delegates to the unchanged production forward."""
from copy import deepcopy
import math
from types import SimpleNamespace

import torch
from torch import nn

from ablation.registry import settings, states, validate, config_hash
from mmfnd.latent_evidence_deliberation import (
    LLMGuidedLatentEvidenceDeliberation, PAIR_EVIDENCE_INDICES, NUM_EVIDENCE, EVIDENCE_ORDER, RELATION_ORDER,
    strict_latent_attention_mask,
)
from mmfnd.model import ExplainableMMFND, UncertaintyCalibratedDecision


class SmallCausalEvaluator(nn.Module):
    """Random Transformer with strict pair slot visibility, H -> h -> H."""
    def __init__(self, qwen_hidden, cfg):
        super().__init__()
        dim = cfg["hidden_dim"]
        self.input_adapter = nn.Linear(qwen_hidden, dim)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, cfg["num_heads"], cfg["feedforward_dim"],
                                       cfg["dropout"], activation="gelu", batch_first=True, norm_first=True)
            for _ in range(cfg["num_layers"])
        ])
        self.norm = nn.LayerNorm(dim)
        self.output_adapter = nn.Linear(dim, qwen_hidden)

    @staticmethod
    def causal_mask(length, device):
        return strict_latent_attention_mask(length, device)

    def forward(self, tokens):
        hidden = self.input_adapter(tokens)
        length, dim = hidden.shape[1:]
        positions = torch.arange(length, device=hidden.device).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, dim, 2, device=hidden.device).float() * (-math.log(10000) / dim))
        positional = hidden.new_zeros(length, dim)
        positional[:, 0::2] = torch.sin(positions * frequency)
        positional[:, 1::2] = torch.cos(positions * frequency)
        hidden = hidden + positional
        mask = self.causal_mask(length, hidden.device)
        for layer in self.layers:
            hidden = layer(hidden, src_mask=mask)
        return self.output_adapter(self.norm(hidden))


class BypassDecision(UncertaintyCalibratedDecision):
    def forward(self, encoded, reasoned):
        joint = self.final_norm(reasoned["fused_feature"])
        logits = self.classifier(joint)
        batch = joint.shape[0]
        zero = joint.new_zeros(batch)
        return {"logits": logits, "preliminary_logits": logits, "calibrated": {
            "feature": joint, "uncertainty": zero, "components": joint.new_zeros(batch, 4),
            "component_weights": joint.new_zeros(4), "temperature": torch.ones_like(zero),
            "correction_norm": zero, "correction_scale": joint.new_zeros(()),
        }}


class MLPClassificationModule(nn.Module):
    """Replace the complete post-encoder classification module with one MLP."""

    def __init__(self, dim, dropout, view_dropout_probability):
        super().__init__()
        self.view_dropout_probability = float(view_dropout_probability)
        self.input_norm = nn.LayerNorm(4 * dim)
        self.hidden = nn.Linear(4 * dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(dim, 2)

    def forward(self, encoded, ablate_component=None):
        evidence = torch.stack([
            encoded["text"], encoded["vision"],
            encoded["intrinsic_feature"], encoded["interaction"],
        ], dim=1)
        available = torch.ones(
            evidence.shape[:2], dtype=torch.bool, device=evidence.device
        )
        if ablate_component == "text":
            available[:, (0, 2, 3)] = False
        elif ablate_component == "image":
            available[:, (1, 2, 3)] = False
        elif ablate_component == "intrinsic_event":
            available[:, 2] = False
        if (
            self.training
            and ablate_component is None
            and self.view_dropout_probability > 0.0
        ):
            drop_sample = torch.rand(evidence.size(0), device=evidence.device) < self.view_dropout_probability
            drop_index = torch.randint(3, (evidence.size(0),), device=evidence.device)
            rows = torch.arange(evidence.size(0), device=evidence.device)[drop_sample]
            available[rows, drop_index[drop_sample]] = False
            available[:, 3] = available[:, 0] & available[:, 1]
        masked = evidence * available.unsqueeze(-1).to(evidence.dtype)
        hidden = self.activation(self.hidden(self.input_norm(masked.flatten(1))))
        return self.output(self.dropout(hidden)), hidden, evidence, available


class NoLGLEDCompatibility(nn.Module):
    """Parameter-free interface required by the unchanged training logger."""

    latent_judge_num_layers = 0

    def selected_qwen_layers(self, qwen_model):
        return []

    def runtime(self, qwen_model):
        return SimpleNamespace(first_shared_layer=None, last_shared_layer=None)


class AblationDeliberation(LLMGuidedLatentEvidenceDeliberation):
    def configure(self, config):
        self.variant = config["ablation"]["name"]
        self.state = states(self.variant)
        self.fixed_mix = config["ablation"]["fixed_mix_coefficient"]
        if self.variant == "full":
            return
        if self.variant == "mlp_classifier":
            raise RuntimeError("mlp_classifier replaces the whole classifier and must bypass LG-LED")
        if not self.state["raw_mean_residual"]:
            del self.fusion_residual_scale
        if self.state["evaluator"] == "none":
            for name in list(self._modules):
                if name not in {"fusion_norm", "direct_fusion_head"}:
                    delattr(self, name)
            return
        if self.variant == "non_llm_evaluator":
            with torch.random.fork_rng(devices=[]):
                self.evaluator = SmallCausalEvaluator(self.qwen_hidden_size, config["ablation"]["non_llm_evaluator"])
        if self.variant == "no_global_token":
            self.judge_tokens.embedding = nn.Embedding.from_pretrained(
                self.judge_tokens.embedding.weight[:6].detach().clone(), freeze=False)
            self.judge_tokens.__class__ = PairTokenBank
        if self.variant == "no_critical_minority":
            self.minority_head = None
        if not self.state["direct_candidate"]:
            self.direct_fusion_head = None

    def dynamic_gate(self, disagreement):
        return torch.sigmoid(self.routing_scale * (disagreement - self.routing_threshold))

    def _without_deliberation(self, evidence, available):
        batch = evidence.shape[0]
        zero4, zero6 = evidence.new_zeros(batch, 4), evidence.new_zeros(batch, 6)
        zero1 = evidence.new_zeros(batch, 1)
        scores = self.direct_fusion_head(evidence, zero4).float()
        weights = self._masked_softmax(scores, available)
        feature = (weights.unsqueeze(-1) * evidence.float()).sum(1)
        # Undefined diagnostic fields are neutral internal interface placeholders.
        # The export adapter writes null + a reason, never reports them as beliefs.
        return {
            "relation_evidence": evidence.new_zeros(batch, 6, 3),
            "relation_alpha": evidence.new_ones(batch, 6, 3),
            "relation_probs": evidence.new_full((batch, 6, 3), 1 / 3),
            "relation_strength": zero6 + 3, "relation_uncertainty": zero6,
            "evidence_confidence": zero4, "evidence_deviation": zero4,
            "evidence_mean_uncertainty": zero4, "evidence_mean_conflict": zero4,
            "minority_score": zero4, "global_judge_logits": zero4,
            "global_judge_weights": zero4, "deliberative_weights": zero4,
            "direct_weights": weights, "final_evidence_weights": weights,
            "sample_disagreement": zero1, "routing_gate": zero1,
            "direct_feature": feature, "deliberative_feature": evidence.new_zeros(batch, self.evidence_dim),
            "fused_feature": self.fusion_norm(feature), "evidence_available": available,
            "pair_available": torch.stack([available[:, i] & available[:, j] for i, j in PAIR_EVIDENCE_INDICES], 1),
        }

    def forward(self, evidence_features: list[torch.Tensor],
                qwen_model: nn.Module,
                available: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if self.variant == "full":
            return super().forward(evidence_features, qwen_model, available)
        if len(evidence_features) != NUM_EVIDENCE:
            raise ValueError(
                f"LG-LED expects {NUM_EVIDENCE} evidence tensors in "
                f"{EVIDENCE_ORDER}, got {len(evidence_features)}"
            )
        evidence = torch.stack(evidence_features, dim=1)
        if evidence.size(-1) != self.evidence_dim:
            raise ValueError(
                f"evidence dim is {evidence.size(-1)}, expected {self.evidence_dim}"
            )
        if available is None:
            available = torch.ones(
                evidence.shape[:2], dtype=torch.bool, device=evidence.device
            )
        if available.shape != evidence.shape[:2]:
            raise ValueError(
                f"availability shape {tuple(available.shape)} does not match "
                f"evidence shape {tuple(evidence.shape[:2])}"
            )
        available = available.bool()
        if not available.any(dim=1).all():
            raise ValueError("Every sample needs at least one available evidence")
        evidence = torch.where(available.unsqueeze(-1), evidence, torch.zeros_like(evidence))
        if self.state["evaluator"] == "none":
            return self._without_deliberation(evidence, available)
        pair_available = torch.stack([
            available[:, left] & available[:, right]
            for left, right in PAIR_EVIDENCE_INDICES
        ], dim=1)

        projected = self._project(evidence)
        projected = projected * available.unsqueeze(-1).to(projected.dtype)
        judges = self.judge_tokens(evidence.size(0)).to(projected.dtype)
        latent_sequence = torch.cat([projected, judges], dim=1)
        expected_length = 10 if self.variant == "no_global_token" else 11
        if latent_sequence.size(1) != expected_length:
            raise RuntimeError("Incorrect evidence/pair/global slots")
        if self.variant == "non_llm_evaluator":
            latent_output = self.evaluator(latent_sequence)
        else:
            latent_output = self._run_qwen_last_layers(latent_sequence, qwen_model)
        evidence_states = latent_output[:, :4]
        pair_states = latent_output[:, 4:10]
        if self.variant == "no_global_token":
            valid = pair_available.unsqueeze(-1)
            global_state = (pair_states * valid).sum(1) / valid.sum(1).clamp_min(1)
        else:
            global_state = latent_output[:, 10]

        relation = self.relation_head(pair_states)
        if self.variant == "no_uncertainty":
            relation["raw_relation_uncertainty"] = relation["relation_uncertainty"].detach()
            relation["relation_uncertainty"] = torch.zeros_like(relation["relation_uncertainty"])
        if not self.use_evidential_relation:
            relation["relation_probs"] = relation[
                "relation_evidence"
            ].softmax(dim=-1)
            relation["relation_uncertainty"] = torch.zeros_like(
                relation["relation_uncertainty"]
            )
        pair_available = torch.stack([
            available[:, left] & available[:, right]
            for left, right in PAIR_EVIDENCE_INDICES
        ], dim=1)
        pair_available_float = pair_available.to(relation["relation_probs"].dtype)
        conflict = relation["relation_probs"][:, :, RELATION_ORDER.index("conflict")]
        effective_uncertainty = (
            relation["relation_uncertainty"]
            if self.use_uncertainty else torch.zeros_like(
                relation["relation_uncertainty"]
            )
        )
        certified_conflict = conflict * (1.0 - effective_uncertainty)
        certified_conflict = certified_conflict * pair_available_float
        membership = self.pair_membership.to(certified_conflict.dtype)
        valid_membership = (
            pair_available_float.unsqueeze(1) * membership.unsqueeze(0)
        )
        evidence_pair_count = valid_membership.sum(dim=-1).clamp_min(1.0)
        deviation = (
            certified_conflict.unsqueeze(1) * membership.unsqueeze(0)
        ).sum(dim=-1) / evidence_pair_count
        mean_uncertainty = (
            relation["relation_uncertainty"].unsqueeze(1)
            * valid_membership
        ).sum(dim=-1) / evidence_pair_count
        mean_conflict = (
            conflict.unsqueeze(1) * valid_membership
        ).sum(dim=-1) / evidence_pair_count

        confidence = self.confidence_head(evidence_states).float()
        confidence = confidence * available.to(confidence.dtype)
        minority_features = torch.cat([
            evidence_states,
            confidence.to(evidence_states.dtype).unsqueeze(-1),
            deviation.to(evidence_states.dtype).unsqueeze(-1),
            mean_uncertainty.to(evidence_states.dtype).unsqueeze(-1),
            mean_conflict.to(evidence_states.dtype).unsqueeze(-1),
        ], dim=-1)
        minority = (self.minority_head(minority_features).float()
                    if self.minority_head is not None else torch.zeros_like(confidence))
        minority = minority * available.to(minority.dtype)

        global_logits = self.global_judge_head(global_state).float()
        if not self.use_global_judge:
            global_logits = torch.zeros_like(global_logits)
        global_weights = self._masked_softmax(global_logits, available)
        adjudication_features = torch.cat([
            evidence_states,
            confidence.to(evidence_states.dtype).unsqueeze(-1),
            deviation.to(evidence_states.dtype).unsqueeze(-1),
            minority.to(evidence_states.dtype).unsqueeze(-1),
            mean_uncertainty.to(evidence_states.dtype).unsqueeze(-1),
            global_logits.to(evidence_states.dtype).unsqueeze(-1),
        ], dim=-1)
        adjudication_scores = self.adjudication_head(
            adjudication_features
        ).float()
        deliberative_weights = self._masked_softmax(
            adjudication_scores, available
        )
        if self.state["direct_candidate"]:
            direct_confidence = (torch.zeros_like(confidence)
                                 if self.state["direct_confidence"] == "constant_zero" else confidence)
            direct_scores = self.direct_fusion_head(evidence, direct_confidence).float()
            direct_weights = self._masked_softmax(direct_scores, available)
        else:
            direct_weights = torch.zeros_like(confidence)

        evidence_float = evidence.float()
        deliberative_feature = (
            deliberative_weights.unsqueeze(-1) * evidence_float
        ).sum(dim=1)
        direct_feature = (
            direct_weights.unsqueeze(-1) * evidence_float
        ).sum(dim=1)
        valid_pair_count = pair_available_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        sample_disagreement = certified_conflict.sum(
            dim=1, keepdim=True
        ) / valid_pair_count
        mode = self.state["mixture"]
        if mode == "deliberation":
            routing_gate = torch.ones_like(sample_disagreement)
            fused = deliberative_feature
            final_evidence_weights = deliberative_weights
        else:
            routing_gate = (torch.full_like(sample_disagreement, self.fixed_mix)
                            if mode == "fixed" else self.dynamic_gate(sample_disagreement))
            fused = (1.0 - routing_gate) * direct_feature + routing_gate * deliberative_feature
            final_evidence_weights = (1.0 - routing_gate) * direct_weights + routing_gate * deliberative_weights
        evidence_mean = (
            evidence_float * available.unsqueeze(-1).to(evidence_float.dtype)
        ).sum(dim=1) / available.sum(dim=1, keepdim=True).clamp_min(1).to(
            evidence_float.dtype
        )
        if self.state["raw_mean_residual"]:
            fused = fused + self.fusion_residual_scale * evidence_mean
        fused = self.fusion_norm(fused)

        return {
            **relation,
            "evidence_confidence": confidence,
            "evidence_deviation": deviation,
            "evidence_mean_uncertainty": mean_uncertainty,
            "evidence_mean_conflict": mean_conflict,
            "minority_score": minority,
            "global_judge_logits": global_logits,
            "global_judge_weights": global_weights,
            "deliberative_weights": deliberative_weights,
            "direct_weights": direct_weights,
            "final_evidence_weights": final_evidence_weights,
            "sample_disagreement": sample_disagreement,
            "routing_gate": routing_gate,
            "projected_evidence_tokens": projected,
            "judge_token_embeddings": judges,
            "latent_sequence": latent_sequence,
            "latent_output": latent_output,
            "direct_feature": direct_feature,
            "deliberative_feature": deliberative_feature,
            "fused_feature": fused,
            "evidence_available": available,
            "pair_available": pair_available,
        }


class PairTokenBank(nn.Module):
    def forward(self, batch_size):
        indices = torch.arange(6, device=self.embedding.weight.device)
        return self.embedding(indices).unsqueeze(0).expand(batch_size, -1, -1)


class AblationMMFND(ExplainableMMFND):
    def __init__(self, config):
        if "ablation" not in config:
            config["ablation"] = settings()
        validate(config)
        super().__init__(config)
        variant = config["ablation"]["name"]
        if variant == "mlp_classifier":
            dim = int(config["model"]["hidden_dim"])
            with torch.random.fork_rng(devices=[]):
                replacement = MLPClassificationModule(
                    dim, config["model"]["dropout"],
                    config["model"].get("view_dropout_probability", 0.0),
                )
            self._lgled_compatibility = NoLGLEDCompatibility()
            self.reasoner = None
            self.decision = replacement
        else:
            self.lgled.__class__ = AblationDeliberation
            self.lgled.configure(config)
            if not states(variant)["decision_correction"]:
                self.decision.__class__ = BypassDecision
                self.decision.calibrator = None
        self.runtime_config["ablation_module_states"] = states(variant)
        count = lambda module: sum(p.numel() for p in module.parameters())
        lgled = None if self.reasoner is None else self.lgled
        self.runtime_config["ablation_parameter_counts"] = {
            "total": count(self), "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "lgled_total": count(lgled) if lgled is not None else 0,
            "lgled_trainable": sum(p.numel() for p in lgled.parameters() if p.requires_grad) if lgled is not None else 0,
            "replacement_transformer": count(lgled.evaluator.layers) + count(lgled.evaluator.norm)
                if lgled is not None and hasattr(lgled, "evaluator") else 0,
            "replacement_adapters": count(lgled.evaluator.input_adapter) + count(lgled.evaluator.output_adapter)
                if lgled is not None and hasattr(lgled, "evaluator") else 0,
            "replacement_mlp_classifier": count(self.decision) if variant == "mlp_classifier" else 0,
        }
        self.runtime_config["ablation_runtime_config_hash"] = config_hash({
            key: value for key, value in self.runtime_config.items() if key != "ablation_runtime_config_hash"
        })

    @property
    def lgled(self):
        if self.reasoner is None:
            return self._lgled_compatibility
        return self.reasoner.lgled

    def forward(self, batch, ablate_component=None):
        if self.runtime_config["ablation"]["name"] == "mlp_classifier":
            output = self._forward_mlp_classifier(batch, ablate_component)
        else:
            output = super().forward(batch, ablate_component)
        if self.training and "labels" in batch:
            from ablation.loss import compute_loss
            loss, components = compute_loss(output, batch["labels"], self.runtime_config)
            # DDP sees precisely the loss graph. Diagnostic tensors are detached,
            # so find_unused_parameters can correctly identify inactive branches.
            output = {key: value.detach() if isinstance(value, torch.Tensor) else value
                      for key, value in output.items()}
            output["training_loss"] = loss
            output["loss_components"] = components
        return output

    def _forward_mlp_classifier(self, batch, ablate_component=None):
        if ablate_component not in (None, "text", "image", "intrinsic_event"):
            raise ValueError("ablate_component must be one of: text, image, intrinsic_event")
        encoded = self.encoder(batch, ablate_component)
        logits, hidden, evidence, available = self.decision(encoded, ablate_component)
        batch_size = logits.size(0)
        zero = logits.new_zeros(batch_size)
        zero1 = logits.new_zeros(batch_size, 1)
        zero2 = logits.new_zeros(batch_size, 2)
        zero3 = logits.new_zeros(batch_size, 3)
        zero4 = logits.new_zeros(batch_size, 4)
        zero6 = logits.new_zeros(batch_size, 6)
        pair_available = torch.stack([
            available[:, left] & available[:, right]
            for left, right in PAIR_EVIDENCE_INDICES
        ], dim=1)
        intrinsic = encoded["intrinsic"]
        return {
            "logits": logits,
            "preliminary_logits": logits,
            "uncertainty": zero,
            "uncertainty_components": logits.new_zeros(batch_size, 4),
            "uncertainty_component_weights": logits.new_zeros(4),
            "uncertainty_temperature": torch.ones_like(zero),
            "uncertainty_correction_norm": zero,
            "uncertainty_correction_scale": logits.new_zeros(()),
            "latent_relation_uncertainty": zero,
            "modality_weights": zero3,
            "causal_gate_effects": zero3,
            "causal_probe_logits": zero2,
            "causal_probe_ablated_logits": logits.new_zeros(batch_size, 3, 2),
            "text_relation_ambiguity": intrinsic["text_relation_ambiguity"],
            "event_contradiction": intrinsic["event_contradiction"],
            "visual_evidence_inconsistency": intrinsic["visual_evidence_inconsistency"],
            "visual_source_inconsistency": intrinsic["visual_evidence_inconsistency"],
            "multi_image_dispersion": intrinsic["multi_image_dispersion"],
            "multi_image_available": intrinsic["multi_image_available"],
            "intrinsic_alignment_attention": intrinsic["alignment_attention"],
            "evidence_features": evidence,
            "relation_evidence": logits.new_zeros(batch_size, 6, 3),
            "relation_alpha": logits.new_ones(batch_size, 6, 3),
            "relation_probs": logits.new_full((batch_size, 6, 3), 1 / 3),
            "relation_strength": zero6 + 3,
            "relation_uncertainty": zero6,
            "evidence_confidence": zero4,
            "evidence_deviation": zero4,
            "evidence_mean_uncertainty": zero4,
            "evidence_mean_conflict": zero4,
            "minority_score": zero4,
            "global_judge_logits": zero4,
            "global_judge_weights": zero4,
            "deliberative_weights": zero4,
            "direct_weights": zero4,
            "final_evidence_weights": zero4,
            "sample_disagreement": zero1,
            "routing_gate": zero1,
            "direct_feature": logits.new_zeros(hidden.shape),
            "deliberative_feature": logits.new_zeros(hidden.shape),
            "fused_feature": hidden,
            "evidence_available": available,
            "pair_available": pair_available,
            "text_embedding": encoded["text"],
            "vision_embedding": encoded["vision"],
            "per_image_embedding": encoded["per_image"],
            "event_matched_similarity": encoded["event_matched_similarity"],
            "event_mismatched_similarity": encoded["event_mismatched_similarity"],
            "source_matched_similarity": encoded["source_matched_similarity"],
            "source_mismatched_similarity": encoded["source_mismatched_similarity"],
            "source_ranking_valid": encoded["source_ranking_valid"],
        }
