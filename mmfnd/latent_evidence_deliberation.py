"""LLM-guided latent evidence deliberation (LG-LED).

This module never loads a language model.  The owning multimodal model passes
its already-loaded Qwen2 model into ``forward`` so text encoding and latent
adjudication share exactly the same transformer layers and LoRA adapters.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


EVIDENCE_ORDER = ("text", "vision", "intrinsic_feature", "interaction")
PAIR_ORDER = (
    ("text", "vision"),
    ("text", "intrinsic_feature"),
    ("text", "interaction"),
    ("vision", "intrinsic_feature"),
    ("vision", "interaction"),
    ("intrinsic_feature", "interaction"),
)
RELATION_ORDER = ("agreement", "ambiguity", "conflict")
EVIDENCE_INDEX = {name: index for index, name in enumerate(EVIDENCE_ORDER)}
PAIR_INDEX = {pair: index for index, pair in enumerate(PAIR_ORDER)}
PAIR_EVIDENCE_INDICES = tuple(
    (EVIDENCE_INDEX[left], EVIDENCE_INDEX[right]) for left, right in PAIR_ORDER
)

NUM_EVIDENCE = len(EVIDENCE_ORDER)
NUM_PAIRS = len(PAIR_ORDER)
NUM_JUDGES = NUM_PAIRS + 1
EVIDENCE_SLICE = slice(0, NUM_EVIDENCE)
PAIR_JUDGE_SLICE = slice(NUM_EVIDENCE, NUM_EVIDENCE + NUM_PAIRS)
GLOBAL_JUDGE_INDEX = NUM_EVIDENCE + NUM_PAIRS
LATENT_SEQUENCE_LENGTH = NUM_EVIDENCE + NUM_JUDGES


def strict_latent_attention_mask(sequence_length: int,
                                 device: torch.device) -> torch.Tensor:
    """Boolean attention mask (True blocks), also supporting no-global ablation.

    Evidence slots stay isolated so later layers cannot relay other evidence
    into a pair. Each pair reads only its two evidence slots and its own state;
    the global judge can read every slot. Apply this mask at EVERY layer.
    """
    if sequence_length not in {GLOBAL_JUDGE_INDEX, LATENT_SEQUENCE_LENGTH}:
        raise ValueError("strict latent attention expects 10 or 11 tokens")
    allowed = torch.eye(sequence_length, device=device, dtype=torch.bool)
    for pair_index, (left, right) in enumerate(PAIR_EVIDENCE_INDICES):
        allowed[NUM_EVIDENCE + pair_index, left] = True
        allowed[NUM_EVIDENCE + pair_index, right] = True
    if sequence_length == LATENT_SEQUENCE_LENGTH:
        allowed[GLOBAL_JUDGE_INDEX, :] = True
    return ~allowed


class SharedEvidenceProjector(nn.Module):
    """The single D -> H projection shared by every evidence role."""

    def __init__(self, evidence_dim: int, qwen_hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(evidence_dim, qwen_hidden_size)
        self.norm = nn.LayerNorm(qwen_hidden_size)

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(evidence))


class EvidenceRoleEmbedding(nn.Module):
    def __init__(self, qwen_hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(NUM_EVIDENCE, qwen_hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        roles = torch.arange(NUM_EVIDENCE, device=self.embedding.weight.device)
        return self.embedding(roles).unsqueeze(0).expand(batch_size, -1, -1)


class LatentJudgeTokenBank(nn.Module):
    def __init__(self, qwen_hidden_size: int):
        super().__init__()
        self.embedding = nn.Embedding(NUM_JUDGES, qwen_hidden_size)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, batch_size: int) -> torch.Tensor:
        indices = torch.arange(NUM_JUDGES, device=self.embedding.weight.device)
        return self.embedding(indices).unsqueeze(0).expand(batch_size, -1, -1)


class EvidentialRelationHead(nn.Module):
    def __init__(self, qwen_hidden_size: int):
        super().__init__()
        self.output = nn.Linear(qwen_hidden_size, len(RELATION_ORDER))

    def forward(self, pair_states: torch.Tensor) -> dict[str, torch.Tensor]:
        # Dirichlet evidence arithmetic stays in FP32 under BF16 autocast.
        evidence = F.softplus(self.output(pair_states).float())
        alpha = evidence + 1.0
        strength = alpha.sum(dim=-1)
        probabilities = alpha / strength.unsqueeze(-1)
        uncertainty = float(len(RELATION_ORDER)) / strength
        return {
            "relation_evidence": evidence,
            "relation_alpha": alpha,
            "relation_probs": probabilities,
            "relation_strength": strength,
            "relation_uncertainty": uncertainty,
        }


def _scalar_head(input_dim: int, hidden_dim: int, dropout: float,
                 sigmoid: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),
    ]
    if sigmoid:
        layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


class EvidenceConfidenceHead(nn.Module):
    def __init__(self, qwen_hidden_size: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(qwen_hidden_size)
        self.output = _scalar_head(qwen_hidden_size, hidden_dim, dropout, True)

    def forward(self, evidence_states: torch.Tensor) -> torch.Tensor:
        return self.output(self.norm(evidence_states)).squeeze(-1)


class CriticalMinorityHead(nn.Module):
    def __init__(self, qwen_hidden_size: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.output = _scalar_head(qwen_hidden_size + 4, hidden_dim, dropout, True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(-1)


class GlobalJudgeHead(nn.Module):
    def __init__(self, qwen_hidden_size: int):
        super().__init__()
        self.output = nn.Linear(qwen_hidden_size, NUM_EVIDENCE)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        return self.output(global_state)


class EvidenceAdjudicationHead(nn.Module):
    def __init__(self, qwen_hidden_size: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.output = _scalar_head(qwen_hidden_size + 5, hidden_dim, dropout, False)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.output(features).squeeze(-1)


class DirectFusionHead(nn.Module):
    def __init__(self, evidence_dim: int):
        super().__init__()
        self.output = nn.Linear(evidence_dim + 1, 1)

    def forward(self, evidence: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        features = torch.cat([evidence, confidence.unsqueeze(-1)], dim=-1)
        return self.output(features).squeeze(-1)


class MLPPairJudge(nn.Module):
    """Non-LLM pairwise baseline, instantiated only for the requested ablation."""

    def __init__(self, evidence_dim: int, qwen_hidden_size: int,
                 hidden_dim: int, dropout: float):
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(evidence_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, qwen_hidden_size),
            nn.LayerNorm(qwen_hidden_size),
        )

    def forward(self, evidence: torch.Tensor) -> torch.Tensor:
        pairs = []
        for left, right in PAIR_EVIDENCE_INDICES:
            left_value, right_value = evidence[:, left], evidence[:, right]
            pairs.append(torch.cat([
                left_value, right_value, (left_value - right_value).abs(),
                left_value * right_value,
            ], dim=-1))
        return self.output(torch.stack(pairs, dim=1))


@dataclass(frozen=True)
class LatentJudgeRuntime:
    qwen_layer_count: int
    first_shared_layer: int
    last_shared_layer: int


class LLMGuidedLatentEvidenceDeliberation(nn.Module):
    """Uncertainty-aware cross-evidence adjudication in shared Qwen space."""

    evidence_order = EVIDENCE_ORDER
    pair_order = PAIR_ORDER
    relation_order = RELATION_ORDER

    def __init__(self, evidence_dim: int, qwen_hidden_size: int, cfg: dict):
        super().__init__()
        self.evidence_dim = int(evidence_dim)
        self.qwen_hidden_size = int(qwen_hidden_size)
        self.projector_type = str(cfg.get("projector_type", "shared"))
        self.judge_type = str(cfg.get("judge_type", "qwen_latent"))
        self.latent_judge_num_layers = int(cfg.get("latent_judge_num_layers", 2))
        if self.latent_judge_num_layers not in {1, 2, 4}:
            raise ValueError("latent_judge_num_layers must be one of 1, 2 or 4")
        if self.projector_type not in {"shared", "independent"}:
            raise ValueError("projector_type must be shared or independent")
        if self.judge_type not in {"qwen_latent", "mlp_pair"}:
            raise ValueError("judge_type must be qwen_latent or mlp_pair")

        if self.projector_type == "shared":
            self.shared_projector = SharedEvidenceProjector(
                self.evidence_dim, self.qwen_hidden_size
            )
            self.independent_projectors = None
        else:
            self.shared_projector = None
            self.independent_projectors = nn.ModuleList([
                SharedEvidenceProjector(self.evidence_dim, self.qwen_hidden_size)
                for _ in range(NUM_EVIDENCE)
            ])
        self.role_embedding = EvidenceRoleEmbedding(self.qwen_hidden_size)
        self.role_norm = nn.LayerNorm(self.qwen_hidden_size)
        self.judge_tokens = LatentJudgeTokenBank(self.qwen_hidden_size)

        hidden_dim = int(cfg.get("head_hidden_dim", 256))
        dropout = float(cfg.get("dropout", 0.1))
        self.relation_head = EvidentialRelationHead(self.qwen_hidden_size)
        self.confidence_head = EvidenceConfidenceHead(
            self.qwen_hidden_size, hidden_dim, dropout
        )
        self.minority_head = CriticalMinorityHead(
            self.qwen_hidden_size, hidden_dim, dropout
        )
        self.global_judge_head = GlobalJudgeHead(self.qwen_hidden_size)
        self.adjudication_head = EvidenceAdjudicationHead(
            self.qwen_hidden_size, hidden_dim, dropout
        )
        self.direct_fusion_head = DirectFusionHead(self.evidence_dim)
        self.mlp_pair_judge = (
            MLPPairJudge(
                self.evidence_dim, self.qwen_hidden_size, hidden_dim, dropout
            )
            if self.judge_type == "mlp_pair" else None
        )

        membership = torch.zeros(NUM_EVIDENCE, NUM_PAIRS)
        for pair_index, (left, right) in enumerate(PAIR_EVIDENCE_INDICES):
            membership[left, pair_index] = 1.0
            membership[right, pair_index] = 1.0
        self.register_buffer("pair_membership", membership, persistent=False)
        self.routing_threshold = float(cfg.get("routing_threshold", 0.30))
        self.routing_scale = float(cfg.get("routing_scale", 10.0))
        self.use_role_embedding = bool(cfg.get("use_role_embedding", True))
        self.use_evidential_relation = bool(
            cfg.get("use_evidential_relation", True)
        )
        self.use_uncertainty = bool(cfg.get("use_uncertainty", True))
        self.use_critical_minority = bool(cfg.get("use_critical_minority", True))
        self.use_global_judge = bool(cfg.get("use_global_judge", True))
        self.use_selective_routing = bool(cfg.get("use_selective_routing", True))
        self.fusion_residual_scale = nn.Parameter(torch.tensor(0.1))
        self.fusion_norm = nn.LayerNorm(self.evidence_dim)

    def _project(self, evidence: torch.Tensor) -> torch.Tensor:
        if self.shared_projector is not None:
            projected = self.shared_projector(evidence)
        else:
            projected = torch.stack([
                projector(evidence[:, index])
                for index, projector in enumerate(self.independent_projectors)
            ], dim=1)
        if self.use_role_embedding:
            projected = projected + self.role_embedding(evidence.size(0)).to(
                projected.dtype
            )
        return self.role_norm(projected)

    @staticmethod
    def _masked_softmax(scores: torch.Tensor,
                        available: torch.Tensor) -> torch.Tensor:
        if not bool(available.any(dim=1).all().item()):
            raise ValueError("every sample must contain at least one evidence view")
        masked = scores.masked_fill(~available, torch.finfo(scores.dtype).min)
        weights = masked.softmax(dim=1) * available.to(scores.dtype)
        return weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )

    def selected_qwen_layers(self, qwen_model: nn.Module) -> list[nn.Module]:
        layers = list(qwen_model.layers)
        if len(layers) < self.latent_judge_num_layers:
            raise ValueError(
                f"Qwen has {len(layers)} layers, cannot reuse the last "
                f"{self.latent_judge_num_layers}"
            )
        # A plain list preserves the original module objects without registering
        # or copying them under LG-LED.
        return layers[-self.latent_judge_num_layers:]

    def runtime(self, qwen_model: nn.Module) -> LatentJudgeRuntime:
        layer_count = len(qwen_model.layers)
        return LatentJudgeRuntime(
            qwen_layer_count=layer_count,
            first_shared_layer=layer_count - self.latent_judge_num_layers,
            last_shared_layer=layer_count - 1,
        )

    def _run_qwen_last_layers(self, latent_sequence: torch.Tensor,
                              qwen_model: nn.Module) -> torch.Tensor:
        shared_layers = self.selected_qwen_layers(qwen_model)
        if qwen_model.config._attn_implementation not in {"eager", "sdpa"}:
            raise ValueError("strict latent attention requires Qwen eager or sdpa")
        target_dtype = next(qwen_model.parameters()).dtype
        hidden_states = latent_sequence.to(dtype=target_dtype)
        batch_size, sequence_length = hidden_states.shape[:2]
        cache_position = torch.arange(sequence_length, device=hidden_states.device)
        position_ids = cache_position.unsqueeze(0).expand(batch_size, -1)
        blocked = strict_latent_attention_mask(sequence_length, hidden_states.device)
        # Pass the explicit 4-D additive mask directly to every shared layer.
        # -inf gives blocked edges exactly zero probability, including in BF16.
        attention_mask = hidden_states.new_zeros(sequence_length, sequence_length)
        attention_mask.masked_fill_(blocked, float("-inf"))
        attention_mask = attention_mask[None, None].expand(batch_size, 1, -1, -1)
        position_embeddings = qwen_model.rotary_emb(hidden_states, position_ids)
        for layer in shared_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        return qwen_model.norm(hidden_states)

    def forward(self, evidence_features: list[torch.Tensor],
                qwen_model: nn.Module,
                available: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
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
        evidence = evidence * available.unsqueeze(-1).to(evidence.dtype)

        projected = self._project(evidence)
        projected = projected * available.unsqueeze(-1).to(projected.dtype)
        judges = self.judge_tokens(evidence.size(0)).to(projected.dtype)
        latent_sequence = torch.cat([projected, judges], dim=1)
        if latent_sequence.size(1) != LATENT_SEQUENCE_LENGTH:
            raise RuntimeError("LG-LED latent sequence must contain 11 tokens")

        if self.judge_type == "qwen_latent":
            latent_output = self._run_qwen_last_layers(latent_sequence, qwen_model)
            evidence_states = latent_output[:, EVIDENCE_SLICE]
            pair_states = latent_output[:, PAIR_JUDGE_SLICE]
            global_state = latent_output[:, GLOBAL_JUDGE_INDEX]
        else:
            evidence_states = projected
            pair_states = self.mlp_pair_judge(evidence)
            global_state = projected.mean(dim=1)
            latent_output = torch.cat([
                evidence_states, pair_states, global_state.unsqueeze(1)
            ], dim=1)

        relation = self.relation_head(pair_states)
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
        minority = self.minority_head(minority_features).float()
        if not self.use_critical_minority:
            minority = torch.zeros_like(minority)
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
        direct_scores = self.direct_fusion_head(evidence, confidence).float()
        direct_weights = self._masked_softmax(direct_scores, available)

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
        routing_gate = torch.sigmoid(
            self.routing_scale * (sample_disagreement - self.routing_threshold)
        )
        if not self.use_selective_routing:
            routing_gate = torch.ones_like(routing_gate)
        fused = (
            (1.0 - routing_gate) * direct_feature
            + routing_gate * deliberative_feature
        )
        final_evidence_weights = (
            (1.0 - routing_gate) * direct_weights
            + routing_gate * deliberative_weights
        )
        evidence_mean = (
            evidence_float * available.unsqueeze(-1).to(evidence_float.dtype)
        ).sum(dim=1) / available.sum(dim=1, keepdim=True).clamp_min(1).to(
            evidence_float.dtype
        )
        fused = self.fusion_norm(
            fused + self.fusion_residual_scale * evidence_mean
        )

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
