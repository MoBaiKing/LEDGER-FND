from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from peft.tuners.lora import model as peft_lora_model
from transformers import AutoModel

from mmfnd.latent_evidence_deliberation import LLMGuidedLatentEvidenceDeliberation

# A stale CPU-only bitsandbytes package can make PEFT
# probe CUDA-only adapter dispatch and print misleading errors on Apple Silicon.
# This model uses ordinary PyTorch LoRA, so disable only that optional dispatcher
# when CUDA is unavailable; CUDA behavior remains untouched.
if not torch.cuda.is_available():
    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False


def masked_mean(sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.unsqueeze(-1).to(sequence.dtype)
    return (sequence * mask).sum(1) / mask.sum(1).clamp_min(1.0)


class CausalReliabilityGate(nn.Module):
    """Fuse text, image and intrinsic evidence with leave-one-view probes.

    The probe estimates how much its prediction changes when one internal view is
    removed.  This is a model-level intervention signal, not a claim of real-world
    causality.  It is combined with a learned reliability score before fusion.
    """

    def __init__(self, dim: int, effect_scale: float = 1.0):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim // 2), nn.GELU(), nn.Linear(dim // 2, 1)
        )
        self.probe = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 2)
        )
        # The derived intrinsic view is randomly initialized while text and image
        # start from pretrained encoders.  A conservative prior prevents the new
        # view from occupying one third of the fusion before it becomes reliable.
        self.view_bias = nn.Parameter(torch.tensor([0.0, 0.0, -1.0]))
        self.effect_scale = float(effect_scale)

    def forward(self, features: list[torch.Tensor],
                available: torch.Tensor | None = None) -> tuple:
        stacked = torch.stack(features, dim=1)
        if available is None:
            available = torch.ones(
                stacked.shape[:2], dtype=torch.bool, device=stacked.device
            )
        available_float = available.to(stacked.dtype)
        full = (
            stacked * available_float.unsqueeze(-1)
        ).sum(dim=1) / available_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        full_logits = self.probe(full)
        full_probs = full_logits.softmax(dim=-1)
        ablated_logits, effects = [], []
        for index in range(stacked.size(1)):
            retained = available_float.clone()
            retained[:, index] = 0.0
            without = (
                stacked * retained.unsqueeze(-1)
            ).sum(dim=1) / retained.sum(dim=1, keepdim=True).clamp_min(1.0)
            logits = self.probe(without)
            ablated_logits.append(logits)
            effects.append(
                0.5 * (full_probs - logits.softmax(dim=-1)).abs().sum(dim=-1)
            )
        ablated_logits = torch.stack(ablated_logits, dim=1)
        effects = torch.stack(effects, dim=1) * available_float
        scores = torch.cat([self.scorer(feature) for feature in features], dim=1)
        scores = scores + self.view_bias.unsqueeze(0)
        scores = scores + self.effect_scale * effects
        if available is not None:
            scores = scores.masked_fill(~available.bool(), -1e4)
        weights = scores.softmax(dim=1)
        fused = sum(weights[:, i:i + 1] * feature for i, feature in enumerate(features))
        return fused, weights, effects, full_logits, ablated_logits


class IntrinsicUncertaintyResidualDisentangler(nn.Module):
    """IURD: calibrate uncertainty and isolate conflict-aligned residuals.

    The uncertainty definition is specific to this model and combines four
    internal signals: decision-boundary ambiguity, disagreement among the text,
    image and event views, leave-one-view intervention sensitivity, and LG-LED
    latent relation uncertainty. A bounded correction removes only the component
    of the joint representation aligned with the dominant cross-view conflict.
    """

    component_names = (
        "decision_margin",
        "view_conflict",
        "intervention_sensitivity",
        "latent_relation_uncertainty",
    )

    def __init__(self, dim: int, max_correction: float = 0.1,
                 max_temperature_delta: float = 0.5):
        super().__init__()
        self.component_logits = nn.Parameter(torch.zeros(4))
        self.correction_logit = nn.Parameter(torch.tensor(-2.0))
        self.temperature_logit = nn.Parameter(torch.tensor(-2.0))
        self.norm = nn.LayerNorm(dim)
        self.max_correction = float(max_correction)
        self.max_temperature_delta = float(max_temperature_delta)

    def forward(self, feature: torch.Tensor, views: list[torch.Tensor],
                consensus: torch.Tensor, intervention_effects: torch.Tensor,
                latent_relation_uncertainty: torch.Tensor,
                preliminary_logits: torch.Tensor,
                available: torch.Tensor | None = None) -> dict:
        preliminary_probabilities = preliminary_logits.softmax(dim=-1)
        decision_margin = (
            1.0
            - (
                preliminary_probabilities[:, 0]
                - preliminary_probabilities[:, 1]
            ).abs()
        ).clamp(0.0, 1.0)

        stacked = torch.stack(views, dim=1)
        expanded_consensus = consensus.unsqueeze(1)
        view_distances = (
            1.0
            - F.cosine_similarity(stacked, expanded_consensus, dim=-1)
        ).mul(0.5).clamp(0.0, 1.0)
        if available is None:
            available = torch.ones_like(view_distances, dtype=torch.bool)
        available_float = available.to(view_distances.dtype)
        view_distances = view_distances * available_float
        view_conflict = view_distances.sum(dim=-1) / available_float.sum(
            dim=-1
        ).clamp_min(1.0)
        intervention_sensitivity = (
            2.0 * (intervention_effects * available_float).sum(dim=-1)
            / available_float.sum(dim=-1).clamp_min(1.0)
        ).clamp(0.0, 1.0)

        components = torch.stack([
            decision_margin,
            view_conflict,
            intervention_sensitivity,
            latent_relation_uncertainty,
        ], dim=-1)
        component_weights = self.component_logits.softmax(dim=-1)
        uncertainty = (
            components * component_weights.unsqueeze(0)
        ).sum(dim=-1)

        # Extract the residual direction associated with the most discordant
        # internal views. Only the projection onto that direction is removed.
        residual_logits = (view_distances / 0.1).masked_fill(
            ~available, -1e4
        )
        residual_weights = residual_logits.softmax(dim=-1) * available_float
        conflict_residual = (
            residual_weights.unsqueeze(-1)
            * (stacked - expanded_consensus)
        ).sum(dim=1)
        conflict_direction = F.normalize(conflict_residual, dim=-1)
        aligned_projection = (
            feature * conflict_direction
        ).sum(dim=-1, keepdim=True) * conflict_direction
        correction_scale = (
            self.max_correction * torch.sigmoid(self.correction_logit)
        )
        correction = (
            correction_scale
            * uncertainty.detach().unsqueeze(-1)
            * aligned_projection
        )
        corrected = self.norm(feature - correction)

        # Temperature correction cannot flip a class; it only softens uncertain
        # decisions. Detaching uncertainty prevents the classifier from gaming it.
        temperature = (
            1.0
            + self.max_temperature_delta
            * torch.sigmoid(self.temperature_logit)
            * uncertainty.detach()
        )
        return {
            "feature": corrected,
            "uncertainty": uncertainty,
            "components": components,
            "component_weights": component_weights,
            "temperature": temperature,
            "correction_norm": correction.norm(dim=-1),
            "correction_scale": correction_scale,
        }


class IntrinsicEventEvidence(nn.Module):
    """Build evidence only from relations inside the supplied text and images.

    The module performs text self-relation modeling, local text-image alignment,
    and per-image event construction. Cross-image dispersion is computed only when
    a sample supplies multiple images. The visual evidence term also contains
    text-image discrepancy, so it is not itself a pure multi-image consistency
    measure or a provenance-verification score.
    """

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.text_relation = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        self.text_to_visual = nn.MultiheadAttention(
            dim, heads, dropout=dropout, batch_first=True
        )
        bottleneck = max(dim // 2, 64)
        self.event_fusion = nn.Sequential(
            nn.LayerNorm(dim * 7), nn.Linear(dim * 7, bottleneck), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(bottleneck, dim),
        )
        self.score_projection = nn.Sequential(
            nn.Linear(3, dim), nn.Tanh(),
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def aggregate(features: torch.Tensor, owners: torch.Tensor,
                  batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        total = features.new_zeros(batch_size, features.size(-1))
        squared = features.new_zeros(batch_size, features.size(-1))
        counts = features.new_zeros(batch_size, 1)
        total.index_add_(0, owners, features.to(total.dtype))
        # Under CUDA BF16 autocast, square may be promoted to FP32 while the
        # accumulator created from the input remains BF16. index_add_ requires
        # an exact dtype match.
        squared.index_add_(0, owners, features.square().to(squared.dtype))
        counts.index_add_(
            0, owners, torch.ones_like(owners, dtype=features.dtype).unsqueeze(-1)
        )
        mean = total / counts.clamp_min(1.0)
        variance = (squared / counts.clamp_min(1.0) - mean.square()).clamp_min(0.0)
        # Keep the dispersion as mean variance instead of sqrt(variance).  For a
        # single-image sample the variance is exactly zero; sqrt has an infinite
        # derivative at zero and produced finite forward values but NaN gradients
        # on MPS. Mean variance is zero in the same case and has stable gradients.
        return mean, variance.mean(dim=-1)

    def forward(self, text_tokens: torch.Tensor, text_mask: torch.Tensor,
                visual_tokens: torch.Tensor, per_image: torch.Tensor,
                owners: torch.Tensor) -> dict:
        batch_size = text_tokens.size(0)
        related_text, _ = self.text_relation(
            text_tokens, text_tokens, text_tokens,
            key_padding_mask=~text_mask.bool(), need_weights=False,
        )
        text_summary = masked_mean(text_tokens, text_mask)
        relation_summary = masked_mean(related_text, text_mask)
        text_conflict = masked_mean((text_tokens - related_text).abs(), text_mask)

        image_text_tokens = text_tokens[owners]
        image_text_mask = text_mask[owners]
        aligned_text, alignment = self.text_to_visual(
            image_text_tokens, visual_tokens, visual_tokens,
            need_weights=True, average_attn_weights=False,
        )
        aligned_text_summary = masked_mean(aligned_text, image_text_mask)
        # Reuse the forward alignment map to summarize the attended patches.
        # This removes a second cross-attention block while retaining a
        # text-conditioned visual event representation.
        patch_attention = alignment.mean(dim=(1, 2))
        aligned_visual_summary = torch.bmm(
            patch_attention.unsqueeze(1), visual_tokens
        ).squeeze(1)
        image_text_summary = text_summary[owners]
        image_relation_summary = relation_summary[owners]
        image_text_conflict = text_conflict[owners]
        per_image_event = self.event_fusion(torch.cat([
            image_relation_summary,
            image_text_conflict,
            per_image,
            (image_text_summary - per_image).abs(),
            image_text_summary * per_image,
            aligned_text_summary,
            aligned_visual_summary,
        ], dim=-1))
        event, event_dispersion = self.aggregate(
            per_image_event, owners, batch_size
        )
        _, visual_dispersion = self.aggregate(per_image, owners, batch_size)
        image_counts = per_image.new_zeros(batch_size)
        image_counts.index_add_(
            0, owners,
            torch.ones_like(owners, dtype=per_image.dtype),
        )
        multi_image_available = image_counts > 1

        global_discrepancy_per_image = (
            1.0 - F.cosine_similarity(image_text_summary, per_image, dim=-1)
        ).mul(0.5).clamp(0.0, 1.0)
        local_discrepancy_per_image = (
            1.0 - F.cosine_similarity(
                aligned_text_summary, aligned_visual_summary, dim=-1
            )
        ).mul(0.5).clamp(0.0, 1.0)
        global_discrepancy, _ = self.aggregate(
            global_discrepancy_per_image.unsqueeze(-1), owners, batch_size
        )
        local_discrepancy, _ = self.aggregate(
            local_discrepancy_per_image.unsqueeze(-1), owners, batch_size
        )
        text_relation_ambiguity = (
            1.0 - F.cosine_similarity(text_summary, relation_summary, dim=-1)
        ).mul(0.5).clamp(0.0, 1.0)
        visual_evidence_inconsistency = (
            0.5 * global_discrepancy.squeeze(-1)
            + 0.25 * event_dispersion.tanh()
            + 0.25 * visual_dispersion.tanh()
        ).clamp(0.0, 1.0)
        contradiction = (
            0.5 * global_discrepancy.squeeze(-1)
            + 0.5 * local_discrepancy.squeeze(-1)
        ).clamp(0.0, 1.0)
        scores = torch.stack([
            text_relation_ambiguity, contradiction,
            visual_evidence_inconsistency,
        ], dim=-1)
        intrinsic = self.norm(event + self.score_projection(scores))

        # Average alignment over heads and all images belonging to a sample.
        per_image_alignment = alignment.mean(dim=1)
        alignment_sum = per_image_alignment.new_zeros(
            batch_size, per_image_alignment.size(1), per_image_alignment.size(2)
        )
        alignment_count = per_image_alignment.new_zeros(batch_size, 1, 1)
        alignment_sum.index_add_(0, owners, per_image_alignment)
        alignment_count.index_add_(
            0, owners,
            torch.ones(
                owners.size(0), 1, 1,
                device=owners.device, dtype=per_image_alignment.dtype,
            ),
        )
        return {
            "feature": intrinsic,
            "text_relation_ambiguity": text_relation_ambiguity,
            "event_contradiction": contradiction,
            "visual_evidence_inconsistency": visual_evidence_inconsistency,
            # Compatibility alias for existing result readers.
            "visual_source_inconsistency": visual_evidence_inconsistency,
            "multi_image_dispersion": (
                0.5 * event_dispersion.tanh()
                + 0.5 * visual_dispersion.tanh()
            ),
            "multi_image_available": multi_image_available,
            "alignment_attention": alignment_sum / alignment_count.clamp_min(1.0),
        }


class MultimodalIntrinsicEvidenceEncoder(nn.Module):
    """Module 1: Qwen-LoRA/SigLIP encoding and news-internal evidence."""

    def __init__(self, cfg: dict, dim: int, num_heads: int):
        super().__init__()
        if not bool(cfg.get("qwen_use_lora", True)):
            raise ValueError("CUTE-FND v3 forbids full Qwen-7B fine-tuning; enable LoRA")
        qwen_dtype_name = str(cfg.get("qwen_dtype", "bfloat16")).lower()
        qwen_dtypes = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        if qwen_dtype_name not in qwen_dtypes:
            raise ValueError(
                "model.qwen_dtype must be bfloat16, float16 or float32"
            )
        text_base = AutoModel.from_pretrained(
            cfg["text_backbone"], local_files_only=True,
            torch_dtype=qwen_dtypes[qwen_dtype_name],
        )
        text_base.config.use_cache = bool(cfg.get("qwen_use_cache", False))
        if text_base.config.use_cache:
            raise ValueError("model.qwen_use_cache must be false during v3 training")
        if bool(cfg.get("qwen_gradient_checkpointing", True)):
            text_base.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            text_base.enable_input_require_grads()
        lora = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=int(cfg.get("lora_rank", 8)),
            lora_alpha=int(cfg.get("lora_alpha", 16)),
            lora_dropout=float(cfg.get("lora_dropout", 0.05)),
            bias="none",
            target_modules=list(cfg.get(
                "lora_target_modules", ["q_proj", "v_proj"]
            )),
        )
        self.text_encoder = get_peft_model(text_base, lora)
        vision_model = AutoModel.from_pretrained(
            cfg["vision_backbone"], local_files_only=True
        )
        self.vision_encoder = getattr(vision_model, "vision_model", vision_model)
        self.text_proj = nn.Linear(text_base.config.hidden_size, dim)
        self.vision_proj = nn.Linear(self.vision_encoder.config.hidden_size, dim)
        self.intrinsic_evidence = IntrinsicEventEvidence(
            dim, num_heads, float(cfg["dropout"])
        )
        self.cross_consistency = nn.Sequential(
            nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.GELU()
        )
        self._freeze_vision(cfg)

    def shared_qwen_model(self) -> nn.Module:
        """Return the one PEFT-instrumented Qwen2Model used by both paths."""
        model = self.text_encoder.get_base_model()
        if not hasattr(model, "layers"):
            model = getattr(model, "model", model)
        if not all(hasattr(model, name) for name in ("layers", "rotary_emb", "norm")):
            raise TypeError(
                f"Unsupported shared Qwen base model: {type(model).__name__}"
            )
        return model

    def _freeze_vision(self, cfg: dict) -> None:
        if not cfg.get("freeze_vision", False):
            return
        for parameter in self.vision_encoder.parameters():
            parameter.requires_grad = False
        last_n = int(cfg.get("unfreeze_vision_last_n", 0))
        vision_layers = getattr(
            getattr(self.vision_encoder, "encoder", None), "layers", []
        )
        for layer in list(vision_layers)[-last_n:] if last_n > 0 else []:
            for parameter in layer.parameters():
                parameter.requires_grad = True
        post_layernorm = getattr(self.vision_encoder, "post_layernorm", None)
        if last_n > 0 and post_layernorm is not None:
            for parameter in post_layernorm.parameters():
                parameter.requires_grad = True

    @staticmethod
    def aggregate_images(
        features: torch.Tensor, owners: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        output = features.new_zeros(batch_size, features.size(-1))
        counts = features.new_zeros(batch_size, 1)
        output.index_add_(0, owners, features)
        counts.index_add_(
            0, owners,
            torch.ones_like(owners, dtype=features.dtype).unsqueeze(-1),
        )
        return output / counts.clamp_min(1.0)

    def forward(self, batch: dict, ablate_component: str | None = None) -> dict:
        text_out = self.text_encoder(
            input_ids=batch["text_input_ids"],
            attention_mask=batch["text_attention_mask"],
            return_dict=True,
        )
        text_tokens = self.text_proj(text_out.last_hidden_state)
        # The last non-padding Qwen state summarizes the preceding instruction
        # and news tokens under causal attention.
        if getattr(self, "general_last_valid_index", False):
            mask = batch["text_attention_mask"].bool()
            if not mask.any(-1).all():
                raise ValueError("empty token sequence")
            positions = torch.arange(mask.shape[1], device=mask.device)[None]
            last_indices = positions.expand_as(mask).masked_fill(~mask, -1).amax(-1)
        else:
            last_indices = (batch["text_attention_mask"].sum(dim=1) - 1).clamp_min(0)
        text = text_tokens[
            torch.arange(text_tokens.size(0), device=text_tokens.device),
            last_indices,
        ]
        raw_vision_tokens = self.vision_encoder(
            pixel_values=batch["pixel_values"]
        ).last_hidden_state
        vision_tokens = self.vision_proj(raw_vision_tokens)
        per_image = vision_tokens.mean(dim=1)
        vision = self.aggregate_images(
            per_image, batch["image_owner"], text.size(0)
        )

        if text.size(0) > 1:
            mismatched_vision = vision.roll(shifts=1, dims=0)
            event_matched_similarity = F.cosine_similarity(text, vision, dim=-1)
            event_mismatched_similarity = F.cosine_similarity(
                text, mismatched_vision, dim=-1
            )
            image_sum = per_image.new_zeros(text.size(0), per_image.size(-1))
            image_count = per_image.new_zeros(text.size(0), 1)
            image_sum.index_add_(0, batch["image_owner"], per_image)
            image_count.index_add_(
                0, batch["image_owner"],
                torch.ones(
                    per_image.size(0), 1, dtype=per_image.dtype,
                    device=per_image.device,
                ),
            )
            owner_count = image_count[batch["image_owner"]]
            own_source = (
                image_sum[batch["image_owner"]] - per_image
            ) / (owner_count - 1.0).clamp_min(1.0)
            source_ranking_valid = owner_count.squeeze(-1) > 1.0
            source_matched_similarity = F.cosine_similarity(
                per_image, own_source, dim=-1
            )
            source_mismatched_similarity = F.cosine_similarity(
                per_image, mismatched_vision[batch["image_owner"]], dim=-1
            )
        else:
            zero = text.new_zeros(text.size(0))
            event_matched_similarity = zero
            event_mismatched_similarity = zero
            source_matched_similarity = zero
            source_mismatched_similarity = zero
            source_ranking_valid = torch.zeros_like(zero, dtype=torch.bool)

        if ablate_component == "text":
            text_tokens = torch.zeros_like(text_tokens)
            text = torch.zeros_like(text)
        if ablate_component == "image":
            vision_tokens = torch.zeros_like(vision_tokens)
            per_image = torch.zeros_like(per_image)
            vision = torch.zeros_like(vision)

        intrinsic = self.intrinsic_evidence(
            text_tokens, batch["text_attention_mask"], vision_tokens,
            per_image, batch["image_owner"],
        )
        intrinsic_feature = intrinsic["feature"]
        if ablate_component in ("text", "image", "intrinsic_event"):
            intrinsic_feature = torch.zeros_like(intrinsic_feature)
        interaction = self.cross_consistency(
            torch.cat([(text - vision).abs(), text * vision], dim=-1)
        )
        return {
            "text": text,
            "text_tokens": text_tokens,
            "vision": vision,
            "vision_tokens": vision_tokens,
            "per_image": per_image,
            "intrinsic": intrinsic,
            "intrinsic_feature": intrinsic_feature,
            "interaction": interaction,
            "event_matched_similarity": event_matched_similarity,
            "event_mismatched_similarity": event_mismatched_similarity,
            "source_matched_similarity": source_matched_similarity,
            "source_mismatched_similarity": source_mismatched_similarity,
            "source_ranking_valid": source_ranking_valid,
        }


class UncertaintyAwareEvidenceReasoner(nn.Module):
    """Module 2: reliability probes and Qwen-guided latent adjudication."""

    def __init__(self, cfg: dict, dim: int, qwen_hidden_size: int):
        super().__init__()
        self.reliability_gate = CausalReliabilityGate(
            dim, float(cfg.get("causal_effect_scale", 0.5))
        )
        self.view_dropout_probability = float(
            cfg.get("view_dropout_probability", 0.0)
        )
        if not 0.0 <= self.view_dropout_probability <= 1.0:
            raise ValueError("view_dropout_probability must be between 0 and 1")
        lgled_cfg = cfg.get("lgled", {})
        if not bool(lgled_cfg.get("enabled", True)):
            raise ValueError("CUTE-FND v3 requires model.lgled.enabled=true")
        self.lgled = LLMGuidedLatentEvidenceDeliberation(
            dim, qwen_hidden_size, lgled_cfg
        )

    def forward(
        self, encoded: dict, ablate_component: str | None,
        qwen_model: nn.Module,
    ) -> dict:
        text = encoded["text"]
        vision = encoded["vision"]
        intrinsic_feature = encoded["intrinsic_feature"]
        available = torch.ones(
            text.size(0), 3, dtype=torch.bool, device=text.device
        )
        if ablate_component == "text":
            available[:, 0] = False
            available[:, 2] = False
        elif ablate_component == "image":
            available[:, 1] = False
            available[:, 2] = False
        elif ablate_component == "intrinsic_event":
            available[:, 2] = False
        if (
            self.training
            and ablate_component is None
            and self.view_dropout_probability > 0.0
        ):
            drop_sample = torch.rand(text.size(0), device=text.device) < (
                self.view_dropout_probability
            )
            drop_index = torch.randint(3, (text.size(0),), device=text.device)
            rows = torch.arange(text.size(0), device=text.device)[drop_sample]
            available[rows, drop_index[drop_sample]] = False

        views = [text, vision, intrinsic_feature]
        masked_views = [
            feature * available[:, index:index + 1].to(feature.dtype)
            for index, feature in enumerate(views)
        ]
        reliable, weights, effects, probe_logits, probe_ablated = (
            self.reliability_gate(masked_views, available)
        )
        interaction_available = available[:, 0] & available[:, 1]
        interaction = encoded["interaction"] * interaction_available.unsqueeze(
            -1
        ).to(encoded["interaction"].dtype)
        evidence_available = torch.cat([
            available, interaction_available.unsqueeze(-1)
        ], dim=-1)
        lgled = self.lgled(
            [*masked_views, interaction], qwen_model, evidence_available
        )
        return {
            "reliable": reliable,
            "views": masked_views,
            "view_available": available,
            "modality_weights": weights,
            "causal_effects": effects,
            "probe_logits": probe_logits,
            "probe_ablated": probe_ablated,
            "lgled": lgled,
            "fused_feature": lgled["fused_feature"],
            "latent_relation_uncertainty": lgled[
                "relation_uncertainty"
            ].mean(dim=1),
        }


class UncertaintyCalibratedDecision(nn.Module):
    """Module 3: conflict residual removal and calibrated binary decision."""

    def __init__(self, dim: int):
        super().__init__()
        self.final_norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, 2)
        self.calibrator = IntrinsicUncertaintyResidualDisentangler(dim)

    def forward(self, encoded: dict, reasoned: dict) -> dict:
        joint = self.final_norm(reasoned["fused_feature"])
        preliminary_logits = self.classifier(joint)
        calibrated = self.calibrator(
            joint,
            reasoned["views"],
            reasoned["reliable"],
            reasoned["causal_effects"],
            reasoned["latent_relation_uncertainty"],
            preliminary_logits,
            reasoned["view_available"],
        )
        logits = self.classifier(calibrated["feature"]) / calibrated[
            "temperature"
        ].unsqueeze(-1)
        return {
            "logits": logits,
            "preliminary_logits": preliminary_logits,
            "calibrated": calibrated,
        }


class ExplainableMMFND(nn.Module):
    """CUTE-FND v3 with one shared Qwen and LG-LED adjudication."""

    architecture_version = "qwen_lora_lgled_v3"

    def __init__(self, config: dict):
        super().__init__()
        self.runtime_config = config
        cfg = config["model"]
        configured_version = cfg.get("architecture_version")
        if configured_version != self.architecture_version:
            raise ValueError(
                f"architecture_version must be {self.architecture_version}, "
                f"got {configured_version!r}"
            )
        dim = int(cfg["hidden_dim"])
        num_heads = int(cfg["num_heads"])
        if num_heads <= 0 or dim % num_heads != 0:
            valid_heads = [
                value for value in range(1, min(dim, 32) + 1)
                if dim % value == 0
            ]
            raise ValueError(
                f"hidden_dim={dim} must be divisible by num_heads={num_heads}; "
                f"valid choices include {valid_heads}."
            )
        self.encoder = MultimodalIntrinsicEvidenceEncoder(cfg, dim, num_heads)
        qwen_hidden_size = int(
            self.encoder.shared_qwen_model().config.hidden_size
        )
        self.reasoner = UncertaintyAwareEvidenceReasoner(
            cfg, dim, qwen_hidden_size
        )
        self.decision = UncertaintyCalibratedDecision(dim)

    @property
    def lgled(self) -> LLMGuidedLatentEvidenceDeliberation:
        return self.reasoner.lgled

    def forward(self, batch: dict, ablate_component: str | None = None) -> dict:
        if ablate_component not in (None, "text", "image", "intrinsic_event"):
            raise ValueError(
                "ablate_component must be one of: text, image, intrinsic_event"
            )
        encoded = self.encoder(batch, ablate_component)
        reasoned = self.reasoner(
            encoded, ablate_component, self.encoder.shared_qwen_model()
        )
        decision = self.decision(encoded, reasoned)
        calibrated = decision["calibrated"]
        intrinsic = encoded["intrinsic"]
        lgled = reasoned["lgled"]
        return {
            "logits": decision["logits"],
            "preliminary_logits": decision["preliminary_logits"],
            "uncertainty": calibrated["uncertainty"],
            "uncertainty_components": calibrated["components"],
            "uncertainty_component_weights": calibrated["component_weights"],
            "uncertainty_temperature": calibrated["temperature"],
            "uncertainty_correction_norm": calibrated["correction_norm"],
            "uncertainty_correction_scale": calibrated["correction_scale"],
            "latent_relation_uncertainty": reasoned[
                "latent_relation_uncertainty"
            ],
            "modality_weights": reasoned["modality_weights"],
            "causal_gate_effects": reasoned["causal_effects"],
            "causal_probe_logits": reasoned["probe_logits"],
            "causal_probe_ablated_logits": reasoned["probe_ablated"],
            "text_relation_ambiguity": intrinsic["text_relation_ambiguity"],
            "event_contradiction": intrinsic["event_contradiction"],
            "visual_evidence_inconsistency": intrinsic[
                "visual_evidence_inconsistency"
            ],
            "visual_source_inconsistency": intrinsic[
                "visual_evidence_inconsistency"
            ],
            "multi_image_dispersion": intrinsic["multi_image_dispersion"],
            "multi_image_available": intrinsic["multi_image_available"],
            "intrinsic_alignment_attention": intrinsic["alignment_attention"],
            "evidence_features": torch.stack([
                encoded["text"], encoded["vision"],
                encoded["intrinsic_feature"], encoded["interaction"],
            ], dim=1),
            **lgled,
            "text_embedding": encoded["text"],
            "vision_embedding": encoded["vision"],
            "per_image_embedding": encoded["per_image"],
            "event_matched_similarity": encoded["event_matched_similarity"],
            "event_mismatched_similarity": encoded["event_mismatched_similarity"],
            "source_matched_similarity": encoded["source_matched_similarity"],
            "source_mismatched_similarity": encoded[
                "source_mismatched_similarity"
            ],
            "source_ranking_valid": encoded["source_ranking_valid"],
        }


def multimodal_loss(outputs: dict, labels: torch.Tensor, weights: dict,
                    label_smoothing: float = 0.0,
                    positive_label: int = 0) -> tuple[torch.Tensor, dict]:
    ce = F.cross_entropy(outputs["logits"], labels, label_smoothing=label_smoothing)
    text = F.normalize(outputs["text_embedding"], dim=-1)
    vision = F.normalize(outputs["vision_embedding"], dim=-1)
    similarity = text @ vision.T / 0.07
    targets = torch.arange(labels.size(0), device=labels.device)
    contrastive = 0.5 * (F.cross_entropy(similarity, targets) + F.cross_entropy(similarity.T, targets))
    causal_probe_classification = F.cross_entropy(
        outputs["causal_probe_logits"], labels, label_smoothing=label_smoothing
    )
    causal_fidelity = F.kl_div(
        F.log_softmax(outputs["causal_probe_logits"], dim=-1),
        F.softmax(outputs["logits"].detach(), dim=-1),
        reduction="batchmean",
    )
    if labels.size(0) > 1:
        event_counterfactual_ranking = F.relu(
            0.2
            - outputs["event_matched_similarity"]
            + outputs["event_mismatched_similarity"]
        ).mean()
        source_ranking_valid = outputs["source_ranking_valid"]
        if bool(source_ranking_valid.any().item()):
            visual_source_ranking = F.relu(
                0.2
                - outputs["source_matched_similarity"][source_ranking_valid]
                + outputs["source_mismatched_similarity"][source_ranking_valid]
            ).mean()
        else:
            visual_source_ranking = outputs["logits"].sum() * 0.0
    else:
        event_counterfactual_ranking = outputs["logits"].sum() * 0.0
        visual_source_ranking = outputs["logits"].sum() * 0.0
    negative_label = 1 - int(positive_label)
    positive_score = torch.tanh(
        (
            outputs["logits"][:, positive_label]
            - outputs["logits"][:, negative_label]
        ) / 2.0
    )
    positive_scores = positive_score[labels == positive_label]
    negative_scores = positive_score[labels == negative_label]
    if positive_scores.numel() > 0 and negative_scores.numel() > 0:
        causal_veracity_ranking = 0.2 * F.softplus(
            (
                0.3
                - positive_scores.unsqueeze(1)
                + negative_scores.unsqueeze(0)
            ) / 0.2
        ).mean()
    else:
        causal_veracity_ranking = outputs["logits"].sum() * 0.0
    probs = outputs["logits"].softmax(dim=-1)
    one_hot = F.one_hot(labels, num_classes=2).to(probs.dtype)
    brier = (probs - one_hot).pow(2).sum(dim=-1).mean()
    true_class_error = (
        1.0 - probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    ).detach()
    uncertainty_calibration = (
        F.smooth_l1_loss(outputs["uncertainty"], true_class_error)
        + brier
    )
    components = {
        "classification": ce,
        "contrastive": contrastive,
        "causal_probe_classification": causal_probe_classification,
        "causal_fidelity": causal_fidelity,
        "event_counterfactual_ranking": event_counterfactual_ranking,
        "visual_source_ranking": visual_source_ranking,
        "causal_veracity_ranking": causal_veracity_ranking,
        "uncertainty_calibration": uncertainty_calibration,
        # No fabricated pair labels are used. This optional scalar only reserves
        # an interface for future strength regularization and defaults to zero.
        "evidential_regularization": outputs["relation_strength"].mean(),
    }
    total = sum(float(weights.get(name, 0.0)) * value for name, value in components.items())
    return total, {name: float(value.detach()) for name, value in components.items()}
