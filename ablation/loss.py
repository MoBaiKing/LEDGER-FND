"""Loss dispatch: original Full objective; lazy, dependency-aware ablation losses."""
import torch
import torch.nn.functional as F
from mmfnd.model import multimodal_loss as production_loss
from ablation.registry import unavailable_losses


def compute_loss(outputs, labels, config):
    name = config["ablation"]["name"]
    weights = config["loss"]
    smoothing = float(config["train"]["label_smoothing"])
    if name == "full":
        return production_loss(outputs, labels, weights, smoothing, 0)
    disabled = unavailable_losses(name)
    values = {}
    for key, weight in weights.items():
        if key in disabled:
            if weight != 0:
                raise ValueError(f"Disabled loss enabled: {key}")
            continue
        if weight == 0:
            continue
        if key == "classification":
            value = F.cross_entropy(outputs["logits"], labels, label_smoothing=smoothing)
        elif key == "contrastive":
            similarity = F.normalize(outputs["text_embedding"], dim=-1) @ F.normalize(outputs["vision_embedding"], dim=-1).T / 0.07
            target = torch.arange(labels.numel(), device=labels.device)
            value = 0.5 * (F.cross_entropy(similarity, target) + F.cross_entropy(similarity.T, target))
        elif key == "causal_probe_classification":
            value = F.cross_entropy(outputs["causal_probe_logits"], labels, label_smoothing=smoothing)
        elif key == "causal_fidelity":
            value = F.kl_div(F.log_softmax(outputs["causal_probe_logits"], -1),
                             F.softmax(outputs["logits"].detach(), -1), reduction="batchmean")
        elif key in {"event_counterfactual_ranking", "visual_source_ranking"}:
            if labels.numel() < 2:
                value = outputs["logits"].new_zeros(())
            elif key == "event_counterfactual_ranking":
                value = F.relu(0.2 - outputs["event_matched_similarity"] + outputs["event_mismatched_similarity"]).mean()
            else:
                valid = outputs["source_ranking_valid"]
                value = (F.relu(0.2 - outputs["source_matched_similarity"][valid] + outputs["source_mismatched_similarity"][valid]).mean()
                         if valid.any() else outputs["logits"].new_zeros(()))
        elif key == "causal_veracity_ranking":
            score = torch.tanh((outputs["logits"][:, 0] - outputs["logits"][:, 1]) / 2)
            pos, neg = score[labels == 0], score[labels == 1]
            value = (0.2 * F.softplus((0.3 - pos[:, None] + neg[None, :]) / 0.2).mean()
                     if pos.numel() and neg.numel() else outputs["logits"].new_zeros(()))
        elif key == "uncertainty_calibration":
            probs = outputs["logits"].softmax(-1)
            error = (1 - probs.gather(1, labels[:, None]).squeeze(1)).detach()
            brier = (probs - F.one_hot(labels, 2).to(probs.dtype)).square().sum(-1).mean()
            value = F.smooth_l1_loss(outputs["uncertainty"], error) + brier
        elif key == "evidential_regularization":
            value = outputs["relation_strength"].mean()
        else:
            raise ValueError(f"Unknown active loss: {key}")
        values[key] = value
    if not values:
        raise ValueError("No active loss")
    loss = sum(float(weights[key]) * value for key, value in values.items())
    return loss, {key: float(value.detach()) for key, value in values.items()}


def training_loss(outputs, *unused_args):
    return outputs["training_loss"], outputs["loss_components"]
