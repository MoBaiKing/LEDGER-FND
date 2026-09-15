"""Versioned canonical definitions. Seeds come from training, never constants."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = 2
LEGACY = dict(zip((f"exp{i}" for i in range(1, 10)), (
    "mlp_classifier", "non_llm_evaluator", "no_uncertainty", "no_critical_minority",
    "no_global_token", "fixed_mixture", "direct_only", "deliberation_only", "no_decision_correction")))
VARIANTS = ("full", *[name for name in LEGACY.values() if name != "full"])
CONTROLS = ("path_direct_reference", "path_deliberation_reference", "dual_path_reference")
EXPERIMENTS = VARIANTS
DATASETS = ("weibo21", "weibo", "gossipcop")
FULL_FLAGS = {"enabled": True, "projector_type": "shared", "judge_type": "qwen_latent",
              "use_role_embedding": True, "use_evidential_relation": True,
              "use_uncertainty": True, "use_critical_minority": True,
              "use_global_judge": True, "use_selective_routing": True}
NON_LLM_DEFAULT = {"hidden_dim": 256, "num_layers": 2, "num_heads": 8,
                   "feedforward_dim": 1024, "dropout": 0.1}


def canonical(name):
    name = LEGACY.get(name, name)
    if name not in VARIANTS + CONTROLS:
        raise ValueError(f"Unknown variant: {name}")
    return name


def settings(name="full", fixed=0.5, non_llm=None):
    return {"version": VERSION, "name": canonical(name), "fixed_mix_coefficient": fixed,
            "strict_dependency_check": True, "non_llm_evaluator": {**NON_LLM_DEFAULT, **(non_llm or {})}}


def states(name):
    name = canonical(name)
    direct = name in {"direct_only", "path_direct_reference"}
    simple = name == "mlp_classifier"
    delib = name in {"deliberation_only", "path_deliberation_reference"}
    evaluator = not (direct or simple)
    correction = name not in {"mlp_classifier", "direct_only", "no_decision_correction", *CONTROLS}
    return {
        "variant": name, "upstream_T_V_E_X": True,
        "evaluator": "none" if not evaluator else ("random_transformer" if name == "non_llm_evaluator" else "shared_qwen"),
        "pair_relations": evaluator, "explicit_relation_uncertainty": evaluator and name != "no_uncertainty",
        "critical_minority": evaluator and name != "no_critical_minority",
        "global_summary": "none" if not evaluator else ("masked_pair_mean" if name == "no_global_token" else "learned_token"),
        "direct_candidate": not simple and not delib,
        "direct_confidence": "constant_zero" if direct or name == "dual_path_reference" else ("latent" if evaluator and not delib else "none"),
        "deliberation_candidate": evaluator,
        "mixture": "not_applicable" if simple else ("direct" if direct else ("deliberation" if delib else ("fixed" if name == "fixed_mixture" else "dynamic"))),
        "raw_mean_residual": name not in {"mlp_classifier", "direct_only", "deliberation_only", *CONTROLS},
        "decision_correction": correction,
        "comparison_group": "path_control" if name in CONTROLS else "main",
    }


def unavailable_losses(name):
    state = states(name)
    result = {}
    if name == "mlp_classifier":
        result["causal_probe_classification"] = "Whole classification module replaced; reliability probe is not executed"
        result["causal_fidelity"] = "Whole classification module replaced; reliability probe is not executed"
    if not state["pair_relations"]:
        result["evidential_regularization"] = "Relation head removed; relation strength is undefined"
    if not state["decision_correction"]:
        result["uncertainty_calibration"] = "IURD uncertainty/correction module bypassed"
    return result


def validate(config):
    value = config.get("ablation", settings())
    if value.get("version") != VERSION:
        raise ValueError("Legacy ablation v1 is incompatible; regenerate a new suite")
    if set(value) != set(settings()):
        raise ValueError("Unknown/missing ablation settings; use resolved v2 schema")
    name = canonical(value["name"])
    if value["strict_dependency_check"] is not True:
        raise ValueError("strict_dependency_check must be true")
    mix = value["fixed_mix_coefficient"]
    if not isinstance(mix, (int, float)) or not 0 <= mix <= 1:
        raise ValueError("fixed_mix_coefficient must be in [0, 1]")
    cfg = value["non_llm_evaluator"]
    if set(cfg) != set(NON_LLM_DEFAULT):
        raise ValueError("Unknown/missing non-LLM evaluator settings")
    for key in ("hidden_dim", "num_layers", "num_heads", "feedforward_dim"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"non_llm_evaluator.{key} must be a positive integer")
    if cfg["hidden_dim"] % cfg["num_heads"] or cfg["hidden_dim"] % 2:
        raise ValueError("Transformer hidden_dim must be even and divisible by num_heads")
    if not 0 <= cfg["dropout"] < 1:
        raise ValueError("Transformer dropout must be in [0,1)")
    if config["model"].get("architecture_version") != "qwen_lora_lgled_v3":
        raise ValueError("not_applicable: architecture is not the v3 dual-path LG-LED model")
    for key, expected in FULL_FLAGS.items():
        if config["model"].get("lgled", {}).get(key, expected) != expected:
            raise ValueError(f"Conflicting base model.lgled.{key}; use ablation.name")
    dataset = config["dataset"]
    if dataset.get("positive_label") != 0 or dataset.get("class_names") != {"0": "fake", "1": "real"}:
        raise ValueError("Required label contract: Fake=0, Real=1")
    for key, reason in unavailable_losses(name).items():
        if config["loss"].get(key, 0) != 0:
            raise ValueError(f"Disabled loss {key} is nonzero: {reason}")
    return value


def resolve_config(base, name, seed, output_dir, fixed=0.5, non_llm=None):
    config = deepcopy(base)
    config["ablation"] = settings(name, fixed, non_llm)
    config["seed"] = int(seed)
    config["train"]["output_dir"] = str(output_dir)
    config["model"]["lgled"]["export_diagnostics"] = True
    config["ablation_disabled_losses"] = unavailable_losses(name)
    config["ablation_original_loss_weights"] = deepcopy(base["loss"])
    for loss in config["ablation_disabled_losses"]:
        config["loss"][loss] = 0.0
    validate(config)
    config["ablation_module_states"] = states(name)
    return config


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
