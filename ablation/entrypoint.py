"""Reuse the production train/evaluate CLI with isolated model and DDP adapters."""
import argparse
from functools import partial
import importlib
import json
import os
from pathlib import Path
import random
import sys
import time

from ablation.registry import config_hash, states


def rng_state():
    import numpy as np
    import torch
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state):
    import numpy as np
    import torch
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def install_contract():
    from mmfnd import engine
    original = engine._checkpoint_contract
    def contract(config):
        result = original(config)
        result["ablation_settings"] = json.dumps(config.get("ablation"), sort_keys=True)
        result["ablation_lgled"] = json.dumps(config["model"].get("lgled", {}), sort_keys=True)
        result["ablation_loss"] = json.dumps(config.get("loss", {}), sort_keys=True)
        result["ablation_protocol"] = config_hash(config.get("ablation_provenance", {}))
        result["ablation_seed"] = int(config["seed"])
        return result
    engine._checkpoint_contract = contract


def evaluation_adapter(original):
    def evaluate(model, *args, **kwargs):
        from mmfnd.engine import unwrap_model
        raw = unwrap_model(model)
        cfg = raw.runtime_config
        state = states(cfg["ablation"]["name"])
        captured = []
        def capture(module, inputs, outputs):
            for index in range(outputs["logits"].shape[0]):
                captured.append({
                    "logits_before_correction": outputs["preliminary_logits"][index].float().detach().cpu().tolist(),
                    "logits_final": outputs["logits"][index].float().detach().cpu().tolist(),
                    "evidence_available": outputs["evidence_available"][index].cpu().tolist(),
                    "pair_available": outputs["pair_available"][index].cpu().tolist(),
                })
        hook = raw.register_forward_hook(capture)
        try:
            metrics, rows, threshold = original(model, *args, **kwargs)
        finally:
            hook.remove()
        if len(rows) != len(captured):
            raise RuntimeError("Prediction/diagnostic row count mismatch")
        for row, extra in zip(rows, captured):
            row.update(extra)
            row.update(sample_id=row["id"], dataset=cfg["dataset"]["name"], seed=cfg["seed"],
                       variant=cfg["ablation"]["name"], true_label=row["label"], pred_label=row["prediction"],
                       fake_prob=row["fake_probability"], threshold=threshold,
                       fixed_mix_coefficient=cfg["ablation"]["fixed_mix_coefficient"])
            if not state["pair_relations"]:
                row["lgled"] = None
                row["lgled_status"] = "not_applicable: evaluator removed"
            if cfg["ablation"]["name"] == "mlp_classifier":
                row["modality_weights"] = None
                row["causal_gate_effects"] = None
                row["reasoner_status"] = "not_applicable: whole classification module replaced by MLP"
            if not state["decision_correction"]:
                row["uncertainty"] = None
                row["uncertainty_analysis"] = None
                row["correction_status"] = "disabled: logits_final equals logits_before_correction"
        if not state["pair_relations"]:
            metrics["lgled"] = None
            metrics["lgled_status"] = "not_applicable: evaluator removed"
        if cfg["ablation"]["name"] == "mlp_classifier":
            metrics["reasoner_status"] = "not_applicable: whole classification module replaced by MLP"
        metrics.update(variant=cfg["ablation"]["name"], seed=cfg["seed"], module_states=state,
                       fixed_mix_coefficient=cfg["ablation"]["fixed_mix_coefficient"],
                       parameter_counts=cfg["ablation_parameter_counts"])
        return metrics, rows, threshold
    return evaluate


def configure_entry(entry, action, resume=None):
    import torch
    from mmfnd import engine
    from ablation.model import AblationMMFND
    from ablation.loss import training_loss
    install_contract()
    entry.ExplainableMMFND = AblationMMFND
    entry.evaluate = evaluation_adapter(entry.evaluate)
    if action != "train":
        return
    entry.DDP = partial(torch.nn.parallel.DistributedDataParallel, find_unused_parameters=True)
    entry.multimodal_loss = training_loss
    original_reduce, original_save, original_load = entry.reduce_training_stats, entry.save_checkpoint, entry.load_checkpoint
    gathered = []
    def reduce(*args, **kwargs):
        result = original_reduce(*args, **kwargs)
        local = rng_state()
        gathered.clear()
        if torch.distributed.is_initialized():
            gathered.extend([None] * torch.distributed.get_world_size())
            torch.distributed.all_gather_object(gathered, local)
        else:
            gathered.append(local)
        return result
    def save(path, *args, **kwargs):
        original_save(path, *args, **kwargs)
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        rank_states = list(gathered) or [rng_state()]
        rank_states[0] = rng_state()
        checkpoint["ablation_rng_states"] = rank_states
        torch.save(checkpoint, path)
    restored = False
    def load(path, *args, **kwargs):
        nonlocal restored
        checkpoint = original_load(path, *args, **kwargs)
        if resume is not None and Path(path).resolve() == resume and not restored:
            rank = int(os.environ.get("RANK", 0))
            saved = checkpoint.get("ablation_rng_states")
            if saved is None or len(saved) != int(os.environ.get("WORLD_SIZE", 1)):
                raise ValueError("Resume requires matching world size and saved RNG states")
            restore_rng(saved[rank])
            restored = True
        return checkpoint
    entry.reduce_training_stats, entry.save_checkpoint, entry.load_checkpoint = reduce, save, load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("train", "evaluate"))
    args = parser.parse_args(sys.argv[1:2])
    rest = sys.argv[2:]
    resume = Path(rest[rest.index("--resume") + 1]).resolve() if "--resume" in rest else None
    entry = importlib.import_module(args.action)
    configure_entry(entry, args.action, resume)
    sys.argv = [args.action + ".py", *rest]
    if "--help" in rest or "-h" in rest:
        entry.main()
        return
    import torch
    config_path = Path(rest[rest.index("--config") + 1])
    config = json.loads(config_path.read_text())
    run_name = rest[rest.index("--run-name") + 1] if "--run-name" in rest else None
    run = resume.parent.parent if resume else (Path(config["train"]["output_dir"]) / run_name if run_name else None)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    error = None
    try:
        print(json.dumps({"ablation_module_states": states(config.get("ablation", {}).get("name", "full"))}, ensure_ascii=False), flush=True)
        entry.main()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if run is not None and run.exists():
            payload = {"rank": int(os.environ.get("RANK", 0)), "wall_seconds": time.perf_counter() - start,
                       "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
                       "memory_reason": None if torch.cuda.is_available() else "CPU run: CUDA memory not applicable",
                       "error": error}
            runtime_path = run / f"runtime_rank{payload['rank']}.json"
            attempts = []
            if runtime_path.exists():
                previous = json.loads(runtime_path.read_text())
                attempts = previous.get("attempts", [previous])
            attempts.append(dict(payload))
            payload["attempts"] = attempts
            payload["wall_seconds"] = sum(item["wall_seconds"] for item in attempts)
            measured = [item["peak_memory_allocated_bytes"] for item in attempts if item["peak_memory_allocated_bytes"] is not None]
            payload["peak_memory_allocated_bytes"] = max(measured, default=None)
            runtime_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
