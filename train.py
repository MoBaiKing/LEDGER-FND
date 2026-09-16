#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import math
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from tqdm import tqdm
from transformers import get_cosine_schedule_with_warmup

from mmfnd.data import move_batch
from mmfnd.dataset_contract import bind_dataset_workspace, validate_dataset_semantics
from mmfnd.engine import (
    autocast_context, evaluate, load_checkpoint, save_checkpoint,
    unwrap_model, write_jsonl,
)
from mmfnd.evaluation import PROTOCOL, checkpoint_threshold_selection, evaluation_info, log_evaluation
from mmfnd.factory import build_loader, build_processor, build_model
from mmfnd.latent_evidence_deliberation import EVIDENCE_ORDER, RELATION_ORDER
from mmfnd.model import ExplainableMMFND, multimodal_loss
from mmfnd.utils import cleanup_distributed, dump_json, init_distributed, load_config, resolve_path, seed_everything


def build_optimizer(model, config: dict) -> AdamW:
    learning_rates = {
        "lora": float(config["train"]["lora_learning_rate"]),
        "vision": float(config["train"]["vision_learning_rate"]),
        "head": float(config["train"]["head_learning_rate"]),
    }
    weight_decay = float(config["train"]["weight_decay"])
    grouped: dict[tuple[str, bool], list[torch.nn.Parameter]] = {}
    for name, parameter in unwrap_model(model).named_parameters():
        if not parameter.requires_grad:
            continue
        role = "lora" if "lora_" in name else (
            "vision" if name.startswith("encoder.vision_encoder") else "head"
        )
        no_decay = name.endswith(".bias") or parameter.ndim == 1 or "norm" in name.lower()
        grouped.setdefault((role, no_decay), []).append(parameter)
    parameter_groups = [
        {
            "params": parameters,
            "lr": learning_rates[role],
            "weight_decay": 0.0 if no_decay else weight_decay,
            "name": f"{role}_{'no_decay' if no_decay else 'decay'}",
        }
        for (role, no_decay), parameters in grouped.items()
    ]
    return AdamW(parameter_groups)


def checkpoint_score(metrics: dict, monitor: str) -> float:
    if monitor != "macro_f1":
        raise ValueError("This protocol requires train.monitor=macro_f1 after validation threshold tuning")
    value = metrics.get(monitor)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"validation metric {monitor!r} is unavailable")
    return float(value)


def average_checkpoints(paths: list[Path], model, device) -> None:
    if not paths:
        raise RuntimeError("no validation checkpoints available for averaging")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
    for checkpoint in checkpoints:
        checkpoint_threshold_selection(checkpoint)
    reference_state = checkpoints[0]["model_state_dict"]
    reference_contract = checkpoints[0].get("checkpoint_contract")
    for checkpoint in checkpoints[1:]:
        state = checkpoint["model_state_dict"]
        if checkpoint.get("checkpoint_contract") != reference_contract:
            raise RuntimeError("top-k checkpoint architecture/contract mismatch")
        if state.keys() != reference_state.keys():
            raise RuntimeError("top-k checkpoint key mismatch")
        for name in state:
            if state[name].shape != reference_state[name].shape:
                raise RuntimeError(f"top-k checkpoint shape mismatch: {name}")
    averaged = {
        name: torch.stack([checkpoint["model_state_dict"][name].float() for checkpoint in checkpoints]).mean(0)
        for name in reference_state
    }
    if getattr(unwrap_model(model), "architecture_version", "") == "qwen_lora_lgled_masked_r1":
        raw = unwrap_model(model)
        if set(averaged) != {n for n,p in raw.named_parameters() if p.requires_grad}:
            raise RuntimeError("R1 averaged trainable key mismatch")
        merged = raw.state_dict(); merged.update(averaged)
        raw.load_state_dict(merged, strict=True)
        return
    incompatible = unwrap_model(model).load_state_dict(averaged, strict=False)
    trainable = {name for name, parameter in unwrap_model(model).named_parameters() if parameter.requires_grad}
    if incompatible.unexpected_keys or set(incompatible.missing_keys) & trainable:
        raise RuntimeError("averaged checkpoint cannot be loaded into current architecture")


def reduce_training_stats(running: dict[str, float], batches: int, device, distributed: bool):
    names = sorted(running)
    tensor = torch.tensor([running[name] for name in names] + [float(batches)], device=device, dtype=torch.float64)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    count = max(float(tensor[-1].item()), 1.0)
    return {name: float(tensor[index].item() / count) for index, name in enumerate(names)}


def lgled_training_diagnostics(outputs: dict) -> dict[str, float]:
    """Small detached summaries suitable for per-epoch DDP logging."""
    if outputs.get("architecture_version") == "qwen_lora_lgled_masked_r1":
        valid = outputs["pair_available"]
        means = (outputs["relation_probs"] * valid[...,None]).sum((0,1)) / valid.sum().clamp_min(1)
        return {f"r1/relation_{name}": float(means[i].detach()) for i,name in enumerate(("A","M","C"))}
    relation_means = outputs["relation_probs"].float().mean(dim=(0, 1))
    diagnostics = {
        f"lgled/mean_{name}": float(relation_means[index].detach())
        for index, name in enumerate(RELATION_ORDER)
    }
    diagnostics.update({
        "lgled/mean_relation_uncertainty": float(
            outputs["relation_uncertainty"].float().mean().detach()
        ),
        "lgled/mean_relation_strength": float(
            outputs["relation_strength"].float().mean().detach()
        ),
        "lgled/mean_disagreement": float(
            outputs["sample_disagreement"].float().mean().detach()
        ),
        "lgled/mean_routing_gate": float(
            outputs["routing_gate"].float().mean().detach()
        ),
    })
    for index, name in enumerate(EVIDENCE_ORDER):
        diagnostics.update({
            f"lgled/{name}_confidence": float(
                outputs["evidence_confidence"][:, index].float().mean().detach()
            ),
            f"lgled/{name}_deviation": float(
                outputs["evidence_deviation"][:, index].float().mean().detach()
            ),
            f"lgled/{name}_minority": float(
                outputs["minority_score"][:, index].float().mean().detach()
            ),
            f"lgled/{name}_final_weight": float(
                outputs["final_evidence_weights"][:, index].float().mean().detach()
            ),
        })
    return diagnostics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--r1-target-cache", type=Path)
    parser.add_argument("--final-test-frozen", action="store_true", help="Explicit final test after final weights/validation threshold are frozen")
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--early-stop-patience", type=int)
    parser.add_argument("--per-gpu-batch-size", type=int)
    parser.add_argument("--grad-accum-steps", type=int)
    parser.add_argument("--smoke-steps", type=int, default=0, help="Run only N distributed training steps")
    args = parser.parse_args()

    context = init_distributed()
    try:
        root = Path(__file__).resolve().parent
        config = load_config(root / args.config)
        manifest_dir = bind_dataset_workspace(root, config, args.dataset, args.manifest_dir)
        if args.seed is not None:
            config["seed"] = int(args.seed)
        if args.epochs is not None:
            if args.epochs <= 0:
                raise ValueError("--epochs must be positive")
            config["train"]["epochs"] = int(args.epochs)
        if args.early_stop_patience is not None:
            if args.early_stop_patience <= 0:
                raise ValueError("--early-stop-patience must be positive")
            config["train"]["early_stop_patience"] = int(
                args.early_stop_patience
            )
        if args.per_gpu_batch_size is not None and args.per_gpu_batch_size <= 0:
            raise ValueError("--per-gpu-batch-size must be positive")
        if args.grad_accum_steps is not None and args.grad_accum_steps <= 0:
            raise ValueError("--grad-accum-steps must be positive")
        # Reference-cache identity includes the reference fitting batch settings.
        # Validate that immutable identity before applying student-only throughput
        # overrides below.
        reference_batch = int(config["train"]["per_gpu_batch_size"])
        reference_accumulation = int(config["train"]["grad_accum_steps"])
        positive_label, class_names = validate_dataset_semantics(config)
        monitor = str(config["train"].get("monitor", "macro_f1")).lower()
        if monitor != "macro_f1":
            raise ValueError("This protocol requires train.monitor=macro_f1 after validation threshold tuning")
        is_r1 = config["model"]["architecture_version"] == "qwen_lora_lgled_masked_r1"
        target_cache = None
        if is_r1:
            from mmfnd.r1_cache import student_source_fingerprint
            config["r1_student_source_fingerprint"] = student_source_fingerprint(root)
        if "replay_seed" in config["data"]:
            config["data"]["replay_seed"] = int(config["seed"])
        encoder_baseline = is_r1 and bool(config["model"].get("r1", {}).get("baseline"))
        if encoder_baseline:
            from mmfnd.r1_cache import backbone_fingerprint, digest
            _, backbone_hash = backbone_fingerprint(root, config)
            config["r1_backbone_identity"] = {"model_fingerprint": backbone_hash,
                "preprocess_fingerprint": digest(config["data"]), "source_fingerprint": "r1_encoder_baseline"}
        if is_r1 and (positive_label != 0 or float(config["train"]["label_smoothing"]) != 0):
            raise ValueError("R1 requires Fake=0 and label_smoothing=0")
        if is_r1 and not encoder_baseline:
            if args.r1_target_cache is None:
                raise ValueError("R1 requires an exact OOF target cache")
            config["data"]["replay_seed"] = int(config["seed"])
            from mmfnd.reference_pipeline import prepare_identity
            from mmfnd.r1_cache import TargetCache
            _, _, fingerprint = prepare_identity(root, config)
            target_cache = TargetCache(args.r1_target_cache, fingerprint)
            if target_cache.epsilon != float(config["loss"].get("epsilon",1e-7)):
                raise ValueError("Target normalization scale and student NLL epsilon mismatch")
            config["r1_backbone_identity"] = {k:fingerprint[k] for k in ("model_fingerprint","preprocess_fingerprint","source_fingerprint")}
            config["r1_reference_cache"] = str(args.r1_target_cache.resolve())
            config["r1_reference_fingerprint"] = fingerprint
        if args.per_gpu_batch_size is not None:
            config["train"]["per_gpu_batch_size"] = int(args.per_gpu_batch_size)
        if args.grad_accum_steps is not None:
            config["train"]["grad_accum_steps"] = int(args.grad_accum_steps)
        if args.per_gpu_batch_size is not None or args.grad_accum_steps is not None:
            reference_effective_batch = (
                reference_batch * reference_accumulation * context.world_size
            )
            student_effective_batch = (
                int(config["train"]["per_gpu_batch_size"])
                * int(config["train"]["grad_accum_steps"])
                * context.world_size
            )
            if student_effective_batch != reference_effective_batch:
                raise ValueError(
                    "Student batch overrides must preserve the reference effective "
                    f"batch size ({student_effective_batch} != "
                    f"{reference_effective_batch})"
                )
            config["student_batch_override"] = {
                "reference_per_gpu_batch_size": reference_batch,
                "reference_grad_accum_steps": reference_accumulation,
                "student_per_gpu_batch_size": int(config["train"]["per_gpu_batch_size"]),
                "student_grad_accum_steps": int(config["train"]["grad_accum_steps"]),
                "effective_batch_size": student_effective_batch,
            }
        seed_everything(int(config["seed"]) + (0 if is_r1 or config.get("protocol") == "r1_replay_pool_v1" else context.rank))
        precision = str(config["train"].get("precision", "bf16")).lower()
        if precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError("precision must be bf16, fp16 or fp32")
        if precision == "bf16" and context.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            precision = "fp16"

        if context.is_main:
            print(f"device={context.device}; world_size={context.world_size}; precision={precision}")
            print(f"dataset={config['dataset']['name']}; manifest_dir={manifest_dir}")
        processor = build_processor(root, config)
        train_loader = build_loader(
            root, config, "train", processor, shuffle=True,
            distributed_context=context,
        )
        val_loader = build_loader(root, config, "val", processor) if context.is_main else None
        model = build_model(config).to(context.device)
        if context.is_main and hasattr(model, "lgled"):
            qwen_model = model.encoder.shared_qwen_model()
            shared_layers = model.lgled.selected_qwen_layers(qwen_model)
            qwen_layers = list(qwen_model.layers)
            runtime = model.lgled.runtime(qwen_model)
            print(json.dumps({
                "architecture": model.architecture_version,
                "qwen_path": config["model"]["text_backbone"],
                "qwen_class": type(qwen_model).__name__,
                "qwen_hidden_size": int(qwen_model.config.hidden_size),
                "qwen_num_hidden_layers": int(qwen_model.config.num_hidden_layers),
                "latent_judge_layer_indices": [
                    runtime.first_shared_layer, runtime.last_shared_layer,
                ],
                "shared_layer_reference": all(
                    actual is expected for actual, expected in zip(
                        shared_layers,
                        qwen_layers[-model.lgled.latent_judge_num_layers:],
                    )
                ),
                "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
            }, ensure_ascii=False, indent=2))
        if context.distributed:
            model = DDP(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                output_device=context.local_rank if context.device.type == "cuda" else None,
                broadcast_buffers=False,
            )
        optimizer = build_optimizer(model, config)
        accumulation = int(config["train"]["grad_accum_steps"])
        steps_per_epoch = math.ceil(len(train_loader) / accumulation)
        total_steps = steps_per_epoch * int(config["train"]["epochs"])
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            int(total_steps * float(config["train"]["warmup_ratio"])),
            total_steps,
        )
        scaler = torch.amp.GradScaler(
            "cuda", enabled=precision == "fp16" and context.device.type == "cuda"
        )
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)

        # Reject old-protocol resumes BEFORE changing their saved logs/configs.
        resume_checkpoint = None
        if args.resume:
            resume_checkpoint = load_checkpoint(args.resume, model, context.device)
            checkpoint_threshold_selection(resume_checkpoint)
            if resume_checkpoint.get("optimizer_state_dict") is None or resume_checkpoint.get("scheduler_state_dict") is None:
                raise ValueError("--resume requires a resumable last.pth, not a model-only best/final checkpoint")

        output_root = resolve_path(root, config["train"]["output_dir"])
        if args.resume:
            output_dir = args.resume.resolve().parent.parent
        else:
            run_id = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = output_root / run_id
        checkpoint_dir = output_dir / "checkpoints"
        if context.is_main:
            checkpoint_dir.mkdir(parents=True, exist_ok=bool(args.resume))
            (checkpoint_dir / "topk").mkdir(parents=True, exist_ok=True)
            dump_json(config, output_dir / "config.json")
            dump_json({
                "started_at": datetime.now().astimezone().isoformat(),
                "world_size": context.world_size,
                "per_gpu_batch_size": int(config["train"]["per_gpu_batch_size"]),
                "effective_batch_size": int(config["train"]["per_gpu_batch_size"]) * context.world_size * accumulation,
                "dataset": config["dataset"]["name"], "manifest_dir": str(manifest_dir),
                "seed": int(config["seed"]), "precision": precision,
                "epochs": int(config["train"]["epochs"]),
                "early_stop_patience": int(
                    config["train"]["early_stop_patience"]
                ),
                "monitor": monitor,
                "evaluation_protocol": PROTOCOL,
                "threshold_source": "validation", "threshold_objective": "macro_f1",
                "evaluation_label_semantics": {str(k): v for k, v in class_names.items()},
                "positive_label": positive_label,
                "resume_from": str(args.resume) if args.resume else None,
            }, output_dir / "run_info.json")
        if context.distributed:
            dist.barrier()

        start_epoch, best_score, bad_epochs, global_step = 1, -float("inf"), 0, 0
        history: list[dict] = []
        topk: list[dict] = []
        if args.resume:
            checkpoint = resume_checkpoint
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            if scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
                scaler.load_state_dict(checkpoint["scaler_state_dict"])
            state = checkpoint.get("training_state", {})
            start_epoch = int(checkpoint["epoch"]) + 1
            saved_monitor = str(state.get("monitor", monitor))
            if saved_monitor != monitor:
                raise ValueError(
                    f"cannot resume with monitor={monitor!r}; checkpoint uses "
                    f"{saved_monitor!r}"
                )
            best_score = float(state.get(
                "best_score",
                state.get("best_auc", -float("inf")),
            ))
            bad_epochs = int(state.get("bad_epochs", 0))
            global_step = int(state.get("global_step", 0))
            history = list(state.get("history", []))
            topk = list(state.get("topk", []))

        if is_r1 and args.resume:
            from mmfnd.r1_runtime import restore_rng
            rng_states = resume_checkpoint.get("training_state", {}).get("rng_states")
            if not rng_states or len(rng_states) != context.world_size:
                raise ValueError("R1 resume requires matching per-rank RNG states/world size")
            restore_rng(rng_states[context.rank])
        optimizer.zero_grad(set_to_none=True)
        training_started = time.monotonic()
        observed_examples = 0
        stop_training = False
        last_epoch = start_epoch - 1
        loader_batch_size = int(
            getattr(train_loader, "batch_size", config["train"]["per_gpu_batch_size"])
        )
        loader_sampler = getattr(train_loader, "sampler", None)
        loader_dataset = getattr(train_loader, "dataset", None)
        loader_examples = (
            len(loader_sampler)
            if loader_sampler is not None
            else len(loader_dataset)
            if loader_dataset is not None
            else len(train_loader) * loader_batch_size
        )
        for epoch in range(start_epoch, int(config["train"]["epochs"]) + 1):
            last_epoch = epoch
            if hasattr(train_loader.sampler, "set_epoch"):
                train_loader.sampler.set_epoch(epoch)
            model.train()
            running = {"total": 0.0}
            batch_count = 0
            iterator = tqdm(train_loader, desc=f"train {epoch}", disable=not context.is_main)
            for step, raw_batch in enumerate(iterator, start=1):
                batch = move_batch(raw_batch, context.device)
                observed_examples += len(batch["labels"])
                window_start = ((step - 1) // accumulation) * accumulation + 1
                window_end = min(window_start + accumulation - 1, len(train_loader))
                window_examples = min(
                    (window_end - window_start + 1) * loader_batch_size,
                    loader_examples - (window_start - 1) * loader_batch_size,
                )
                if window_examples <= 0:
                    raise RuntimeError("invalid gradient accumulation window")
                update_now = step == window_end
                sync_context = model.no_sync() if context.distributed and not update_now else nullcontext()
                with sync_context:
                    with autocast_context(context.device, precision):
                        if is_r1:
                            from mmfnd.losses_masked_r1 import masked_r1_loss
                            latent_mask = batch["availability"].clone()
                            dropout = float(config["model"]["r1"].get("latent_view_dropout",0))
                            if dropout:
                                latent_mask &= torch.rand(latent_mask.shape,device=context.device) >= dropout
                                empty = ~latent_mask.any(-1)
                                latent_mask[empty,0] = True
                            outputs = model(batch, latent_view_mask=latent_mask)
                            if encoder_baseline:
                                loss = torch.nn.functional.cross_entropy(outputs["logits"].float(), batch["labels"])
                                components = {"classification": float(loss.detach())}
                            else:
                                q = target_cache.lookup(batch["ids"],batch["augmentation_ids"],batch["availability"],context.device)
                                loss,components = masked_r1_loss(outputs,batch["labels"],q,target_cache.scale,config["loss"],epoch)
                        else:
                            outputs = model(batch)
                            loss, components = multimodal_loss(
                                outputs, batch["labels"], config["loss"],
                                float(config["train"]["label_smoothing"]), positive_label,
                            )
                        # Losses are micro-batch means. Weight by the actual
                        # number of examples so a short final batch has the
                        # same per-example contribution as batch_size=1.
                        scaled_loss = loss * len(batch["labels"]) / window_examples
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"non-finite loss epoch={epoch} step={step}: {components}")
                    scaler.scale(scaled_loss).backward()
                if update_now:
                    scaler.unscale_(optimizer)
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), float(config["train"].get("gradient_clip_norm", 1.0))
                    )
                    if not torch.isfinite(gradient_norm):
                        raise FloatingPointError(f"non-finite gradient epoch={epoch} step={step}")
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                running["total"] += float(loss.detach())
                for name, value in components.items():
                    running[name] = running.get(name, 0.0) + value
                for name, value in lgled_training_diagnostics(outputs).items():
                    running[name] = running.get(name, 0.0) + value
                batch_count += 1
                if args.smoke_steps and global_step >= args.smoke_steps:
                    stop_training = True
                    break

            train_stats = reduce_training_stats(running, batch_count, context.device, context.distributed)
            rng_states = None
            if is_r1:
                from mmfnd.r1_runtime import capture_rng
                local_rng = capture_rng()
                if context.distributed:
                    rng_states = [None] * context.world_size
                    dist.all_gather_object(rng_states, local_rng)
                else:
                    rng_states = [local_rng]
            if args.smoke_steps:
                local_peak = (
                    torch.cuda.max_memory_allocated(context.device)
                    if context.device.type == "cuda" else 0
                )
                peak_tensor = torch.tensor(
                    [local_peak], device=context.device, dtype=torch.float64
                )
                if context.distributed:
                    gathered_peaks = [torch.zeros_like(peak_tensor) for _ in range(context.world_size)]
                    dist.all_gather(gathered_peaks, peak_tensor)
                    rank_peak_memory = [int(value.item()) for value in gathered_peaks]
                else:
                    rank_peak_memory = [int(local_peak)]
                if context.is_main:
                    smoke_path = checkpoint_dir / "smoke.pth"
                    save_checkpoint(smoke_path, model, optimizer, scheduler, epoch, train_stats, config, scaler=scaler)
                    load_checkpoint(smoke_path, model, context.device)
                    print(json.dumps({
                        "smoke_steps": global_step,
                        "train": train_stats,
                        "checkpoint": str(smoke_path),
                        "rank_peak_memory_bytes": rank_peak_memory,
                    }, ensure_ascii=False, indent=2))
                if context.distributed:
                    dist.barrier()
                return

            if context.distributed:
                dist.barrier()
            if context.is_main:
                val_metrics, _, selected_threshold = evaluate(
                    unwrap_model(model), val_loader, context.device,
                    positive_label, class_names, precision=precision, split="val", tune_threshold=True,
                    checkpoint_reference={"kind": "epoch", "epoch": epoch},
                )
                log_evaluation(val_metrics)
                selection_score = checkpoint_score(val_metrics, monitor)
                epoch_metrics = {
                    "epoch": epoch,
                    "train_total_loss": train_stats.pop("total"),
                    "train_loss_components": train_stats,
                    "learning_rates": {group.get("name", str(index)): group["lr"] for index, group in enumerate(optimizer.param_groups)},
                    "val": val_metrics,
                    "selection_metric": monitor,
                    "selection_score": selection_score,
                }
                history.append(epoch_metrics)
                if selection_score > best_score:
                    best_score, bad_epochs = selection_score, 0
                    save_checkpoint(
                        checkpoint_dir / "best.pth", model, None, None, epoch, val_metrics, config,
                        decision_threshold=selected_threshold,
                    )
                else:
                    bad_epochs += 1
                top_path = checkpoint_dir / "topk" / f"epoch_{epoch:02d}.pth"
                save_checkpoint(top_path, model, None, None, epoch, val_metrics, config,
                                decision_threshold=selected_threshold)
                topk.append({
                    "selection_score": selection_score,
                    "monitor": monitor,
                    "macro_f1": float(val_metrics["macro_f1"]),
                    "accuracy": float(val_metrics["accuracy"]),
                    "auc": float(val_metrics["auc"]),
                    "epoch": epoch,
                    "decision_threshold": selected_threshold,
                    "evaluation_protocol": PROTOCOL,
                    "path": str(top_path),
                })
                topk.sort(key=lambda item: (-item["selection_score"], item["epoch"]))
                keep = int(config["train"]["top_k_checkpoint_average"])
                for removed in topk[keep:]:
                    Path(removed["path"]).unlink(missing_ok=True)
                topk = topk[:keep]
                if is_r1:
                    from mmfnd.r1_runtime import capture_rng
                    rng_states[0] = capture_rng()
                state = {
                    "monitor": monitor, "best_score": best_score,
                    "best_val_macro_f1": best_score,
                    "bad_epochs": bad_epochs,
                    "global_step": global_step, "history": history, "topk": topk,
                    **({"rng_states": rng_states} if is_r1 else {}),
                }
                save_checkpoint(
                    checkpoint_dir / "last.pth", model, optimizer, scheduler,
                    epoch, val_metrics, config, decision_threshold=selected_threshold,
                    scaler=scaler, training_state=state,
                )
                dump_json(history, output_dir / "history.json")
                print(json.dumps(epoch_metrics, ensure_ascii=False, indent=2))
                stop_training = bad_epochs >= int(config["train"]["early_stop_patience"])
            if context.distributed:
                stop_tensor = torch.tensor([int(stop_training)], device=context.device)
                dist.broadcast(stop_tensor, src=0)
                stop_training = bool(stop_tensor.item())
                dist.barrier()
            if stop_training:
                if context.is_main:
                    print("early stopping")
                break

        if context.distributed:
            dist.barrier()
        if context.is_main:
            top_paths = [Path(item["path"]) for item in topk]
            average_checkpoints(top_paths, model, context.device)
            final_path = checkpoint_dir / "final_averaged.pth"
            # Averaging creates a NEW model. Tune on its own validation outputs,
            # never reuse an epoch's threshold or average thresholds numerically.
            val_metrics, val_rows, selected_threshold = evaluate(
                unwrap_model(model), val_loader, context.device, positive_label,
                class_names, precision=precision, split="val", tune_threshold=True,
                checkpoint_reference={"kind": "top_k_average", "epochs": [item["epoch"] for item in topk]},
            )
            log_evaluation(val_metrics)
            save_checkpoint(
                final_path, model, None, None, last_epoch, val_metrics, config,
                decision_threshold=selected_threshold,
                training_state={"averaged_checkpoints": topk},
            )
            val_dir = output_dir / "evaluation" / "final" / "val"
            dump_json(val_metrics, val_dir / "metrics.json")
            dump_json(evaluation_info(val_metrics), val_dir / "evaluation_info.json")
            write_jsonl(val_rows, val_dir / "predictions.jsonl")

            if (is_r1 or config.get("protocol") == "r1_replay_pool_v1") and not args.final_test_frozen:
                from mmfnd.r1_cache import file_sha
                summary = {"checkpoint": str(final_path), "checkpoint_sha256": file_sha(final_path),
                           "decision_threshold": selected_threshold, "val": val_metrics,
                           "test": "NOT RUN: explicit --final-test-frozen required", "topk": topk,
                           "final_model_kind": "top_k_average",
                           "resource_cost": {"training_seconds": time.monotonic()-training_started,
                               "observed_examples_with_sampler_padding": observed_examples*context.world_size,
                               "examples_per_second": observed_examples*context.world_size/max(1e-6,time.monotonic()-training_started),
                               "peak_cuda_bytes_rank0": torch.cuda.max_memory_allocated(context.device) if context.device.type == "cuda" else 0,
                               "total_parameters": sum(p.numel() for p in unwrap_model(model).parameters()),
                               "trainable_parameters": sum(p.numel() for p in unwrap_model(model).parameters() if p.requires_grad),
                               "effective_batch_size": int(config["train"]["per_gpu_batch_size"])*accumulation*context.world_size,
                               "world_size": context.world_size, "reference_cost_manifest": str(args.r1_target_cache.parent / "reference_manifest.json") if target_cache is not None else None}}
                dump_json(summary, output_dir / "final_summary.json")
                print(f"completed_run={output_dir}; test=NOT_RUN", flush=True)
            else:
                # Test is exposed exactly once, after the final model and threshold are frozen.
                final_checkpoint = load_checkpoint(final_path, model, context.device)
                frozen_selection = checkpoint_threshold_selection(final_checkpoint)
                test_loader = build_loader(root, config, "test", processor)
                test_metrics, test_rows, _ = evaluate(
                    unwrap_model(model), test_loader, context.device, positive_label,
                    class_names, selected_threshold, precision,
                    split="test", threshold_selection=frozen_selection,
                )
                log_evaluation(test_metrics)
                test_dir = output_dir / "evaluation" / "final" / "test"
                dump_json(test_metrics, test_dir / "metrics.json")
                dump_json(evaluation_info(test_metrics), test_dir / "evaluation_info.json")
                write_jsonl(test_rows, test_dir / "predictions.jsonl")
                summary = {
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "checkpoint": str(final_path), "decision_threshold": selected_threshold,
                    **evaluation_info(test_metrics),
                    "best_checkpoint": str(checkpoint_dir / "best.pth"),
                    "best_val_macro_f1": best_score, "best_epoch": topk[0]["epoch"],
                    "test_macro_f1": test_metrics["macro_f1"],
                    "final_model_kind": "top_k_average", "topk": topk,
                    "val_calibrated": val_metrics, "test": test_metrics,
                }
                dump_json(summary, output_dir / "final_summary.json")
                print(json.dumps(summary, ensure_ascii=False, indent=2))
                print(f"completed_run={output_dir}")
        if context.distributed:
            dist.barrier()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
