#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime
import json
import math
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
    select_robust_threshold, unwrap_model, write_jsonl,
)
from mmfnd.factory import build_loader, build_processor
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
    if monitor not in {"macro_f1", "accuracy", "auc"}:
        raise ValueError("train.monitor must be macro_f1, accuracy or auc")
    value = metrics.get(monitor)
    if value is None:
        raise ValueError(f"validation metric {monitor!r} is unavailable")
    return float(value)


def average_checkpoints(paths: list[Path], model, device) -> None:
    if not paths:
        raise RuntimeError("no validation checkpoints available for averaging")
    checkpoints = [torch.load(path, map_location="cpu", weights_only=False) for path in paths]
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest-dir", required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--early-stop-patience", type=int)
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
        positive_label, class_names = validate_dataset_semantics(config)
        monitor = str(config["train"].get("monitor", "macro_f1")).lower()
        if monitor not in {"macro_f1", "accuracy", "auc"}:
            raise ValueError("train.monitor must be macro_f1, accuracy or auc")
        seed_everything(int(config["seed"]) + context.rank)
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
        model = ExplainableMMFND(config).to(context.device)
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
                "resume_from": str(args.resume) if args.resume else None,
            }, output_dir / "run_info.json")
        if context.distributed:
            dist.barrier()

        start_epoch, best_score, bad_epochs, global_step = 1, -float("inf"), 0, 0
        history: list[dict] = []
        topk: list[dict] = []
        if args.resume:
            checkpoint = load_checkpoint(args.resume, model, context.device)
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

        optimizer.zero_grad(set_to_none=True)
        stop_training = False
        last_epoch = start_epoch - 1
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
                window_start = ((step - 1) // accumulation) * accumulation + 1
                window_end = min(window_start + accumulation - 1, len(train_loader))
                window_size = window_end - window_start + 1
                update_now = step == window_end
                sync_context = model.no_sync() if context.distributed and not update_now else nullcontext()
                with sync_context:
                    with autocast_context(context.device, precision):
                        outputs = model(batch)
                        loss, components = multimodal_loss(
                            outputs, batch["labels"], config["loss"],
                            float(config["train"]["label_smoothing"]), positive_label,
                        )
                        scaled_loss = loss / window_size
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
                batch_count += 1
                if args.smoke_steps and global_step >= args.smoke_steps:
                    stop_training = True
                    break

            train_stats = reduce_training_stats(running, batch_count, context.device, context.distributed)
            if args.smoke_steps:
                if context.is_main:
                    smoke_path = checkpoint_dir / "smoke.pth"
                    save_checkpoint(smoke_path, model, optimizer, scheduler, epoch, train_stats, config, scaler=scaler)
                    load_checkpoint(smoke_path, model, context.device)
                    print(json.dumps({"smoke_steps": global_step, "train": train_stats, "checkpoint": str(smoke_path)}, ensure_ascii=False, indent=2))
                if context.distributed:
                    dist.barrier()
                return

            if context.distributed:
                dist.barrier()
            if context.is_main:
                val_metrics, _, _ = evaluate(
                    unwrap_model(model), val_loader, context.device,
                    positive_label, class_names, decision_threshold=0.5,
                    precision=precision,
                )
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
                else:
                    bad_epochs += 1
                top_path = checkpoint_dir / "topk" / f"epoch_{epoch:02d}.pth"
                save_checkpoint(top_path, model, None, None, epoch, val_metrics, config)
                topk.append({
                    "selection_score": selection_score,
                    "monitor": monitor,
                    "macro_f1": float(val_metrics["macro_f1"]),
                    "accuracy": float(val_metrics["accuracy"]),
                    "auc": float(val_metrics["auc"]),
                    "epoch": epoch,
                    "path": str(top_path),
                })
                topk.sort(key=lambda item: (-item["selection_score"], item["epoch"]))
                keep = int(config["train"]["top_k_checkpoint_average"])
                for removed in topk[keep:]:
                    Path(removed["path"]).unlink(missing_ok=True)
                topk = topk[:keep]
                state = {
                    "monitor": monitor, "best_score": best_score,
                    "bad_epochs": bad_epochs,
                    "global_step": global_step, "history": history, "topk": topk,
                }
                save_checkpoint(
                    checkpoint_dir / "last.pth", model, optimizer, scheduler,
                    epoch, val_metrics, config, scaler=scaler, training_state=state,
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
            save_checkpoint(
                final_path, model, None, None, last_epoch,
                {"averaged_epochs": [item["epoch"] for item in topk]}, config,
            )
            val_metrics_05, val_rows_05, _ = evaluate(
                unwrap_model(model), val_loader, context.device, positive_label,
                class_names, 0.5, precision,
            )
            selected_threshold, maximum_f1 = select_robust_threshold(
                [row["label"] for row in val_rows_05],
                [row["positive_probability"] for row in val_rows_05],
                positive_label,
                float(config["train"]["threshold_min"]),
                float(config["train"]["threshold_max"]),
                float(config["train"]["threshold_step"]),
                float(config["train"]["threshold_plateau_delta"]),
            )
            val_metrics, val_rows, _ = evaluate(
                unwrap_model(model), val_loader, context.device, positive_label,
                class_names, selected_threshold, precision,
            )
            val_metrics["calibration_max_positive_f1"] = maximum_f1
            save_checkpoint(
                final_path, model, None, None, last_epoch, val_metrics, config,
                decision_threshold=selected_threshold,
                training_state={"averaged_checkpoints": topk},
            )
            val_dir = output_dir / "evaluation" / "final" / "val"
            dump_json(val_metrics, val_dir / "metrics.json")
            dump_json({"decision_threshold": selected_threshold}, val_dir / "evaluation_info.json")
            write_jsonl(val_rows, val_dir / "predictions.jsonl")

            # Test is exposed exactly once, after the final model and threshold are frozen.
            test_loader = build_loader(root, config, "test", processor)
            test_metrics, test_rows, _ = evaluate(
                unwrap_model(model), test_loader, context.device, positive_label,
                class_names, selected_threshold, precision,
            )
            test_dir = output_dir / "evaluation" / "final" / "test"
            dump_json(test_metrics, test_dir / "metrics.json")
            dump_json({"decision_threshold": selected_threshold}, test_dir / "evaluation_info.json")
            write_jsonl(test_rows, test_dir / "predictions.jsonl")
            summary = {
                "completed_at": datetime.now().astimezone().isoformat(),
                "checkpoint": str(final_path), "decision_threshold": selected_threshold,
                "topk": topk, "val_at_0_5": val_metrics_05,
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
