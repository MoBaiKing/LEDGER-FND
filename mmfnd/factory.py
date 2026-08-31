from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import AutoImageProcessor, AutoTokenizer

from mmfnd.data import CUTEFNDMultimodalDataset
from mmfnd.utils import resolve_path


def build_processor(project_root: Path, config: dict):
    tokenizer = AutoTokenizer.from_pretrained(
        str(resolve_path(project_root, config["model"]["text_backbone"])),
        local_files_only=True,
    )
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return SimpleNamespace(
        tokenizer=tokenizer,
        image_processor=AutoImageProcessor.from_pretrained(
            str(resolve_path(project_root, config["model"]["vision_backbone"])),
            local_files_only=True,
        ),
    )


def build_loader(
    project_root: Path,
    config: dict,
    split: str,
    processor,
    shuffle: bool = False,
    sampler=None,
    distributed_context=None,
):
    image_root = resolve_path(
        project_root,
        config["data"].get("image_root", config["data"]["root"]),
    )
    processed = resolve_path(project_root, config["data"]["processed_dir"])
    dataset_config = {
        **config["data"],
        "dataset_name": config["dataset"]["name"],
    }
    dataset = CUTEFNDMultimodalDataset(
        processed / f"{split}.jsonl", image_root, processor,
        dataset_config, train=split == "train" and shuffle,
    )
    if distributed_context is not None and distributed_context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=distributed_context.world_size,
            rank=distributed_context.rank,
            shuffle=shuffle,
        )
    num_workers = int(config["data"]["num_workers"])
    loader_options = {}
    if num_workers > 0:
        loader_options.update(
            persistent_workers=bool(
                config["data"].get("persistent_workers", True)
            ),
            prefetch_factor=int(config["data"].get("prefetch_factor", 2)),
        )
    return DataLoader(
        dataset,
        batch_size=int(config["train"]["per_gpu_batch_size"]),
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=bool(config["data"].get("pin_memory", False)),
        collate_fn=dataset.collate_fn,
        **loader_options,
    )
