from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from mmfnd.image_preprocessing import load_preprocessed_image


class CUTEFNDMultimodalDataset(Dataset):
    def __init__(self, manifest: Path, data_root: Path, processor, config: dict, train: bool = False):
        self.records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line]
        self.data_root = data_root
        self.processor = processor
        self.max_images = int(config["max_images"])
        self.max_text_length = int(config["max_text_length"])
        self.text_instruction = str(config.get("text_instruction", ""))
        self.image_preprocessing = dict(config.get("image_preprocessing", {}))
        self.train = train
        self.dataset_name = str(config["dataset_name"]).lower()
        wrong_dataset = sorted({
            str(record.get("dataset", "")).lower()
            for record in self.records
            if str(record.get("dataset", "")).lower() != self.dataset_name
        })
        if wrong_dataset:
            raise ValueError(
                f"清洗清单包含非 {self.dataset_name} 数据: {wrong_dataset}"
            )
        invalid_labels = sorted({
            record.get("label") for record in self.records
            if record.get("label") not in (0, 1)
        }, key=str)
        if invalid_labels:
            raise ValueError(f"清洗清单包含非二分类标签: {invalid_labels}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        paths = record["images"][: self.max_images]
        images = []
        for relative in paths:
            images.append(load_preprocessed_image(
                self.data_root / relative,
                self.image_preprocessing,
                train=self.train,
            ))
        if self.train and len(images) > 1:
            paired = list(zip(paths, images))
            random.shuffle(paired)
            paths, images = map(list, zip(*paired))
        return {
            "id": record["id"],
            "text": record["text"],
            "images": images,
            "image_paths": paths,
            "label": int(record["label"]),
            "category": record.get("category", ""),
        }

    def collate_fn(self, batch: list[dict]) -> dict:
        prompted_texts = [
            f"{self.text_instruction}{item['text']}" for item in batch
        ]
        text_tokens = self.processor.tokenizer(
            prompted_texts, padding=True, truncation=True,
            max_length=self.max_text_length, return_tensors="pt",
        )
        flat_images, owners = [], []
        for owner, item in enumerate(batch):
            for image in item["images"]:
                flat_images.append(image)
                owners.append(owner)
        # Every image returned by load_preprocessed_image is an RGB PIL image,
        # which becomes an HWC array inside the Hugging Face processor.  Make
        # that layout explicit: for pathological 1-pixel images, automatic
        # inference otherwise mistakes the height/width dimension for the
        # channel dimension and Pillow receives an invalid (1, 1, width) array.
        vision = self.processor.image_processor(
            images=flat_images,
            return_tensors="pt",
            input_data_format="channels_last",
        )
        return {
            "ids": [item["id"] for item in batch],
            "text_input_ids": text_tokens["input_ids"],
            "text_attention_mask": text_tokens["attention_mask"],
            "pixel_values": vision["pixel_values"],
            "image_owner": torch.tensor(owners, dtype=torch.long),
            "labels": torch.tensor([item["label"] for item in batch], dtype=torch.long),
            "image_paths": [item["image_paths"] for item in batch],
            "texts": [item["text"] for item in batch],
            "categories": [item["category"] for item in batch],
        }


def move_batch(batch: dict, device: torch.device) -> dict:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


# Backward-compatible import name for older auxiliary scripts.
WeiboMultimodalDataset = CUTEFNDMultimodalDataset
