from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from sklearn.model_selection import train_test_split

from dataset_tools.common import SPLITS, verify_image, write_workspace
from mmfnd.utils import resolve_path


def _fingerprint(values: list[str]) -> str:
    """Return a stable fingerprint for auditing a generated split."""
    payload = "\n".join(sorted(values)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _image_error(path: Path) -> str | None:
    if not path.is_file():
        return "missing"
    if path.stat().st_size == 0:
        return "empty"
    try:
        with Image.open(path) as image:
            image.verify()
    except (OSError, ValueError, UnidentifiedImageError):
        return "decode_failed"
    return None


def _label_counts(rows: list[dict]) -> dict[str, int]:
    return {
        str(label): count
        for label, count in sorted(
            Counter(int(row["label"]) for row in rows).items()
        )
    }


def prepare(project_root: Path, config: dict, verify_decode: bool) -> Path:
    """Prepare one overall FineFake binary task without domain partitioning.

    The cleaned Kaggle archive contains one ``FineFake.csv`` and all referenced
    images. We retain samples with non-empty text and a decodable image, then
    create a deterministic 60/20/20 split stratified by the binary label only.
    ``topic`` is never used for splitting or training and every sample receives
    the single category ``general``.
    """
    data_root = resolve_path(project_root, config["data"]["root"])
    image_root = resolve_path(project_root, config["data"]["image_root"])
    source = data_root / "FineFake.csv"
    if not source.is_file():
        raise FileNotFoundError(f"FineFake 原始标注不存在: {source}")

    frame = pd.read_csv(
        source,
        usecols=["text", "image_path", "label", "topic", "platform"],
    )
    required = {"text", "image_path", "label"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} 缺少字段: {sorted(missing)}")

    usable: list[dict] = []
    excluded = Counter()
    excluded_examples: list[dict] = []
    source_topics = Counter()

    for original_index, row in frame.iterrows():
        text = str(row.get("text") or "").strip()
        raw_image = str(row.get("image_path") or "").strip().replace("\\", "/")
        if not text:
            reason = "empty_text"
        elif not raw_image:
            reason = "empty_image_path"
        else:
            reason = _image_error(image_root / raw_image)

        if reason is not None:
            excluded[reason] += 1
            if len(excluded_examples) < 100:
                excluded_examples.append({
                    "source_index": int(original_index),
                    "image_path": raw_image,
                    "reason": reason,
                })
            continue

        label = int(row["label"])
        if label not in (0, 1):
            raise ValueError(
                f"{source} 第 {original_index} 行 label={label}，只支持二分类 0/1"
            )
        topic = str(row.get("topic") or "Uncategorized").strip()
        source_topics[topic] += 1
        sample_id = f"finefake-{int(original_index):06d}"
        usable.append({
            "id": sample_id,
            "label": label,
            "text": text,
            "images": [raw_image],
            "image_sha256": [verify_image(image_root / raw_image, False)],
            "category": "general",
            "title": "",
            "source": str(row.get("platform") or ""),
        })

    if len(usable) < 10:
        raise ValueError(f"FineFake 可训练样本过少: {len(usable)}")

    seed = int(config.get("seed", 200408))
    labels = [row["label"] for row in usable]
    train_rows, remainder = train_test_split(
        usable,
        test_size=0.4,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )
    val_rows, test_rows = train_test_split(
        remainder,
        test_size=0.5,
        random_state=seed,
        shuffle=True,
        stratify=[row["label"] for row in remainder],
    )
    records_by_split = {
        "train": train_rows,
        "val": val_rows,
        "test": test_rows,
    }

    split_ids = {
        split: {row["id"] for row in rows}
        for split, rows in records_by_split.items()
    }
    for left_index, left in enumerate(SPLITS):
        for right in SPLITS[left_index + 1:]:
            overlap = split_ids[left] & split_ids[right]
            if overlap:
                raise RuntimeError(
                    f"FineFake 分区泄漏: {left}/{right} 重复 {len(overlap)} 条"
                )

    split_images = {
        split: {image for row in rows for image in row["images"]}
        for split, rows in records_by_split.items()
    }
    cross_split_duplicate_images = sum(
        len(split_images[left] & split_images[right])
        for left_index, left in enumerate(SPLITS)
        for right in SPLITS[left_index + 1:]
    )

    report = {
        "adapter": "dataset_tools.finefake.prepare",
        "protocol": "finefake_overall_binary_valid_images_60_20_20",
        "dataset_url": "https://www.kaggle.com/datasets/abushahadatrabby/finefake-dataset/data",
        "source_file": str(source.resolve()),
        "image_root": str(image_root.resolve()),
        "seed": seed,
        "split_ratio": {"train": 0.6, "val": 0.2, "test": 0.2},
        "stratified_by": ["label"],
        "domain_partitioning": False,
        "category_value": "general",
        "topic_usage": "audit_only_not_model_input",
        "label_semantics": {"0": "fake", "1": "real"},
        "source_samples": int(len(frame)),
        "usable_samples": len(usable),
        "excluded_samples": int(sum(excluded.values())),
        "excluded_reasons": dict(sorted(excluded.items())),
        "excluded_examples_first_100": excluded_examples,
        "source_topic_distribution_usable": dict(sorted(source_topics.items())),
        "splits": {
            split: {
                "samples": len(rows),
                "labels": _label_counts(rows),
                "id_sha256": _fingerprint([row["id"] for row in rows]),
            }
            for split, rows in records_by_split.items()
        },
        "cross_split_duplicate_ids": 0,
        "cross_split_duplicate_images": cross_split_duplicate_images,
        "decode_validation": True,
        "requested_verify_decode": bool(verify_decode),
    }
    return write_workspace(project_root, config, records_by_split, report)
