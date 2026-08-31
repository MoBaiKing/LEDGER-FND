from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import pickle
from urllib.parse import unquote, urlparse

from PIL import Image

from dataset_tools.common import (
    LABEL_SEMANTICS,
    SCHEMA_VERSION,
    normalize_text,
    sha256,
    text_sha256,
)
from mmfnd.utils import dump_json, resolve_path


EANN_COMMIT = "dc1faf30368ab1db0752df60b3c0f5c54bcb7fa1"
SPLIT_PICKLES = {
    "train": "train_id.pickle",
    "val": "validate_id.pickle",
    "test": "test_id.pickle",
}
RAW_FILES = (
    ("train_nonrumor.txt", "train", 1),
    ("train_rumor.txt", "train", 0),
    ("test_nonrumor.txt", "test", 1),
    ("test_rumor.txt", "test", 0),
)


def _load_eann_splits(directory: Path) -> tuple[dict[str, tuple[str, int]], dict]:
    assignments: dict[str, tuple[str, int]] = {}
    hashes = {}
    for split, filename in SPLIT_PICKLES.items():
        path = directory / filename
        if not path.is_file():
            raise FileNotFoundError(f"缺少 EANN 官方划分文件: {path}")
        hashes[filename] = sha256(path)
        with path.open("rb") as file:
            values = pickle.load(file, encoding="latin1")
        for raw_id, raw_event in values.items():
            post_id = str(raw_id)
            if post_id in assignments:
                raise ValueError(f"EANN 官方划分中 ID 重复: {post_id}")
            assignments[post_id] = (split, int(raw_event))
    return assignments, hashes


def _image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for directory in ("rumor_images", "nonrumor_images"):
        image_dir = root / directory
        if not image_dir.is_dir():
            raise FileNotFoundError(f"缺少图片目录: {image_dir}")
        for path in sorted(image_dir.iterdir()):
            if not path.is_file() or path.stat().st_size == 0:
                continue
            key = path.stem.casefold()
            if key in index:
                raise ValueError(
                    f"图片主文件名冲突: {index[key].name!r} / {path.name!r}"
                )
            index[key] = path
    return index


def _read_records(
    root: Path,
    assignments: dict[str, tuple[str, int]],
    verify_decode: bool,
) -> tuple[dict[str, list[dict]], dict]:
    image_index = _image_index(root)
    image_hashes: dict[Path, str] = {}
    records_by_split = {split: [] for split in SPLIT_PICKLES}
    seen_ids = set()
    raw_counts = Counter()
    selected_source_counts = Counter()
    selected_image_counts = Counter()

    for filename, official_source_split, label in RAW_FILES:
        path = root / filename
        lines = path.read_text(encoding="utf-8-sig").splitlines()
        if len(lines) % 3:
            raise ValueError(f"{path} 不是严格的三行一条记录: lines={len(lines)}")
        for offset in range(0, len(lines), 3):
            metadata, image_line, raw_text = lines[offset:offset + 3]
            fields = metadata.split("|")
            if len(fields) != 15:
                raise ValueError(
                    f"{filename}:{offset + 1} 元数据字段不是 15 个: {len(fields)}"
                )
            post_id = fields[0].lstrip("\ufeff").strip()
            if not post_id or post_id in seen_ids:
                raise ValueError(f"空或重复微博 ID: {post_id!r}")
            seen_ids.add(post_id)
            raw_counts[(official_source_split, label)] += 1
            if post_id not in assignments:
                continue

            split, event = assignments[post_id]
            text = normalize_text(raw_text)
            if not text:
                raise ValueError(f"EANN 官方划分选中了空正文: {post_id}")

            selected_images = []
            seen_images = set()
            for value in image_line.split("|"):
                if not value or value.casefold() == "null":
                    continue
                filename_from_url = Path(
                    unquote(urlparse(value).path)
                ).name
                selected_image = image_index.get(
                    Path(filename_from_url).stem.casefold()
                )
                if (
                    selected_image is not None
                    and selected_image not in seen_images
                ):
                    selected_images.append(selected_image)
                    seen_images.add(selected_image)
            if not selected_images:
                raise ValueError(f"EANN 官方划分样本没有本地有效图片: {post_id}")
            for selected_image in selected_images:
                if selected_image not in image_hashes:
                    if verify_decode:
                        with Image.open(selected_image) as image:
                            image.verify()
                    image_hashes[selected_image] = sha256(selected_image)

            relative_images = [
                selected_image.relative_to(root).as_posix()
                for selected_image in selected_images
            ]
            records_by_split[split].append({
                "id": f"weibo-{post_id}",
                "post_id": post_id,
                "label": label,
                "label_name": LABEL_SEMANTICS[str(label)],
                "text": text,
                "text_sha256": text_sha256(text),
                "images": relative_images,
                "image_sha256": [
                    image_hashes[selected_image]
                    for selected_image in selected_images
                ],
                "event": str(event),
                "category": f"event_{event}",
                "official_source_split": official_source_split,
                "source_file": filename,
                "source_record": offset // 3 + 1,
                "folder_label_consistent": all(
                    selected_image.parent.name == (
                        "rumor_images" if label == 0 else "nonrumor_images"
                    )
                    for selected_image in selected_images
                ),
            })
            selected_source_counts[(official_source_split, label)] += 1
            selected_image_counts[len(selected_images)] += 1

    missing = sorted(set(assignments) - seen_ids)
    if missing:
        raise ValueError(
            f"原始数据缺少 {len(missing)} 个 EANN 官方划分 ID，例如 {missing[:5]}"
        )
    for split in records_by_split:
        records_by_split[split].sort(key=lambda row: row["post_id"])
    report = {
        "adapter": "dataset_tools.weibo.prepare",
        "protocol": "EANN-KDD18 official ID/event split",
        "eann_commit": EANN_COMMIT,
        "raw_records": sum(raw_counts.values()),
        "selected_records": sum(len(rows) for rows in records_by_split.values()),
        "excluded_not_in_official_eann_split": len(seen_ids) - len(assignments),
        "raw_source_counts": {
            f"{split}_label_{label}": count
            for (split, label), count in sorted(raw_counts.items())
        },
        "selected_source_counts": {
            f"{split}_label_{label}": count
            for (split, label), count in sorted(selected_source_counts.items())
        },
        "image_policy": (
            "all locally available unique images in source URL order"
        ),
        "selected_image_references": sum(
            count * image_count
            for image_count, count in selected_image_counts.items()
        ),
        "multi_image_records": sum(
            count
            for image_count, count in selected_image_counts.items()
            if image_count > 1
        ),
        "max_images_per_record": max(selected_image_counts, default=0),
        "image_count_distribution": {
            str(image_count): count
            for image_count, count in sorted(selected_image_counts.items())
        },
        "text_policy": "HTML unescape + Unicode NFKC + zero-width removal + whitespace collapse",
        "label_policy": "rumor=0=fake; nonrumor=1=real",
        "note": (
            "直接使用 EANN 官方 train/validate/test ID 与 event 映射；"
            "未重新随机划分，也未把同一微博按多张图片拆成多条样本。"
        ),
        "unique_selected_images": len(image_hashes),
    }
    return records_by_split, report


def prepare(project_root: Path, config: dict, verify_decode: bool) -> Path:
    root = resolve_path(project_root, config["data"]["root"]).resolve()
    output = resolve_path(project_root, config["data"]["processed_dir"]).resolve()
    split_dir = resolve_path(
        project_root, config["data"]["eann_split_dir"]
    ).resolve()
    assignments, split_hashes = _load_eann_splits(split_dir)
    records_by_split, report = _read_records(root, assignments, verify_decode)
    report["eann_split_sha256"] = split_hashes

    output.mkdir(parents=True, exist_ok=True)
    split_stats = {}
    for split, records in records_by_split.items():
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as file:
            for record in records:
                record["dataset"] = "weibo"
                record["split"] = split
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        labels = Counter(row["label"] for row in records)
        split_stats[split] = {
            "samples": len(records),
            "labels": {str(key): labels[key] for key in sorted(labels)},
            "image_references": sum(len(row["images"]) for row in records),
        }

    manifest = {
        "dataset": "weibo",
        "schema_version": SCHEMA_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "processed_dir": ".",
        "image_root": str(root),
        "runtime_image_preprocessing": config["data"].get(
            "image_preprocessing", {}
        ),
        "splits": split_stats,
        "source": report,
    }
    dump_json(manifest, output / "dataset_manifest.json")
    dump_json(report, output / "preprocess_report.json")
    return output
