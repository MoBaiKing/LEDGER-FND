from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from dataset_tools.common import (
    SPLITS,
    require_columns,
    verify_image,
    write_workspace,
)
from mmfnd.utils import dump_json, resolve_path


SOURCE_FILES = {
    "train": "multimodal_only_samples/multimodal_train.tsv",
    "val": "multimodal_only_samples/multimodal_validate.tsv",
    "test": "multimodal_only_samples/multimodal_test_public.tsv",
}
SOURCE_PROTOCOL = "fakeddit_v2_public_multimodal_only"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".gif")


def _clean_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value or "")
    return re.sub(r"\s+", " ", normalized).strip()


def _valid_http_url(value: str) -> bool:
    if not value:
        return False
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _fingerprint(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def prepare_metadata(
    project_root: Path,
    config: dict,
    verify_decode: bool = False,
) -> Path:
    del verify_decode
    annotations_root = resolve_path(project_root, config["data"]["root"])
    output_dir = (
        project_root / "workspaces" / config["dataset"]["name"] / "metadata"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    global_ids: dict[str, str] = {}
    url_first_split: dict[str, str] = {}
    text_first_split: dict[str, str] = {}
    cross_split_duplicate_ids = 0
    cross_split_duplicate_urls = 0
    cross_split_duplicate_texts = 0
    split_stats: dict[str, dict] = {}

    for split in SPLITS:
        source = annotations_root / SOURCE_FILES[split]
        metadata_path = output_dir / f"{split}.metadata.jsonl.gz"
        queue_path = output_dir / f"{split}.download_queue.jsonl.gz"
        stats = Counter()
        labels = Counter()
        subreddits = Counter()
        seen_ids: set[str] = set()
        seen_urls: set[str] = set()
        seen_texts: set[str] = set()

        with source.open(
            "r", encoding="utf-8", newline=""
        ) as input_file, gzip.open(
            metadata_path, "wt", encoding="utf-8"
        ) as metadata_file, gzip.open(
            queue_path, "wt", encoding="utf-8"
        ) as queue_file:
            reader = csv.DictReader(input_file, delimiter="\t")
            require_columns(
                set(reader.fieldnames or []),
                {
                    "id",
                    "clean_title",
                    "title",
                    "image_url",
                    "hasImage",
                    "2_way_label",
                },
                source,
            )
            for source_row, row in enumerate(reader, start=2):
                stats["source_rows"] += 1
                sample_id = _clean_text(str(row.get("id") or ""))
                text = _clean_text(
                    str(row.get("clean_title") or "")
                    or str(row.get("title") or "")
                )
                label_value = _clean_text(str(row.get("2_way_label") or ""))
                image_url = _clean_text(str(row.get("image_url") or ""))
                has_image = (
                    _clean_text(str(row.get("hasImage") or "")).lower()
                    == "true"
                )

                if not sample_id:
                    stats["dropped_empty_id"] += 1
                    continue
                if not text:
                    stats["dropped_empty_text"] += 1
                    continue
                if label_value not in {"0", "1"}:
                    stats["dropped_invalid_label"] += 1
                    continue

                valid_url = _valid_http_url(image_url)
                if sample_id in seen_ids:
                    stats["duplicate_id_within_split"] += 1
                    continue
                seen_ids.add(sample_id)
                previous_id_split = global_ids.setdefault(sample_id, split)
                if previous_id_split != split:
                    cross_split_duplicate_ids += 1

                text_hash = _fingerprint(text.lower())
                if text_hash in seen_texts:
                    stats["duplicate_text_within_split"] += 1
                else:
                    seen_texts.add(text_hash)
                previous_text_split = text_first_split.setdefault(
                    text_hash, split
                )
                if previous_text_split != split:
                    cross_split_duplicate_texts += 1

                if valid_url:
                    url_hash = _fingerprint(image_url)
                    if url_hash in seen_urls:
                        stats["duplicate_url_within_split"] += 1
                    else:
                        seen_urls.add(url_hash)
                    previous_url_split = url_first_split.setdefault(
                        url_hash, split
                    )
                    if previous_url_split != split:
                        cross_split_duplicate_urls += 1

                label = int(label_value)
                labels[label] += 1
                subreddit = _clean_text(str(row.get("subreddit") or ""))
                subreddits[subreddit] += 1
                downloadable = has_image and valid_url
                if not image_url:
                    stats["missing_image_url"] += 1
                elif not valid_url:
                    stats["invalid_image_url"] += 1
                if downloadable:
                    stats["downloadable"] += 1

                record = {
                    "dataset": "fakeddit",
                    "id": f"{split}-{sample_id}",
                    "source_id": sample_id,
                    "split": split,
                    "source_row": source_row,
                    "label": label,
                    "label_name": "fake" if label == 0 else "real",
                    "text": text,
                    "title": _clean_text(str(row.get("title") or "")),
                    "image_url": image_url,
                    "image_key": sample_id,
                    "has_image_flag": has_image,
                    "downloadable": downloadable,
                    "category": "general",
                    "subreddit": subreddit,
                    "original_domain": _clean_text(
                        str(row.get("domain") or "")
                    ),
                    "created_utc": _clean_text(
                        str(row.get("created_utc") or "")
                    ),
                    "source_file": source.name,
                }
                metadata_file.write(
                    json.dumps(record, ensure_ascii=False) + "\n"
                )
                if downloadable:
                    queue_file.write(json.dumps({
                        "dataset": "fakeddit",
                        "split": split,
                        "id": record["id"],
                        "source_id": sample_id,
                        "url": image_url,
                        "target_stem": sample_id,
                        "label": label,
                    }, ensure_ascii=False) + "\n")
                stats["kept_rows"] += 1

        split_stats[split] = {
            **dict(stats),
            "labels": {str(key): value for key, value in labels.items()},
            "unique_ids": len(seen_ids),
            "unique_valid_urls": len(seen_urls),
            "unique_texts": len(seen_texts),
            "top_subreddits": subreddits.most_common(20),
            "metadata_file": str(metadata_path.resolve()),
            "download_queue": str(queue_path.resolve()),
        }

    report = {
        "dataset": "fakeddit",
        "source_protocol": SOURCE_PROTOCOL,
        "stage": "metadata_cleaned",
        "train_ready": False,
        "train_blocker": (
            "本地图片目录为空。需要根据下载队列准备图片，然后重新运行"
            "训练清单清洗。"
        ),
        "label_contract": {
            "0": "fake",
            "1": "real",
        },
        "annotations_root": str(annotations_root.resolve()),
        "local_image_root": str(
            resolve_path(project_root, config["data"]["image_root"]).resolve()
        ),
        "splits": split_stats,
        "cross_split_duplicate_ids": cross_split_duplicate_ids,
        "cross_split_duplicate_urls": cross_split_duplicate_urls,
        "cross_split_duplicate_texts": cross_split_duplicate_texts,
    }
    dump_json(report, output_dir / "metadata_audit_report.json")
    dump_json({
        "dataset": "fakeddit",
        "source_protocol": SOURCE_PROTOCOL,
        "stage": "metadata_cleaned",
        "train_ready": False,
        "schema_version": "fakeddit_metadata_v1",
        "output_dir": str(output_dir.resolve()),
        "splits": {
            split: {
                "kept_rows": split_stats[split]["kept_rows"],
                "downloadable": split_stats[split]["downloadable"],
                "labels": split_stats[split]["labels"],
            }
            for split in SPLITS
        },
    }, output_dir / "metadata_manifest.json")
    return output_dir


def _index_images(image_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in image_root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index.setdefault(path.stem, path)
    return index


def prepare(project_root: Path, config: dict, verify_decode: bool) -> Path:
    annotations_root = resolve_path(project_root, config["data"]["root"])
    image_root = resolve_path(project_root, config["data"]["image_root"])
    image_index = _index_images(image_root)
    if not image_index:
        raise FileNotFoundError(
            f"Fakeddit 本地图片目录为空: {image_root}。当前只有 TSV URL，"
            "请先下载官方图片；清洗阶段不会联网抓取，避免不可复现数据。"
        )

    records_by_split: dict[str, list[dict]] = {}
    missing_images: list[dict] = []
    for split, filename in SOURCE_FILES.items():
        source = annotations_root / filename
        with source.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter="\t")
            require_columns(
                set(reader.fieldnames or []),
                {"id", "clean_title", "title", "2_way_label"},
                source,
            )
            records = []
            for row_number, row in enumerate(reader, start=2):
                source_id = str(row.get("id") or "").strip()
                linked_submission_id = str(
                    row.get("linked_submission_id") or ""
                ).strip()
                candidates = [source_id, linked_submission_id]
                image_path = next(
                    (image_index[key] for key in candidates if key in image_index),
                    None,
                )
                if image_path is None:
                    missing_images.append({
                        "split": split,
                        "row": row_number,
                        "id": row.get("id"),
                    })
                    continue
                relative = image_path.relative_to(image_root).as_posix()
                record = {
                    "id": f"{split}-{source_id}",
                    "source_id": source_id,
                    "label": int(row["2_way_label"]),
                    "text": (
                        str(row.get("clean_title") or "").strip()
                        or str(row.get("title") or "").strip()
                    ),
                    "images": [relative],
                    "image_sha256": [
                        verify_image(image_path, verify_decode)
                    ],
                    "category": "general",
                    "title": str(row.get("title") or "").strip(),
                    "source": str(row.get("subreddit") or "").strip(),
                }
                if linked_submission_id:
                    record["linked_submission_id"] = linked_submission_id
                records.append(record)
        records_by_split[split] = records

    if any(not rows for rows in records_by_split.values()):
        raise RuntimeError(
            "Fakeddit 至少一个分区没有可用的本地图片匹配结果；"
            f"未匹配记录数={len(missing_images)}。"
        )
    report = {
        "adapter": "dataset_tools.fakeddit.prepare",
        "source_protocol": SOURCE_PROTOCOL,
        "annotations_root": str(annotations_root.resolve()),
        "image_root": str(image_root.resolve()),
        "missing_local_images": len(missing_images),
        "missing_examples": missing_images[:100],
    }
    return write_workspace(project_root, config, records_by_split, report)
