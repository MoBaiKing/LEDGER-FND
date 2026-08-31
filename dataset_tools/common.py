from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

from PIL import Image

from mmfnd.utils import dump_json, resolve_path


SCHEMA_VERSION = "cute_fnd_multimodal_v1"
SPLITS = ("train", "val", "test")
SPLIT_PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
AUDIT_EXAMPLE_LIMIT = 100
LABEL_SEMANTICS = {"0": "fake", "1": "real"}
_ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2060\ufeff]")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_image(path: Path, verify_decode: bool) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"找不到图片: {path}")
    if verify_decode:
        with Image.open(path) as image:
            image.verify()
    return sha256(path)


class ImageVerifier:
    def __init__(self, verify_decode: bool):
        self.verify_decode = verify_decode
        self._cache: dict[Path, str] = {}

    def __call__(self, path: Path) -> str:
        resolved = path.resolve()
        if resolved not in self._cache:
            self._cache[resolved] = verify_image(resolved, self.verify_decode)
        return self._cache[resolved]

    @property
    def unique_images(self) -> int:
        return len(self._cache)


def normalize_text(value: object) -> str:
    text = html.unescape("" if value is None else str(value))
    text = unicodedata.normalize("NFKC", text)
    text = _ZERO_WIDTH.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def safe_relative_path(value: object) -> str:
    normalized = str(value or "").strip().replace("\\", "/")
    path = Path(normalized)
    if not normalized or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"非法图片相对路径: {value!r}")
    return path.as_posix()


def text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalized_text(value: object) -> str:
    normalized = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _overlap(left: set[str], right: set[str]) -> dict:
    shared = left & right
    return {
        "count": len(shared),
        "examples": sorted(shared)[:AUDIT_EXAMPLE_LIMIT],
    }


def build_dataset_audit(
    dataset_name: str,
    records_by_split: dict[str, list[dict]],
) -> dict:
    id_fields = sorted({
        key
        for records in records_by_split.values()
        for record in records
        for key in record
        if key == "id" or key.endswith("_id")
    })
    image_hashes: dict[str, set[str]] = {}
    normalized_texts: dict[str, set[str]] = {}
    ids: dict[str, dict[str, set[str]]] = {
        field: {} for field in id_fields
    }
    for split in SPLITS:
        records = records_by_split.get(split, [])
        image_hashes[split] = {
            str(value).strip().lower()
            for record in records
            for value in (
                [record.get("image_sha256")]
                if isinstance(record.get("image_sha256"), str)
                else record.get("image_sha256", [])
            )
            if str(value or "").strip()
        }
        normalized_texts[split] = {
            normalized
            for record in records
            if (normalized := _normalized_text(record.get("text", "")))
        }
        for field in id_fields:
            ids[field][split] = {
                str(record[field]).strip()
                for record in records
                if record.get(field) is not None
                and str(record[field]).strip()
            }

    overlaps = {}
    for left, right in SPLIT_PAIRS:
        overlaps[f"{left}-{right}"] = {
            "image_sha256": _overlap(
                image_hashes[left], image_hashes[right]
            ),
            "normalized_text": _overlap(
                normalized_texts[left], normalized_texts[right]
            ),
            "id_fields": {
                field: _overlap(ids[field][left], ids[field][right])
                for field in id_fields
            },
        }
    return {
        "dataset": dataset_name,
        "schema_version": "cute_fnd_dataset_audit_v1",
        "split_policy": "report_only_official_splits_unchanged",
        "text_normalization": "NFKC + whitespace collapse + casefold",
        "example_limit_per_overlap": AUDIT_EXAMPLE_LIMIT,
        "id_fields": id_fields,
        "overlaps": overlaps,
    }


def write_workspace(
    project_root: Path,
    config: dict,
    records_by_split: dict[str, list[dict]],
    report: dict,
) -> Path:
    dataset_name = str(config["dataset"]["name"]).lower()
    output_dir = resolve_path(project_root, config["data"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    split_stats = {}
    for split in SPLITS:
        records = records_by_split.get(split, [])
        destination = output_dir / f"{split}.jsonl"
        with destination.open("w", encoding="utf-8") as file:
            for record in records:
                record["dataset"] = dataset_name
                record["split"] = split
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        split_stats[split] = {
            "samples": len(records),
            "labels": dict(Counter(int(row["label"]) for row in records)),
            "image_references": sum(len(row["images"]) for row in records),
        }

    manifest = {
        "dataset": dataset_name,
        "schema_version": SCHEMA_VERSION,
        "label_semantics": LABEL_SEMANTICS,
        "processed_dir": str(output_dir.resolve()),
        "image_root": str(
            resolve_path(project_root, config["data"]["image_root"]).resolve()
        ),
        "runtime_image_preprocessing": config["data"].get(
            "image_preprocessing", {}
        ),
        "splits": split_stats,
        "source": report,
    }
    dump_json(manifest, output_dir / "dataset_manifest.json")
    dump_json(report, output_dir / "preprocess_report.json")
    dump_json(
        build_dataset_audit(dataset_name, records_by_split),
        output_dir / "dataset_audit.json",
    )
    return output_dir


def require_columns(actual: set[str], required: set[str], source: Path) -> None:
    missing = required - actual
    if missing:
        raise ValueError(f"{source} 缺少字段: {sorted(missing)}")
