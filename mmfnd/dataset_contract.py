from __future__ import annotations

import json
from pathlib import Path

from mmfnd.utils import resolve_path


def _require_multi_image_records(
    manifest_dir: Path,
    dataset_name: str,
) -> None:
    for split in ("train", "val", "test"):
        split_path = manifest_dir / f"{split}.jsonl"
        with split_path.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                record = json.loads(line)
                images = record.get("images")
                if not isinstance(images, list) or not images:
                    raise ValueError(
                        f"{split_path}:{line_number} 的 images 必须是非空列表。"
                    )
                if len(images) > 1:
                    return
    raise ValueError(
        f"{dataset_name} 清单仍是单图版本：train/val/test 中没有任何样本"
        "包含多张图片。请重新生成数据清单，保留每条样本的全部本地有效图片。"
    )


def validate_dataset_semantics(config: dict) -> tuple[int, dict[int, str]]:
    dataset = config.get("dataset", {})
    if "positive_label" not in dataset:
        raise ValueError("配置缺少 dataset.positive_label。")
    positive_label = int(dataset["positive_label"])
    raw_names = dataset.get("class_names")
    if not isinstance(raw_names, dict):
        raise ValueError("dataset.class_names 必须是 label 到类名的映射。")
    try:
        class_names = {int(label): str(name) for label, name in raw_names.items()}
    except (TypeError, ValueError) as error:
        raise ValueError("dataset.class_names 的 key 必须是整数标签。") from error
    if set(class_names) != {0, 1}:
        raise ValueError(
            f"当前二分类 pipeline 要求 class_names 恰好包含标签 0 和 1，得到 {sorted(class_names)}。"
        )
    positive_class = str(dataset.get("positive_class", ""))
    if positive_label not in class_names:
        raise ValueError(f"positive_label={positive_label} 不在 class_names 中。")
    if class_names[positive_label] != positive_class:
        raise ValueError(
            "dataset.positive_class 必须与 class_names[positive_label] 一致："
            f"{positive_class!r} != {class_names[positive_label]!r}。"
        )
    return positive_label, class_names


def normalize_runtime_paths(project_root: Path, config: dict) -> dict:
    validate_dataset_semantics(config)
    for key in ("text_backbone", "vision_backbone"):
        config["model"][key] = str(
            resolve_path(project_root, config["model"][key]).resolve()
        )
    return config


def bind_dataset_workspace(
    project_root: Path,
    config: dict,
    requested_dataset: str,
    manifest_dir_value: str | Path,
) -> Path:
    validate_dataset_semantics(config)
    requested = requested_dataset.strip().lower()
    configured = str(config.get("dataset", {}).get("name", "")).lower()
    if not configured:
        raise ValueError("配置缺少 dataset.name，不能确认数据集身份。")
    if requested != configured:
        raise ValueError(
            f"--dataset={requested!r} 与配置 dataset.name={configured!r} "
            "不一致，已拒绝运行，防止跨数据集串用。"
        )

    manifest_dir = resolve_path(project_root, str(manifest_dir_value)).resolve()
    metadata_path = manifest_dir / "dataset_manifest.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"清洗目录缺少 {metadata_path.name}: {manifest_dir}。"
            "请先运行 prepare_dataset.py，并把 --manifest-dir 指向其输出目录。"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = str(metadata.get("dataset", "")).lower()
    expected_schema = str(
        config.get("dataset", {}).get("schema_version", "")
    )
    if actual != requested:
        raise ValueError(
            f"清洗清单属于 dataset={actual!r}，本次请求为 {requested!r}，"
            "已拒绝运行。"
        )
    if expected_schema and metadata.get("schema_version") != expected_schema:
        raise ValueError(
            "清洗清单 schema_version 不兼容："
            f"{metadata.get('schema_version')!r} != {expected_schema!r}"
        )
    expected_labels = {
        str(label): name for label, name in validate_dataset_semantics(config)[1].items()
    }
    if metadata.get("label_semantics") != expected_labels:
        raise ValueError(
            "清洗清单 label_semantics 与配置不一致："
            f"{metadata.get('label_semantics')!r} != {expected_labels!r}"
        )
    for split in ("train", "val", "test"):
        split_path = manifest_dir / f"{split}.jsonl"
        if not split_path.is_file():
            raise FileNotFoundError(f"清洗目录缺少 {split_path}")
    if requested in {"weibo", "weibo21", "twitter"}:
        max_images = int(config["data"].get("max_images", 1))
        if max_images <= 1:
            raise ValueError(
                f"{requested} 必须启用多图读取，当前 max_images={max_images}。"
            )
        _require_multi_image_records(manifest_dir, requested)

    config["data"]["processed_dir"] = str(manifest_dir)
    manifest_image_root = metadata.get("image_root")
    if manifest_image_root:
        manifest_image_path = Path(str(manifest_image_root)).expanduser()
        # Relative image roots are part of the portable manifest contract.
        # Absolute roots may belong to the machine that produced the manifest,
        # so retain the current config unless that absolute directory exists.
        if not manifest_image_path.is_absolute():
            manifest_image_path = manifest_dir / manifest_image_path
            config["data"]["image_root"] = str(manifest_image_path.resolve())
        elif manifest_image_path.is_dir():
            config["data"]["image_root"] = str(manifest_image_path.resolve())
    config["dataset"]["manifest_dir"] = str(manifest_dir)
    normalize_runtime_paths(project_root, config)
    return manifest_dir
