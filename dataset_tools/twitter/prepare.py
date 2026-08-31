from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import random
from pathlib import Path

from openpyxl import load_workbook

from dataset_tools.common import ImageVerifier, normalize_text, require_columns, resolve_path, safe_relative_path, sha256, text_sha256, write_workspace

SHEET_NAME = "图文对_全部来源"


def _grouped_validation(records, seed: int, ratio: float = 0.1):
    parent = list(range(len(records)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    first_owner = {}
    for index, row in enumerate(records):
        for image_hash in row["image_sha256"]:
            if image_hash in first_owner:
                union(index, first_owner[image_hash])
            else:
                first_owner[image_hash] = index
    components = defaultdict(list)
    for index, row in enumerate(records):
        components[find(index)].append(row)

    targets = {
        label: round(sum(row["label"] == label for row in records) * ratio)
        for label in (0, 1)
    }
    selected_counts = Counter()
    items = list(components.values())
    random.Random(seed).shuffle(items)
    items.sort(key=len, reverse=True)
    train, val = [], []
    for values in items:
        additions = Counter(row["label"] for row in values)
        current_error = sum(
            abs(selected_counts[label] - targets[label]) for label in (0, 1)
        )
        proposed_error = sum(
            abs(selected_counts[label] + additions[label] - targets[label])
            for label in (0, 1)
        )
        if proposed_error < current_error:
            val.extend(values)
            selected_counts.update(additions)
        else:
            train.extend(values)
    key = lambda row: (str(row["post_id"]), str(row["images"][0]))
    return sorted(train, key=key), sorted(val, key=key)


def prepare(project_root: Path, config: dict, verify_decode: bool) -> Path:
    data_root = resolve_path(project_root, config["data"]["root"])
    image_root = resolve_path(project_root, config["data"]["image_root"])
    source = data_root / config["data"].get("source_file", "MediaEval_Twitter.xlsx")
    workbook = load_workbook(source, read_only=True, data_only=True)
    if SHEET_NAME not in workbook.sheetnames:
        raise ValueError(f"{source} 缺少工作表 {SHEET_NAME!r}")
    rows = workbook[SHEET_NAME].iter_rows(values_only=True)
    header = ["" if value is None else str(value).strip() for value in next(rows)]
    required = {"图片名称", "文本内容", "类别", "帖子ID", "事件", "图片相对路径", "图片SHA256", "年份", "官方数据划分", "来源文件", "原始行号"}
    require_columns(set(header), required, source)
    index = {name: header.index(name) for name in header if name}
    verifier, candidates, audit = ImageVerifier(verify_decode), [], Counter()
    for workbook_row, row in enumerate(rows, start=2):
        audit["source_rows"] += 1
        if normalize_text(row[index["年份"]]) != "2015":
            audit["excluded_non_2015"] += 1
            continue
        label_name = normalize_text(row[index["类别"]]).lower()
        if label_name not in {"fake", "real"}:
            audit["excluded_non_binary_or_humor"] += 1
            continue
        text, post_id = normalize_text(row[index["文本内容"]]), normalize_text(row[index["帖子ID"]])
        if not text or not post_id:
            audit["excluded_empty_text_or_id"] += 1
            continue
        relative = safe_relative_path(Path(safe_relative_path(row[index["图片相对路径"]])).name)
        try:
            actual_hash = verifier(image_root / relative)
        except Exception:
            audit["excluded_missing_or_invalid_image"] += 1
            continue
        expected_hash = normalize_text(row[index["图片SHA256"]]).lower()
        if expected_hash and actual_hash != expected_hash:
            audit["excluded_image_hash_mismatch"] += 1
            continue
        official = normalize_text(row[index["官方数据划分"]]).lower()
        if official not in {"2015_dev", "2015_test"}:
            audit["excluded_unknown_official_split"] += 1
            continue
        candidates.append({
            "id": f"twitter-{post_id}", "post_id": post_id,
            "label": 0 if label_name == "fake" else 1, "label_name": label_name,
            "text": text, "text_sha256": text_sha256(text),
            "images": [relative], "image_sha256": [actual_hash], "category": "general",
            "event": normalize_text(row[index["事件"]]),
            "official_source_split": "dev" if official == "2015_dev" else "test",
            "source_file": normalize_text(row[index["来源文件"]]),
            "source_row": int(row[index["原始行号"]] or workbook_row),
            "workbook_row": workbook_row,
        })
    workbook.close()
    by_post = defaultdict(list)
    for record in candidates:
        by_post[record["post_id"]].append(record)
    selected = []
    for post_id in sorted(by_post):
        group = sorted(
            by_post[post_id],
            key=lambda row: (row["source_row"], row["images"][0]),
        )
        if len({row["label"] for row in group}) > 1:
            audit["excluded_post_label_conflicts"] += 1
            continue
        representative = dict(group[0])
        images, image_hashes, seen_hashes = [], [], set()
        for row in group:
            image_hash = row["image_sha256"][0]
            if image_hash in seen_hashes:
                continue
            seen_hashes.add(image_hash)
            images.append(row["images"][0])
            image_hashes.append(image_hash)
        representative["images"] = images
        representative["image_sha256"] = image_hashes
        representative["source_rows"] = [row["source_row"] for row in group]
        selected.append(representative)
        if len(images) > 1:
            audit["multi_image_posts_preserved"] += 1
            audit["retained_extra_image_references"] += len(images) - 1
    dev = [row for row in selected if row["official_source_split"] == "dev"]
    test = sorted(
        [row for row in selected if row["official_source_split"] == "test"],
        key=lambda row: row["post_id"],
    )
    seed = int(config.get("seed", 200408))
    train, val = _grouped_validation(dev, seed)
    records_by_split = {"train": train, "val": val, "test": test}
    sets = {
        split: {
            "post": {row["post_id"] for row in values},
            "image": {
                image_hash
                for row in values
                for image_hash in row["image_sha256"]
            },
        } for split, values in records_by_split.items()
    }
    leakage = {
        f"{left}_{right}_shared_{key}": len(sets[left][key] & sets[right][key])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
        for key in ("post", "image")
    }
    if any(leakage.values()):
        raise RuntimeError(f"Twitter 清洗后仍有跨分区泄漏: {leakage}")
    report = {
        "adapter": "dataset_tools.twitter.prepare",
        "protocol": "mediaeval2015_official_test_image_grouped_validation_v2",
        "source_workbook": str(source.resolve()), "source_sha256": sha256(source),
        "image_root": str(image_root.resolve()),
        "unit_of_analysis": "one post with one deterministically selected image",
        "label_semantics": {"0": "fake", "1": "real"},
        "text_cleaning": "Unicode NFKC + zero-width removal + whitespace normalization",
        "split_policy": {"train_val_source": "MediaEval 2015 official dev", "validation": "10% per label grouped by image SHA256", "test_source": "MediaEval 2015 official test", "seed": seed},
        "source_audit": dict(audit), "verified_unique_images": verifier.unique_images,
        "verify_decode": verify_decode,
        "output_counts": {split: {"samples": len(values), "labels": dict(Counter(row["label"] for row in values))} for split, values in records_by_split.items()},
        "leakage_checks": leakage,
        "completed_at": datetime.now().astimezone().isoformat(),
    }
    return write_workspace(project_root, config, records_by_split, report)
