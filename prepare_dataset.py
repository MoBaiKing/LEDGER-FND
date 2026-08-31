#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

from mmfnd.utils import load_config


ADAPTERS = {
    "weibo": "dataset_tools.weibo.prepare",
    "weibo21": "dataset_tools.weibo21.prepare",
    "gossipcop": "dataset_tools.gossipcop.prepare",
    "twitter": "dataset_tools.twitter.prepare",
    "fakeddit": "dataset_tools.fakeddit.prepare",
    "finefake": "dataset_tools.finefake.prepare",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the isolated preprocessing adapter for one dataset"
    )
    parser.add_argument("--dataset", required=True, choices=sorted(ADAPTERS))
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--verify-decode",
        action="store_true",
        help="使用 Pillow 解码每张本地图片",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="只清洗标注并生成图片下载队列，不生成可训练清单",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent
    config = load_config(project_root / args.config)
    configured = str(config.get("dataset", {}).get("name", "")).lower()
    if configured != args.dataset:
        raise ValueError(
            f"--dataset={args.dataset} 与配置中的 dataset.name={configured!r} "
            "不一致，已拒绝清洗，防止数据串用。"
        )
    module = importlib.import_module(ADAPTERS[args.dataset])
    if args.metadata_only:
        prepare_metadata = getattr(module, "prepare_metadata", None)
        if prepare_metadata is None:
            raise ValueError(
                f"dataset={args.dataset} 没有 metadata-only 阶段"
            )
        output_dir = prepare_metadata(
            project_root, config, args.verify_decode
        )
        manifest = json.loads(
            (output_dir / "metadata_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        print(f"processed_dataset={args.dataset}")
        print("train_ready=false")
        print(f"metadata_dir={output_dir}")
        return
    output_dir = module.prepare(project_root, config, args.verify_decode)
    manifest = json.loads(
        (output_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    summary = {
        "dataset": manifest["dataset"],
        "schema_version": manifest["schema_version"],
        "processed_dir": manifest["processed_dir"],
        "image_root": manifest["image_root"],
        "splits": manifest["splits"],
        "cross_split_duplicate_images": manifest.get("source", {}).get(
            "cross_split_duplicate_images"
        ),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"processed_dataset={args.dataset}")
    print(f"manifest_dir={output_dir}")


if __name__ == "__main__":
    main()
