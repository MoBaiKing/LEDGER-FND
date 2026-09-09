#!/usr/bin/env python3
"""Expose validated v2 datasets inside this repository without duplicating GBs."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


DATASETS = ("gossipcop", "weibo21", "twitter", "weibo")
REQUIRED_FILES = ("dataset_manifest.json", "train.jsonl", "val.jsonl", "test.jsonl")


def validate(source: Path, expected_name: str) -> dict:
    missing = [name for name in REQUIRED_FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{source} is missing {missing}")
    manifest = json.loads((source / "dataset_manifest.json").read_text(encoding="utf-8"))
    if str(manifest.get("dataset", "")).lower() != expected_name:
        raise ValueError(f"dataset identity mismatch at {source}")
    if manifest.get("label_semantics") != {"0": "fake", "1": "real"}:
        raise ValueError(f"label semantics mismatch at {source}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path,
                        default=(Path(os.environ["DATASET_SOURCE_ROOT"])
                                 if os.environ.get("DATASET_SOURCE_ROOT") else None),
                        help="Prepared dataset root (or set DATASET_SOURCE_ROOT)")
    parser.add_argument("--datasets", nargs="+", choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument("--copy", action="store_true",
                        help="Copy data instead of creating space-saving symlinks")
    args = parser.parse_args()
    if args.source_root is None:
        parser.error("--source-root or DATASET_SOURCE_ROOT is required")
    project_root = Path(__file__).resolve().parent.parent
    report = {}
    for name in args.datasets:
        source = (args.source_root / name).resolve()
        manifest = validate(source, name)
        dataset_dir = project_root / "datasets" / name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        destination = dataset_dir / "ready"
        if destination.is_symlink():
            if destination.resolve() != source:
                destination.unlink()
            else:
                report[name] = {"path": str(destination), "source": str(source),
                                "splits": manifest["splits"], "mode": "symlink"}
                continue
        elif destination.exists():
            raise FileExistsError(
                f"refusing to replace existing non-symlink destination: {destination}"
            )
        if args.copy:
            shutil.copytree(source, destination)
            mode = "copy"
        else:
            os.symlink(source, destination, target_is_directory=True)
            mode = "symlink"
        report[name] = {"path": str(destination), "source": str(source),
                        "splits": manifest["splits"], "mode": mode}
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
