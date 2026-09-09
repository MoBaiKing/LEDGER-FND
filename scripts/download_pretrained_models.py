#!/usr/bin/env python3
"""Download and verify the two local-only backbones used by CUTE-FND v3."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from huggingface_hub import snapshot_download


MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "Qwen2.5-7B-Instruct"),
    "siglip": ("google/siglip-base-patch16-224", "siglip-base-patch16-224"),
}


def complete(path: Path) -> bool:
    weights = list(path.glob("*.safetensors"))
    return (
        (path / "config.json").is_file()
        and (path / "tokenizer_config.json").is_file()
        and bool(weights)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", choices=sorted(MODELS),
                        default=list(MODELS))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--shared-cache", type=Path,
                        default=(Path(os.environ["HF_SHARED_CACHE"])
                                 if os.environ.get("HF_SHARED_CACHE") else None),
                        help="Reuse a complete shared local directory when present")
    parser.add_argument("--token", help="Hugging Face token; normally not needed")
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parent.parent
    output_root = (args.output_dir or project_root / "pretrained_models").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    report = {}
    for key in args.models:
        repository, directory_name = MODELS[key]
        destination = output_root / directory_name
        if args.shared_cache is not None:
            shared_candidate = args.shared_cache.expanduser().resolve() / directory_name
            if not destination.exists() and complete(shared_candidate):
                os.symlink(shared_candidate, destination, target_is_directory=True)
        if not complete(destination):
            snapshot_download(
                repo_id=repository,
                local_dir=destination,
                token=args.token,
            )
        if not complete(destination):
            raise RuntimeError(f"incomplete pretrained model: {destination}")
        config = json.loads((destination / "config.json").read_text(encoding="utf-8"))
        report[key] = {
            "repository": repository,
            "path": str(destination),
            "model_type": config.get("model_type"),
            "architectures": config.get("architectures"),
            "hidden_size": config.get("hidden_size"),
            "num_hidden_layers": config.get("num_hidden_layers"),
            "weight_files": len(list(destination.glob("*.safetensors"))),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
