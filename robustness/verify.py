"""Correctness reports for actual preprocessing and shared-evaluator outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from robustness.corruptions import GaussianNoiseCorruption, RandomTypoTokenInjection
from robustness.results import write_json


def assert_nested_close(first, second, *, atol=1e-7, rtol=1e-6):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_close(first[key], second[key], atol=atol, rtol=rtol)
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_close(a, b, atol=atol, rtol=rtol)
    elif isinstance(first, float):
        np.testing.assert_allclose(first, second, atol=atol, rtol=rtol, equal_nan=True)
    else:
        assert first == second, (first, second)


def assert_clean_equivalence(clean_metrics, clean_rows, clean_logits, metrics, rows, logits, corruption):
    torch.testing.assert_close(clean_logits, logits, atol=1e-7, rtol=1e-6)
    assert_nested_close(clean_metrics, metrics)
    assert_nested_close(clean_rows, rows)
    probabilities = lambda data: np.asarray([list(row["class_probabilities"].values()) for row in data])
    delta = np.abs(probabilities(clean_rows) - probabilities(rows)).max()
    return {"robustness": corruption, "severity": 0, "passed": True,
            "logits_max_abs_diff": float((clean_logits - logits).abs().max()),
            "probabilities_max_abs_diff": float(delta),
            "metrics_equal_within_tolerance": True, "atol": 1e-7, "rtol": 1e-6}


def main():
    from mmfnd.dataset_contract import bind_dataset_workspace
    from mmfnd.factory import build_loader, build_processor
    from mmfnd.utils import load_config

    p = argparse.ArgumentParser(description="Real tokenizer/image examples, without loading the model")
    p.add_argument("--config", default="configs/datasets/weibo21.json")
    p.add_argument("--manifest-dir")
    p.add_argument("--output-dir", type=Path, default=ROOT / "robustness/test_artifacts/preprocessing")
    p.add_argument("--corruption-seed", type=int, default=2027)
    args = p.parse_args()
    config = load_config(ROOT / args.config)
    dataset = config["dataset"]["name"]
    bind_dataset_workspace(ROOT, config, dataset, args.manifest_dir or f"datasets/{dataset}/ready")
    processor = build_processor(ROOT, config)
    loader = build_loader(ROOT, config, "test", processor)
    examples = [loader.dataset[i] for i in range(min(4, len(loader.dataset)))]
    batch = loader.dataset.collate_fn(examples)
    tokenizer = processor.tokenizer
    typo_examples = []
    for i, sample in enumerate(examples):
        ids = batch["text_input_ids"][i][batch["text_attention_mask"][i].bool()].tolist()
        entry = {"sample_id": sample["id"], "clean_ids": ids, "clean_tokens": tokenizer.convert_ids_to_tokens(ids), "rates": []}
        for rate in (.05, .10, .20):
            injector = RandomTypoTokenInjection(tokenizer, rate, args.corruption_seed, loader.dataset.max_text_length)
            new, details = injector.inject_sequence(ids, dataset=dataset, sample_id=sample["id"])
            repeated, _ = injector.inject_sequence(ids, dataset=dataset, sample_id=sample["id"])
            assert new == repeated
            assert len(new) <= loader.dataset.max_text_length
            assert [t for t in ids if t in injector.special_ids] == [t for t in new if t in injector.special_ids]
            assert not set(details["injected_token_ids"]) & injector.special_ids
            changed, _ = RandomTypoTokenInjection(tokenizer, rate, args.corruption_seed + 1,
                                                   loader.dataset.max_text_length).inject_sequence(
                                                       ids, dataset=dataset, sample_id=sample["id"])
            if details["requested_insertions"]:
                assert changed != new
            encoded = injector({"input_ids": batch["text_input_ids"][i:i+1],
                                "attention_mask": batch["text_attention_mask"][i:i+1]},
                               dataset=dataset, sample_ids=[sample["id"]])
            assert encoded["input_ids"][0][encoded["attention_mask"][0].bool()].tolist() == new
            entry["rates"].append({"rate": rate, "ids": new, "tokens": tokenizer.convert_ids_to_tokens(new), **details})
        typo_examples.append(entry)
        print(json.dumps(entry, ensure_ascii=False))
    sample = examples[0]
    clean = processor.image_processor(images=[sample["images"][0]], return_tensors="pt",
                                      do_normalize=False, input_data_format="channels_last")["pixel_values"][0]
    images, gaussian_examples = [clean], []
    kwargs = {"dataset": dataset, "sample_id": sample["id"], "image_index": 0, "image_path": sample["image_paths"][0]}
    for sigma in (.05, .10, .20):
        corruption = GaussianNoiseCorruption(sigma, args.corruption_seed)
        noisy = corruption(clean, **kwargs)
        assert torch.equal(noisy, corruption(clean, **kwargs))
        assert not torch.equal(noisy, GaussianNoiseCorruption(sigma, args.corruption_seed + 1)(clean, **kwargs))
        images.append(noisy)
        gaussian_examples.append({"sigma": sigma, "rmse": float((noisy-clean).square().mean().sqrt()),
                                  "min": float(noisy.min()), "max": float(noisy.max()), "shape": list(noisy.shape)})
    assert all(a["rmse"] < b["rmse"] for a, b in zip(gaussian_examples, gaussian_examples[1:]))
    write_json(args.output_dir / "examples.json", {"dataset": dataset, "corruption_seed": args.corruption_seed,
                                                  "gaussian": gaussian_examples, "typo": typo_examples})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(10, 2.7), constrained_layout=True)
    for ax, pixels, sigma in zip(axes, images, (0, .05, .10, .20)):
        ax.imshow(pixels.permute(1, 2, 0).numpy())
        ax.set_title(f"sigma={sigma:.2f}")
        ax.axis("off")
    fig.savefig(args.output_dir / "gaussian_example.png", dpi=150)
    plt.close(fig)
    print(json.dumps(gaussian_examples, indent=2))
    print(f"examples={args.output_dir}")


if __name__ == "__main__":
    main()
