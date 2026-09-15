"""Reuse the original dataset and collate without editing the training path."""
from __future__ import annotations

from copy import copy
from types import SimpleNamespace

from .corruptions import CorruptionSpec, GaussianNoiseCorruption, RandomTypoTokenInjection


class GaussianImageProcessor:
    def __init__(self, processor, corruption, dataset, image_keys):
        # This split is audited for the repo's slow SigLIP processor. Do not
        # silently reinterpret another encoder's preprocessing contract.
        if type(processor).__name__ != "SiglipImageProcessor":
            raise ValueError("Gaussian pipeline requires the repository's SiglipImageProcessor")
        if not processor.do_rescale or abs(processor.rescale_factor - 1 / 255) > 1e-12:
            raise ValueError("Expected the original SigLIP [0,255] -> [0,1] rescale")
        self.processor, self.corruption = processor, corruption
        self.dataset, self.image_keys = dataset, image_keys

    def __call__(self, images, **kwargs):
        pixel_batch = self.processor(images=images, **{**kwargs, "do_normalize": False})["pixel_values"]
        if len(pixel_batch) != len(self.image_keys):
            raise ValueError("Image identities must match flattened images")
        noisy = [self.corruption(pixels, dataset=self.dataset, sample_id=key[0],
                                 image_index=key[1], image_path=key[2]).numpy()
                 for pixels, key in zip(pixel_batch, self.image_keys)]
        # Continue with the SAME processor normalization; no resize/rescale a
        # second time, no PIL round-trip, no 8-bit requantization.
        return self.processor(images=noisy, **{
            **kwargs, "do_resize": False, "do_rescale": False, "input_data_format": "channels_first",
        })


class RobustnessCollator:
    def __init__(self, dataset, spec: CorruptionSpec):
        if dataset.train:
            raise ValueError("Corruption is test-only")
        self.dataset, self.spec = dataset, spec
        self.typo = (RandomTypoTokenInjection(dataset.processor.tokenizer, spec.typo_rate,
                                            spec.corruption_seed, dataset.max_text_length)
                     if spec.active and spec.robustness == "typo" else None)

    def __call__(self, samples):
        if not self.spec.active:
            return self.dataset.collate_fn(samples)
        dataset = self.dataset
        if self.spec.robustness == "gaussian":
            image_keys = [(item["id"], i, path) for item in samples
                          for i, path in enumerate(item["image_paths"])]
            dataset = copy(dataset)
            dataset.processor = SimpleNamespace(
                tokenizer=self.dataset.processor.tokenizer,
                image_processor=GaussianImageProcessor(
                    self.dataset.processor.image_processor,
                    GaussianNoiseCorruption(self.spec.gaussian_sigma, self.spec.corruption_seed),
                    dataset.dataset_name, image_keys),
            )
        batch = dataset.collate_fn(samples)
        if self.typo is not None:
            tokens = self.typo({"input_ids": batch["text_input_ids"],
                                "attention_mask": batch["text_attention_mask"]},
                               dataset=dataset.dataset_name, sample_ids=batch["ids"])
            batch["text_input_ids"], batch["text_attention_mask"] = tokens["input_ids"], tokens["attention_mask"]
        return batch


def build_robustness_loader(root, config, split, processor, spec=CorruptionSpec(), *, limit_samples=None):
    from mmfnd.factory import build_loader

    if split != "test":
        raise ValueError("Robustness loaders are restricted to test, including severity=0")
    loader = build_loader(root, config, split, processor)
    if limit_samples is not None:
        if limit_samples <= 0:
            raise ValueError("limit_samples must be positive")
        # In-memory diagnostic subset only; never changes a manifest on disk.
        loader.dataset.records = loader.dataset.records[:limit_samples]
    identities = [record["id"] for record in loader.dataset.records]
    if len(set(identities)) != len(identities):
        raise ValueError("Test sample IDs must be unique for deterministic per-sample corruption")
    if spec.active:
        loader.collate_fn = RobustnessCollator(loader.dataset, spec)
    return loader
