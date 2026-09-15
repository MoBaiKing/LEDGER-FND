"""Stateless corruption RNGs independent of batching and worker scheduling."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import numpy as np
import torch


def sample_rng(seed: int, *identity) -> np.random.Generator:
    payload = json.dumps([int(seed), *identity], ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return np.random.Generator(np.random.PCG64(int.from_bytes(digest[:16], "big")))


@dataclass(frozen=True)
class CorruptionSpec:
    robustness: str = "none"
    gaussian_sigma: float = 0.0
    typo_rate: float = 0.0
    corruption_seed: int = 2027

    def __post_init__(self):
        if self.robustness not in {"none", "gaussian", "typo"}:
            raise ValueError("robustness must be none, gaussian, or typo")
        if not math.isfinite(self.gaussian_sigma) or self.gaussian_sigma < 0:
            raise ValueError("gaussian_sigma must be finite and nonnegative")
        if not math.isfinite(self.typo_rate) or not 0 <= self.typo_rate <= 1:
            raise ValueError("typo_rate must be finite and in [0, 1]")
        if self.robustness != "gaussian" and self.gaussian_sigma != 0:
            raise ValueError("gaussian_sigma requires robustness=gaussian; mixed corruption is forbidden")
        if self.robustness != "typo" and self.typo_rate != 0:
            raise ValueError("typo_rate requires robustness=typo; mixed corruption is forbidden")

    @property
    def severity(self):
        return self.gaussian_sigma if self.robustness == "gaussian" else self.typo_rate

    @property
    def active(self):
        return self.robustness != "none" and self.severity > 0


class GaussianNoiseCorruption:
    """Add noise to one CPU float pixel tensor AFTER rescale, BEFORE normalize.

    A common epsilon across severities makes intensity comparisons paired;
    distinct sample/image identities always get independent streams.
    """

    def __init__(self, sigma=0.0, seed=2027):
        CorruptionSpec("gaussian", gaussian_sigma=sigma, corruption_seed=seed)
        self.sigma, self.seed = sigma, seed

    def __call__(self, pixels: torch.Tensor, *, dataset, sample_id, image_index=0, image_path=""):
        if self.sigma == 0:
            return pixels  # no conversion, arithmetic, allocation or RNG use
        if pixels.device.type != "cpu" or not pixels.is_floating_point():
            raise ValueError("Gaussian corruption expects a CPU float pixel tensor")
        if not torch.isfinite(pixels).all() or pixels.min() < 0 or pixels.max() > 1:
            raise ValueError("Gaussian corruption requires unnormalized pixels in [0, 1]")
        rng = sample_rng(self.seed, "gaussian", dataset, sample_id, image_index, str(image_path))
        epsilon = torch.from_numpy(rng.standard_normal(tuple(pixels.shape), dtype=np.float32))
        return (pixels + self.sigma * epsilon.to(pixels.dtype)).clamp(0, 1)


class RandomTypoTokenInjection:
    """Insert round(rate * non-special length) tokens, then truncate and repad.

    Counts use round-half-up. Positions are nested across severity and sampled
    without replacement. All original special tokens are protected, including
    EOS when truncating the ordinary-token tail. The current Qwen tokenizer has
    no automatic BOS/EOS template. Its right truncation is therefore exactly a
    prefix slice for ordinary news. Instruction tokens are part of the original
    prompted input sequence and participate just like other non-special tokens.
    """

    def __init__(self, tokenizer, rate=0.0, seed=2027, max_length=192):
        CorruptionSpec("typo", typo_rate=rate, corruption_seed=seed)
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.tokenizer, self.rate, self.seed, self.max_length = tokenizer, rate, seed, max_length
        self.special_ids = set(tokenizer.all_special_ids)
        self.vocabulary = np.asarray(sorted(set(tokenizer.get_vocab().values()) - self.special_ids), dtype=np.int64)
        if rate > 0 and not self.vocabulary.size:
            raise ValueError("Tokenizer has no legal non-special vocabulary")
        if tokenizer.truncation_side not in {"left", "right"}:
            raise ValueError("Unsupported tokenizer truncation_side")

    def inject_sequence(self, ids, *, dataset, sample_id):
        original = [int(token) for token in ids]
        if len(original) > self.max_length:
            raise ValueError("Expected the repository's already-truncated token sequence")
        eligible = [i for i, token in enumerate(original) if token not in self.special_ids]
        count = int(math.floor(self.rate * len(eligible) + 0.5))
        rng = sample_rng(self.seed, "typo", dataset, sample_id)
        positions = rng.permutation(eligible).tolist()
        random_tokens = rng.choice(self.vocabulary, size=len(eligible)).tolist() if eligible else []
        insertions = dict(zip(positions[:count], random_tokens[:count]))
        expanded, origins, anchors = [], [], []
        for i, token in enumerate(original):
            expanded.append(token)
            origins.append(i)
            anchors.append(i)
            if i in insertions:
                expanded.append(int(insertions[i]))
                origins.append(None)
                anchors.append(i)
        overflow = max(0, len(expanded) - self.max_length)
        ordinary = [i for i, token in enumerate(expanded) if token not in self.special_ids]
        removed = set((ordinary[-overflow:] if self.tokenizer.truncation_side == "right" else ordinary[:overflow])
                      if overflow else [])
        result = [token for i, token in enumerate(expanded) if i not in removed]
        kept_origins = [origin for i, origin in enumerate(origins) if i not in removed]
        return result, {
            "original_length": len(original), "eligible_tokens": len(eligible),
            "requested_insertions": count, "inserted_after_positions": sorted(insertions),
            "injected_token_ids": [insertions[i] for i in sorted(insertions)],
            "retained_insertions": sum(origin is None for origin in kept_origins),
            "truncated_original_tokens": len(original) - sum(origin is not None for origin in kept_origins),
            "truncated_tokens": overflow, "final_length": len(result),
            # Optional provenance for baseline token_type_ids / aligned fields.
            # Does not affect sampling, IDs, truncation or the LEDGER pipeline.
            "source_positions": kept_origins,
            "anchor_positions": [anchor for i, anchor in enumerate(anchors) if i not in removed],
        }

    def __call__(self, tokens: dict, *, dataset, sample_ids):
        if self.rate == 0:
            return tokens
        # The repository forwards only IDs and attention mask to Qwen, which
        # generates position_ids internally. Fail if that contract ever changes.
        if set(tokens) != {"input_ids", "attention_mask"}:
            raise ValueError("Unexpected tokenizer fields: update their insertion/padding semantics explicitly")
        if len(sample_ids) != len(tokens["input_ids"]):
            raise ValueError("Sample IDs and token rows must align")
        rows = []
        for ids, mask, sample_id in zip(tokens["input_ids"], tokens["attention_mask"], sample_ids):
            if not torch.all((mask == 0) | (mask == 1)):
                raise ValueError("attention_mask must be binary")
            active = ids[mask.bool()].tolist()
            injected, _ = self.inject_sequence(active, dataset=dataset, sample_id=sample_id)
            rows.append({"input_ids": injected, "attention_mask": [1] * len(injected)})
        return self.tokenizer.pad(rows, padding=True, return_tensors="pt")
