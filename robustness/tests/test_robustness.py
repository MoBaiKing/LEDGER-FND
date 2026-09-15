"""CPU integration tests; all fixtures/checkpoints are temporary and synthetic."""
from __future__ import annotations

from copy import deepcopy
import contextlib
import csv
import io
import json
import math
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from transformers import BertTokenizerFast, SiglipImageProcessor

from mmfnd.engine import evaluate, load_checkpoint, save_checkpoint
from mmfnd.factory import build_loader
from robustness.corruptions import CorruptionSpec, GaussianNoiseCorruption, RandomTypoTokenInjection
from robustness.pipeline import build_robustness_loader
from robustness.results import aggregate_rows, paired_drops, write_csv, write_json
from robustness.verify import assert_clean_equivalence

ROOT = Path(__file__).resolve().parents[2]
TinyModel = runpy.run_path(str(ROOT / "tests/test_evaluation_protocol.py"))["TinyModel"]


class InputModel(TinyModel):
    """Existing evaluator fixture, with logits actually dependent on both modalities."""
    def __init__(self, config):
        super().__init__()
        self.runtime_config = config

    def forward(self, batch):
        text = (batch["text_input_ids"] * batch["text_attention_mask"]).float().sum(1) / 200
        pixels = batch["pixel_values"]
        visual = torch.stack([pixels[batch["image_owner"] == i].mean() for i in range(len(text))])
        fake = (text + visual).sigmoid().tolist()
        return super().forward({**batch, "fake_probs": fake})


class CorruptionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.temp = tempfile.TemporaryDirectory(prefix="robustness-tests-")
        cls.root = Path(cls.temp.name)
        vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz") + ["中", "文", "新", "闻"]
        (cls.root / "vocab.txt").write_text("\n".join(vocab))
        cls.tokenizer = BertTokenizerFast(vocab_file=str(cls.root / "vocab.txt"), do_lower_case=False)
        cls.processor = SimpleNamespace(tokenizer=cls.tokenizer, image_processor=SiglipImageProcessor(size={"height": 16, "width": 16}))
        cls.config = {
            "dataset": {"name": "fixture", "positive_label": 0, "positive_class": "fake", "class_names": {"0": "fake", "1": "real"}},
            "data": {"root": str(cls.root), "image_root": str(cls.root), "processed_dir": str(cls.root),
                     "max_images": 4, "max_text_length": 32, "num_workers": 0, "text_instruction": "", "image_preprocessing": {}},
            "model": {"architecture_version": "test_cpu", "text_backbone": "fixture_text", "vision_backbone": "fixture_vision"},
            "train": {"per_gpu_batch_size": 2, "threshold_min": .2, "threshold_max": .8, "threshold_step": .01,
                      "precision": "fp32", "output_dir": str(cls.root / "runs")}, "seed": 42,
        }
        rng = np.random.default_rng(10)
        Image.fromarray(rng.integers(20, 230, (19, 21, 3), dtype=np.uint8)).save(cls.root / "image.png")
        Image.new("RGBA", (1, 9), (50, 100, 180, 50)).save(cls.root / "transparent.png")
        records = [{"dataset": "fixture", "id": f"sample{i}", "text": " ".join(list("abcdefghijklmnopqrst")[:20-i*4]),
                    "images": ["image.png", "transparent.png"] if i % 2 == 0 else ["image.png"], "label": i % 2} for i in range(4)]
        for split in ("train", "val", "test"):
            (cls.root / f"{split}.jsonl").write_text("\n".join(json.dumps(r) for r in records))
        write_json(cls.root / "dataset_manifest.json", {"dataset": "fixture", "label_semantics": {"0": "fake", "1": "real"}})

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def build(self, spec=CorruptionSpec(), config=None):
        return build_robustness_loader(self.root, config or self.config, "test", self.processor, spec)

    def test_parameter_exclusion_and_validation(self):
        for kwargs in ({"robustness": "none", "typo_rate": .1}, {"robustness": "typo", "gaussian_sigma": .1},
                       {"robustness": "gaussian", "typo_rate": .1}, {"robustness": "gaussian", "gaussian_sigma": -1},
                       {"robustness": "typo", "typo_rate": float("nan")}, {"robustness": "typo", "typo_rate": 1.1}):
            with self.assertRaises(ValueError):
                CorruptionSpec(**kwargs)
        for split in ("train", "val"):
            with self.assertRaisesRegex(ValueError, "test"):
                build_robustness_loader(self.root, self.config, split, self.processor)

    def test_zero_identity_and_rng_state(self):
        pixels = torch.full((3, 16, 16), .5)
        np_state, torch_state = np.random.get_state(), torch.get_rng_state()
        self.assertIs(GaussianNoiseCorruption(0)(pixels, dataset="d", sample_id="a"), pixels)
        tokens = self.tokenizer(["a b c"], return_tensors="pt")
        self.assertIs(RandomTypoTokenInjection(self.tokenizer, 0)(tokens, dataset="d", sample_ids=["a"]), tokens)
        GaussianNoiseCorruption(.1)(pixels, dataset="d", sample_id="a")
        np.testing.assert_equal(np.random.get_state(), np_state)
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))

    def test_gaussian_severity_range_shape_and_seed(self):
        pixels = torch.full((3, 224, 224), .5)
        outputs = [GaussianNoiseCorruption(s)(pixels, dataset="d", sample_id="a") for s in (.05, .1, .2)]
        rmses = [float((x-pixels).square().mean().sqrt()) for x in outputs]
        self.assertTrue(rmses[0] < rmses[1] < rmses[2])
        for x in outputs:
            self.assertEqual(x.shape, pixels.shape)
            self.assertTrue(0 <= x.min() <= x.max() <= 1)
        self.assertTrue(torch.equal(outputs[1], GaussianNoiseCorruption(.1)(pixels, dataset="d", sample_id="a")))
        for kwargs in ({"sample_id": "b"}, {"image_index": 1}, {"dataset": "other"}):
            other = GaussianNoiseCorruption(.1)(pixels, **{"dataset": "d", "sample_id": "a", **kwargs})
            self.assertFalse(torch.equal(outputs[1], other))
        self.assertFalse(torch.equal(outputs[1], GaussianNoiseCorruption(.1, 2028)(pixels, dataset="d", sample_id="a")))
        with self.assertRaisesRegex(ValueError, "unnormalized"):
            GaussianNoiseCorruption(.1)(pixels - 1, dataset="d", sample_id="a")

    def test_gaussian_normalization_exact_and_text_unchanged(self):
        clean = next(iter(self.build()))
        noisy = next(iter(self.build(CorruptionSpec("gaussian", .1))))
        for name in ("text_input_ids", "text_attention_mask", "image_owner", "labels"):
            self.assertTrue(torch.equal(clean[name], noisy[name]), name)
        samples = [self.build().dataset[i] for i in range(2)]
        images = [im for sample in samples for im in sample["images"]]
        pixels = self.processor.image_processor(images=images, do_normalize=False, return_tensors="pt", input_data_format="channels_last")["pixel_values"]
        keys = [(sample["id"], i, path) for sample in samples for i, path in enumerate(sample["image_paths"])]
        expected = torch.stack([GaussianNoiseCorruption(.1)(p, dataset="fixture", sample_id=k[0], image_index=k[1], image_path=k[2])
                                for p, k in zip(pixels, keys)])
        torch.testing.assert_close(noisy["pixel_values"], (expected - .5) / .5, atol=0, rtol=0)

    def test_typo_insertion_retention_and_special_ids(self):
        ids = [self.tokenizer.cls_token_id] + list(range(5, 25)) + [self.tokenizer.sep_token_id]
        counts = []
        for rate in (.05, .1, .2):
            corrupt = RandomTypoTokenInjection(self.tokenizer, rate, max_length=32)
            injected, report = corrupt.inject_sequence(ids, dataset="d", sample_id="a")
            counts.append(report["requested_insertions"])
            self.assertEqual(report["truncated_original_tokens"], 0)
            self.assertFalse(set(report["injected_token_ids"]) & set(self.tokenizer.all_special_ids))
            expected, inserted = [], dict(zip(report["inserted_after_positions"], report["injected_token_ids"]))
            for i, token in enumerate(ids):
                expected.append(token)
                if i in inserted:
                    expected.append(inserted[i])
            self.assertEqual(expected, injected)
            self.assertEqual(injected, corrupt.inject_sequence(ids, dataset="d", sample_id="a")[0])
        self.assertEqual(counts, [1, 2, 4])
        new = RandomTypoTokenInjection(self.tokenizer, .2, seed=2028, max_length=32).inject_sequence(ids, dataset="d", sample_id="a")[0]
        self.assertNotEqual(new, injected)

    def test_typo_tail_truncation_padding_empty_and_special_only(self):
        ids = [self.tokenizer.cls_token_id] + list(range(5, 25)) + [self.tokenizer.sep_token_id]
        for side in ("right", "left"):
            tokenizer = deepcopy(self.tokenizer)
            tokenizer.truncation_side = side
            corrupt = RandomTypoTokenInjection(tokenizer, .2, max_length=22)
            new, report = corrupt.inject_sequence(ids, dataset="d", sample_id="a")
            self.assertEqual(len(new), 22)
            self.assertEqual([t for t in new if t in corrupt.special_ids], [ids[0], ids[-1]])
            full = RandomTypoTokenInjection(tokenizer, .2, max_length=32).inject_sequence(ids, dataset="d", sample_id="a")[0]
            ordinary = full[1:-1]
            self.assertEqual(new[1:-1], ordinary[:20] if side == "right" else ordinary[-20:])
        tokens = {"input_ids": torch.tensor([ids, [2, 3] + [0] * 20]),
                  "attention_mask": torch.tensor([[1] * 22, [1, 1] + [0] * 20])}
        new = corrupt(tokens, dataset="d", sample_ids=["a", "empty"])
        self.assertEqual(new["input_ids"].shape, new["attention_mask"].shape)
        self.assertEqual(new["attention_mask"][1].sum(), 2)
        self.assertEqual(new["input_ids"][1][new["attention_mask"][1] == 0].unique().tolist(), [0])
        self.assertEqual(corrupt.inject_sequence([], dataset="d", sample_id="empty")[0], [])
        with self.assertRaisesRegex(ValueError, "Unexpected tokenizer fields"):
            corrupt({**tokens, "position_ids": tokens["input_ids"]}, dataset="d", sample_ids=["a", "b"])

    def test_typo_image_unchanged_and_worker_batch_determinism(self):
        clean = next(iter(self.build()))
        typo = next(iter(self.build(CorruptionSpec("typo", typo_rate=.2))))
        for name in ("pixel_values", "image_owner", "labels"):
            self.assertTrue(torch.equal(clean[name], typo[name]), name)
        def per_sample(loader):
            result = {}
            for batch in loader:
                for i, identity in enumerate(batch["ids"]):
                    result[identity] = (batch["text_input_ids"][i][batch["text_attention_mask"][i].bool()],
                                        batch["pixel_values"][batch["image_owner"] == i])
            return result
        for spec in (CorruptionSpec("typo", typo_rate=.2), CorruptionSpec("gaussian", .1)):
            first = per_sample(self.build(spec))
            config = deepcopy(self.config)
            config["data"].update(num_workers=2, persistent_workers=False)
            config["train"]["per_gpu_batch_size"] = 3
            second = per_sample(self.build(spec, config))
            for identity in first:
                for x, y in zip(first[identity], second[identity]):
                    self.assertTrue(torch.equal(x, y), identity)

    def test_language_agnostic_real_wordpiece(self):
        for text in ("a b c d e f g h i j", "中文新闻中文新闻中文新闻"):
            tokens = self.tokenizer([text], return_tensors="pt", return_token_type_ids=False)
            result = RandomTypoTokenInjection(self.tokenizer, .2, max_length=64)(tokens, dataset="d", sample_ids=[text])
            self.assertGreater(result["attention_mask"].sum(), tokens["attention_mask"].sum())

    def test_missing_image_and_original_file_semantics(self):
        import hashlib
        path = self.root / "image.png"
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        list(self.build(CorruptionSpec("gaussian", .1)))
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
        dataset = self.build().dataset
        dataset.records = [{**dataset.records[0], "images": ["nonexistent.png"]}]
        with self.assertRaises(FileNotFoundError):
            dataset[0]

    def test_clean_logits_probabilities_metrics_same_checkpoint(self):
        model = InputModel(deepcopy(self.config))
        metrics, _, _ = evaluate(model, self.build(), torch.device("cpu"), 0, {0: "fake", 1: "real"},
                                  split="val", tune_threshold=True, checkpoint_reference={"kind": "epoch", "epoch": 2}, show_progress=False)
        path = self.root / "equivalence.pth"
        save_checkpoint(path, model, None, None, 2, metrics, model.runtime_config)
        checkpoint = load_checkpoint(path, model, torch.device("cpu"))
        outputs = []
        hook = model.register_forward_hook(lambda m, args, out: outputs.append(out["logits"].detach()))
        reference = None
        for spec in (CorruptionSpec(), CorruptionSpec("gaussian"), CorruptionSpec("typo")):
            outputs.clear()
            data = build_loader(self.root, self.config, "test", self.processor) if spec.robustness == "none" else self.build(spec)
            metrics, rows, _ = evaluate(model, data, torch.device("cpu"), 0, {0: "fake", 1: "real"}, split="test",
                                        threshold_selection=checkpoint["threshold_selection"], show_progress=False)
            if reference is None:
                reference = metrics, rows, torch.cat(outputs)
            else:
                report = assert_clean_equivalence(*reference, metrics, rows, torch.cat(outputs), spec.robustness)
                self.assertEqual(report["logits_max_abs_diff"], 0)
        hook.remove()

    def test_paired_drops_and_undefined_values(self):
        result = paired_drops({"macro_f1": .8, "accuracy": 0., "auc": float("nan")}, {"macro_f1": .6, "accuracy": .5, "auc": .5})
        self.assertAlmostEqual(result["f1_drop"], .2)
        self.assertAlmostEqual(result["relative_f1_drop_pct"], 25)
        self.assertEqual(result["acc_drop"], -.5)
        self.assertIsNone(result["relative_acc_drop_pct"])
        self.assertIsNone(result["auc_drop"])

    def test_suite_five_seeds_shared_engine_clean_once_and_plot(self):
        import robustness.evaluate as cli
        from robustness.plot_robustness import plot_results

        paths, models = [], []
        for seed in range(5):
            config = deepcopy(self.config)
            config["seed"] = seed
            model = InputModel(config)
            metrics, _, _ = evaluate(model, self.build(), torch.device("cpu"), 0, {0: "fake", 1: "real"}, split="val",
                                      tune_threshold=True, checkpoint_reference={"kind": "epoch", "epoch": 2}, show_progress=False)
            path = self.root / f"seed{seed}.pth"
            save_checkpoint(path, model, None, None, 2, metrics, config)
            paths.append(path)
        def construct(config):
            model = InputModel(config)
            models.append(model)
            return model
        output = self.root / "five_seed_output"
        argv = ["--suite", "--device", "cpu", "--output-dir", str(output)]
        for path in paths:
            argv.extend(["--checkpoint", str(path)])
        with patch.object(cli, "ExplainableMMFND", construct), patch.object(cli, "build_processor", return_value=self.processor), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            cli.main(argv)
        self.assertTrue((output / "completion.json").is_file())
        # 7 severities, two batches each; clean=0 never evaluated twice.
        self.assertEqual([m.calls for m in models], [14] * 5)
        with (output / "robustness_summary.csv").open(encoding="utf-8-sig") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 35)
        aggregate = json.loads((output / "robustness_aggregate.json").read_text())
        self.assertEqual(len(aggregate), 7)
        self.assertTrue(all(r["n_seeds"] == 5 for r in aggregate))
        self.assertTrue(all(r["macro_f1_std"] == 0 for r in aggregate))
        for seed in range(5):
            base = output / "fixture" / f"seed_{seed}"
            clean = json.loads((base / "clean.json").read_text())
            for name in ("gaussian/sigma_0.00.json", "typo/rate_0.00.json"):
                self.assertEqual(clean["metrics"], json.loads((base / name).read_text())["metrics"])
        plot_results([output / "robustness_summary.csv"], output / "figures")
        for method in ("gaussian", "typo"):
            for ext in ("png", "pdf"):
                self.assertGreater((output / f"figures/{method}_robustness.{ext}").stat().st_size, 1000)
        # Mean/std are tested with distinct per-seed observations too.
        records = [{"dataset": "d", "method": "LEDGER", "corruption": "clean", "severity": 0,
                    "corruption_seed": 2027, "evaluation_scope": "full_test", "samples": 2,
                    "checkpoint_seed": seed, "macro_f1": score} for seed, score in enumerate((.4, .6))]
        summary = aggregate_rows(records)[0]
        self.assertEqual(summary["macro_f1_mean"], .5)
        self.assertAlmostEqual(summary["macro_f1_std"], math.sqrt(.02))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            aggregate_rows(records + records[:1])


if __name__ == "__main__":
    unittest.main(verbosity=2)
