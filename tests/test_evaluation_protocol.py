"""CPU-only protocol tests; no pretrained model downloads or GPU execution."""
import contextlib
import copy
import io
import json
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score

from mmfnd.evaluation import (
    PROTOCOL, checkpoint_threshold_selection, compute_classification_metrics,
    evaluation_info, find_best_macro_f1_threshold, threshold_grid,
)
from mmfnd.engine import evaluate, load_checkpoint, save_checkpoint


def probs(fake):
    fake = np.asarray(fake, dtype=float)
    return np.column_stack([fake, 1 - fake])  # ORIGINAL v3 order: Fake, Real


class MetricTests(unittest.TestCase):
    def test_macro_objective_differs_from_positive_f1(self):
        labels, fake = [0, 1, 1, 1, 1], np.array([.4, .7, .7, .7, .1])
        grid = [.2, .5, .8]
        selected = find_best_macro_f1_threshold(labels, fake, grid, split="val")
        binary_scores = [f1_score(labels, np.where(fake >= t, 0, 1), pos_label=0, average="binary") for t in grid]
        self.assertEqual(grid[int(np.argmax(binary_scores))], .2)
        self.assertEqual(selected["threshold"], .8)
        self.assertAlmostEqual(selected["macro_f1"], 4 / 9)

    def test_exact_ties_ignore_input_order(self):
        for grid in ([.7, .6, .4], [.4, .6, .7]):
            selected = find_best_macro_f1_threshold([0, 1], [.9, .1], grid, split="val")
            self.assertEqual(selected["threshold"], .4)
        self.assertEqual(find_best_macro_f1_threshold([0, 1], [.9, .1], [.4, .5, .6], split="val")["threshold"], .5)

    def test_near_optimal_plateau_is_not_allowed(self):
        selected = find_best_macro_f1_threshold([0] * 500 + [1] * 500,
                                               [.49] + [.9] * 499 + [.1] * 500, [.48, .5], split="val")
        self.assertEqual(selected["threshold"], .48)
        self.assertEqual(selected["macro_f1"], 1)

    def test_shared_predictions_and_probability_metrics(self):
        labels, probabilities = [0, 0, 1, 1], probs([.45, .4, .35, .1])
        tuned = compute_classification_metrics(labels, probabilities, .4)
        other = compute_classification_metrics(labels, probabilities, .8)
        self.assertEqual(tuned["accuracy"], 1)
        self.assertEqual(accuracy_score(labels, probabilities.argmax(axis=1)), .5)
        self.assertEqual(tuned["macro_f1"], (tuned["fake_f1"] + tuned["real_f1"]) / 2)
        self.assertEqual(tuned["confusion_matrix"], [[2, 0], [0, 2]])
        self.assertEqual(tuned["positive_label"], 0)
        self.assertEqual(tuned["label_semantics"], {"0": "fake", "1": "real"})
        for metric in ("auc", "nll", "brier", "ece"):
            self.assertEqual(tuned[metric], other[metric])
        self.assertAlmostEqual(tuned["nll"], log_loss(labels, probabilities, labels=[0, 1]))
        self.assertAlmostEqual(tuned["auc"], roc_auc_score(np.array(labels) == 0, probabilities[:, 0]))
        self.assertAlmostEqual(tuned["brier"], np.mean((probabilities[:, 0] - (np.array(labels) == 0)) ** 2))
        # .60 and .65 share bin [9/15, 10/15): confidence .625, accuracy .5.
        self.assertAlmostEqual(tuned["ece"], .55 / 4 + abs(.5 - .625) / 2 + .1 / 4)

    def test_all_predictions_one_class_and_single_class_auc(self):
        for threshold in (0, 1):
            metrics = compute_classification_metrics([0, 1], probs([.2, .8]), threshold)
            self.assertTrue(all(math.isfinite(metrics[k]) for k in ("accuracy", "macro_f1", "fake_f1", "real_f1")))
        for label in (0, 1):
            with self.assertWarnsRegex(RuntimeWarning, "single-class"):
                metrics = compute_classification_metrics([label, label], probs([0., 1.]), .5)
            self.assertTrue(math.isnan(metrics["auc"]))
            self.assertTrue(math.isfinite(metrics["nll"]))

    def test_invalid_inputs_and_test_tuning_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "validation"):
            find_best_macro_f1_threshold([0, 1], [.8, .2], split="test")
        for labels, fake in (([], []), ([0], [.2, .3]), ([2], [.2]), ([0], [float("nan")])):
            with self.assertRaises(ValueError):
                find_best_macro_f1_threshold(labels, fake, split="val")
        with self.assertRaises(ValueError):
            compute_classification_metrics([0], [[.6, .6]], .5)
        grid = threshold_grid({"threshold_min": .2, "threshold_max": .8, "threshold_step": .01})
        self.assertEqual((len(grid), grid[0], grid[-1]), (61, .2, .8))


class TinyModel(torch.nn.Module):
    """Minimal CPU model satisfying the existing diagnostic output contract."""
    architecture_version = "test_cpu"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.))
        self.runtime_config = {
            "dataset": {"name": "fixture", "positive_label": 0, "positive_class": "fake",
                        "class_names": {"0": "fake", "1": "real"}},
            "model": {"architecture_version": "test_cpu", "text_backbone": "fixture_text", "vision_backbone": "fixture_vision"},
            "train": {"threshold_min": .2, "threshold_max": .8, "threshold_step": .01},
        }
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        n = len(batch["labels"])
        outputs = {"logits": torch.tensor(probs(batch["fake_probs"]), dtype=torch.float32).log() * self.weight}
        outputs["relation_probs"] = torch.full((n, 6, 3), 1 / 3)
        for name in ("relation_uncertainty", "relation_strength"):
            outputs[name] = torch.ones(n, 6)
        for name in ("evidence_confidence", "evidence_deviation", "minority_score", "global_judge_weights",
                     "deliberative_weights", "direct_weights", "final_evidence_weights"):
            outputs[name] = torch.ones(n, 4)
        for name in ("sample_disagreement", "routing_gate"):
            outputs[name] = torch.zeros(n, 1)
        outputs["uncertainty_components"] = torch.zeros(n, 4)
        outputs["uncertainty_component_weights"] = torch.ones(4) / 4
        for name in ("modality_weights", "causal_gate_effects"):
            outputs[name] = torch.zeros(n, 3)
        for name in ("uncertainty", "uncertainty_temperature", "uncertainty_correction_norm", "text_relation_ambiguity",
                     "event_contradiction", "visual_evidence_inconsistency", "multi_image_dispersion", "multi_image_available"):
            outputs[name] = torch.zeros(n)
        return outputs


def loader(labels=(0, 0, 1, 1), fake=(.45, .4, .35, .1)):
    n = len(labels)
    return [{"labels": torch.tensor(labels), "fake_probs": fake, "ids": list(range(n)),
             "image_paths": [[] for _ in range(n)], "texts": [""] * n, "categories": [""] * n}]


def val(model, data=None, reference=None):
    return evaluate(model, data or loader(), torch.device("cpu"), 0, {0: "fake", 1: "real"},
                    show_progress=False, split="val", tune_threshold=True,
                    checkpoint_reference=reference or {"kind": "epoch", "epoch": 2})


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # Keep even availability probes on the CPU path in this test suite.
        guard = patch("torch.cuda.is_available", return_value=False)
        guard.start()
        self.addCleanup(guard.stop)

    def test_all_dataset_configs_preserve_v3_labels_and_macro_monitor(self):
        config_dir = Path(__file__).resolve().parents[1] / "configs/datasets"
        for path in config_dir.glob("*.json"):
            config = json.loads(path.read_text())
            self.assertEqual(config["train"]["monitor"], "macro_f1", str(path))
            self.assertNotIn("threshold_plateau_delta", config["train"], str(path))
            self.assertEqual(config["dataset"]["positive_label"], 0, str(path))
            self.assertEqual(config["dataset"]["class_names"], {"0": "fake", "1": "real"}, str(path))

    def test_manual_legacy_evaluation_requires_explicit_validation_recalibration(self):
        import evaluate as evaluate_cli
        with tempfile.TemporaryDirectory(prefix="v3-manual-eval-test-") as directory:
            root = Path(directory)
            model = TinyModel()
            config = model.runtime_config
            config["seed"] = 123
            config["train"].update(output_dir=directory, precision="fp32")
            source = root / "legacy.pth"
            save_checkpoint(source, model, None, None, 2, {"legacy": True}, config)
            original_bytes = source.read_bytes()
            built_splits = []
            def build_loader(project, cfg, split, processor):
                built_splits.append(split)
                return loader()
            args = ["evaluate.py", "--config", "unused", "--checkpoint", str(source), "--split", "test"]
            with contextlib.ExitStack() as stack:
                for name, replacement in (("load_config", lambda p: config), ("get_device", lambda: torch.device("cpu")),
                                           ("bind_dataset_workspace", lambda *a: root), ("build_processor", lambda *a: None),
                                           ("build_loader", build_loader), ("build_model", lambda c: model)):
                    stack.enter_context(patch.object(evaluate_cli, name, replacement))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "recalibrate"):
                    evaluate_cli.main()
                self.assertEqual(model.calls, 0)
                with patch.object(sys, "argv", args + ["--recalibrate-on-val"]):
                    evaluate_cli.main()
            self.assertEqual(built_splits, ["val", "test"])
            calibrated = next(root.glob("manual_evaluation/**/validation_calibrated.pth"))
            selection = checkpoint_threshold_selection(torch.load(calibrated, weights_only=False))
            test = json.loads((calibrated.parent / "test/metrics.json").read_text())
            self.assertEqual(test["threshold_selection"], selection)
            self.assertEqual(source.read_bytes(), original_bytes, "Legacy checkpoints must not be overwritten")

    def test_training_loop_early_stop_best_topk_and_final_test(self):
        import train
        with tempfile.TemporaryDirectory(prefix="v3-training-protocol-test-") as directory:
            root = Path(directory)
            config = json.loads((Path(train.__file__).parent / "configs/datasets/weibo21.json").read_text())
            config["dataset"].update(name="fixture")
            config["model"]["architecture_version"] = "test_cpu"
            config["train"].update(output_dir=directory, epochs=30, early_stop_patience=2,
                                   per_gpu_batch_size=4, grad_accum_steps=1, precision="fp32", weight_decay=0)
            model = TinyModel()
            model.runtime_config = config
            qwen = SimpleNamespace(layers=[object(), object()], config=SimpleNamespace(hidden_size=1, num_hidden_layers=2))
            model.encoder = SimpleNamespace(shared_qwen_model=lambda: qwen)
            model.lgled = SimpleNamespace(selected_qwen_layers=lambda q: q.layers, latent_judge_num_layers=2,
                                          runtime=lambda q: SimpleNamespace(first_shared_layer=0, last_shared_layer=1))
            class Loader(list):
                sampler = None
            built_splits = []
            def build_loader(project, cfg, split, processor, **kwargs):
                built_splits.append(split)
                return Loader(loader())
            def loss(outputs, labels, *args):
                # Exercise real optimizer/checkpoint control flow with a constant
                # validation curve, without touching any real model or dataset.
                return outputs["logits"].sum() * 0 + 1, {"fixture_loss": 1.}
            context = SimpleNamespace(device=torch.device("cpu"), is_main=True, distributed=False,
                                      rank=0, local_rank=0, world_size=1)
            args = ["train.py", "--dataset", "fixture", "--config", "unused.json",
                    "--manifest-dir", "unused", "--run-name", "fixture_run"]
            with contextlib.ExitStack() as stack:
                for name, replacement in (("load_config", lambda p: config),
                                           ("bind_dataset_workspace", lambda *a: root / "fixture_manifest"),
                                           ("init_distributed", lambda: context), ("cleanup_distributed", lambda: None),
                                           ("build_processor", lambda *a: None), ("build_loader", build_loader),
                                           ("build_model", lambda c: model), ("multimodal_loss", loss),
                                           ("tqdm", lambda value, **kw: value)):
                    stack.enter_context(patch.object(train, name, replacement))
                stack.enter_context(patch.object(sys, "argv", args))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                train.main()
            run = root / "fixture_run"
            history = json.loads((run / "history.json").read_text())
            self.assertEqual(len(history), 3, "Two bad epochs after epoch 1 must trigger patience=2")
            self.assertEqual(built_splits, ["train", "val", "test"], "Test must be built only after training/final selection")
            for row in history:
                self.assertEqual(row["selection_score"], row["val"]["macro_f1"])
                self.assertEqual(row["val"]["threshold_selection"]["checkpoint_reference"]["epoch"], row["epoch"])
                self.assertNotEqual(row["val"]["threshold"], .5)
            for name, epoch in (("best.pth", 1), ("last.pth", 3)):
                checkpoint = torch.load(run / "checkpoints" / name, weights_only=False)
                self.assertEqual(checkpoint["epoch"], epoch)
                checkpoint_threshold_selection(checkpoint)
            final = torch.load(run / "checkpoints/final_averaged.pth", weights_only=False)
            selection = checkpoint_threshold_selection(final)
            self.assertEqual(selection["checkpoint_reference"], {"kind": "top_k_average", "epochs": [1, 2, 3]})
            summary = json.loads((run / "final_summary.json").read_text())
            self.assertEqual(summary["test"]["threshold_selection"], selection)
            self.assertEqual(summary["val_calibrated"]["threshold_selection"], selection)
            self.assertEqual(summary["test"]["accuracy"], summary["test"]["macro_f1"])

    def test_validation_single_pass_and_frozen_test(self):
        model = TinyModel()
        metrics, rows, threshold = val(model)
        self.assertEqual(model.calls, 1, "Threshold grid must reuse one validation inference")
        self.assertEqual([r["label"] for r in rows], [0, 0, 1, 1])
        self.assertEqual([r["prediction"] for r in rows], [0, 0, 1, 1])
        self.assertTrue(all(r["positive_label"] == 0 for r in rows))
        self.assertEqual(metrics["accuracy"], 1)
        self.assertEqual(metrics["threshold_source"], "validation")
        with patch("mmfnd.engine.find_best_macro_f1_threshold", side_effect=AssertionError("test leakage")):
            test, predictions, used = evaluate(
                model, loader(labels=(1, 1, 0, 0)), torch.device("cpu"), 0, {0: "fake", 1: "real"},
                show_progress=False, split="test", threshold_selection=metrics["threshold_selection"],
            )
        self.assertEqual(threshold, used)
        self.assertEqual([r["prediction"] for r in predictions], [r["prediction"] for r in rows])
        self.assertEqual(test["accuracy"], 0)

    def test_test_without_provenance_or_with_tuning_fails_before_forward(self):
        model = TinyModel()
        for kwargs in ({}, {"tune_threshold": True, "checkpoint_reference": {"kind": "epoch", "epoch": 1}}):
            with self.assertRaises(ValueError):
                evaluate(model, loader(), torch.device("cpu"), 0, {0: "fake", 1: "real"}, split="test", **kwargs)
        self.assertEqual(model.calls, 0)

    def test_checkpoints_preserve_threshold_and_reject_wrong_epoch(self):
        model = TinyModel()
        metrics, _, threshold = val(model)
        with tempfile.TemporaryDirectory(prefix="v3-eval-test-") as directory:
            path = Path(directory) / "paired.pth"
            save_checkpoint(path, model, None, None, 2, metrics, model.runtime_config)
            checkpoint = load_checkpoint(path, model, torch.device("cpu"))
            self.assertEqual(checkpoint_threshold_selection(checkpoint)["threshold"], threshold)
            checkpoint["epoch"] = 3
            with self.assertRaisesRegex(ValueError, "epochs"):
                checkpoint_threshold_selection(checkpoint)
            with self.assertRaisesRegex(ValueError, "mismatched"):
                save_checkpoint(path, model, None, None, 2, metrics, model.runtime_config, decision_threshold=.99)
            with self.assertRaises(ValueError):
                checkpoint_threshold_selection({"decision_threshold": .5})

    def test_averaged_model_has_its_own_threshold(self):
        model = TinyModel()
        reference = {"kind": "top_k_average", "epochs": [2, 4]}
        metrics, _, _ = val(model, reference=reference)
        with tempfile.TemporaryDirectory(prefix="v3-eval-test-") as directory:
            path = Path(directory) / "averaged.pth"
            with self.assertRaisesRegex(ValueError, "Averaged"):
                save_checkpoint(path, model, None, None, 6, metrics, model.runtime_config)
            save_checkpoint(path, model, None, None, 6, metrics, model.runtime_config,
                            training_state={"averaged_checkpoints": [{"epoch": 2}, {"epoch": 4}]})
            checkpoint = load_checkpoint(path, model, torch.device("cpu"))
            self.assertEqual(checkpoint_threshold_selection(checkpoint)["checkpoint_reference"], reference)

    def test_independent_seeds_can_choose_different_thresholds(self):
        first, _, _ = val(TinyModel())
        second, _, _ = val(TinyModel(), loader(fake=(.9, .85, .8, .75)))
        self.assertNotEqual(first["threshold"], second["threshold"])

    def test_aggregate_provenance_and_no_legacy_mixing(self):
        import aggregate_seeds
        model = TinyModel()
        validation, _, _ = val(model)
        test, _, _ = evaluate(model, loader(), torch.device("cpu"), 0, {0: "fake", 1: "real"},
                              show_progress=False, split="test", threshold_selection=validation["threshold_selection"])
        with tempfile.TemporaryDirectory(prefix="v3-aggregate-test-") as directory:
            root = Path(directory)
            runs = []
            for seed in range(4):
                run = root / f"seed{seed}"
                values = {
                    "run_info.json": {"dataset": "fixture", "seed": seed, "epochs": 30, "early_stop_patience": 8, "evaluation_protocol": PROTOCOL},
                    "evaluation/final/val/metrics.json": validation,
                    "evaluation/final/test/metrics.json": test,
                    "final_summary.json": evaluation_info(test),
                }
                for filename, value in values.items():
                    path = run / filename
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(json.dumps(value))
                runs.append(str(run))
            args = ["aggregate_seeds.py", "--output", str(root / "aggregate.json"), *runs]
            with patch.object(sys, "argv", args), contextlib.redirect_stdout(io.StringIO()):
                aggregate_seeds.main()
            result = json.loads((root / "aggregate.json").read_text())
            self.assertEqual(result["runs"], 4)
            self.assertEqual(result["aggregate"]["accuracy"], {"mean": 1, "sample_std": 0})
            for name in ("per_seed_metrics.csv", "aggregate_metrics.csv"):
                self.assertTrue((root / name).is_file())
            bad = copy.deepcopy(test)
            bad.pop("evaluation_protocol")
            (Path(runs[0]) / "evaluation/final/test/metrics.json").write_text(json.dumps(bad))
            with patch.object(sys, "argv", args), self.assertRaisesRegex(ValueError, "legacy/mixed"):
                aggregate_seeds.main()


if __name__ == "__main__":
    unittest.main(verbosity=2)
