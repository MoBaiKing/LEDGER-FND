"""CPU behavioral regression tests using an in-memory tiny Qwen (no downloads)."""
from copy import deepcopy
from contextlib import redirect_stdout
import io
import json
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen2Config, Qwen2Model

from ablation.entrypoint import install_contract
from ablation.model import AblationDeliberation, AblationMMFND, MLPClassificationModule
from ablation.registry import VARIANTS, CONTROLS, ROOT, resolve_config, settings, states
from ablation.summarize import freeze_subsets, stats
from mmfnd import engine
from mmfnd.evaluation import PROTOCOL
from mmfnd.latent_evidence_deliberation import LLMGuidedLatentEvidenceDeliberation
from mmfnd.model import ExplainableMMFND, multimodal_loss


def tiny_qwen():
    return Qwen2Model(Qwen2Config(
        vocab_size=32, hidden_size=32, intermediate_size=48,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0, use_cache=False,
    ))


def config_for(exp):
    base = json.loads((ROOT / "configs/datasets/weibo21.json").read_text())
    base["model"].update(hidden_dim=16, num_heads=4)
    base["model"]["lgled"].update(head_hidden_dim=8, dropout=0.0)
    return resolve_config(base, exp, 17, "/tmp/ablation-test-unused", non_llm={"hidden_dim": 16, "num_heads": 4, "feedforward_dim": 32, "dropout": 0.0})


def deliberation(exp):
    cfg = config_for(exp)
    module = LLMGuidedLatentEvidenceDeliberation(16, 32, cfg["model"]["lgled"])
    module.__class__ = AblationDeliberation
    module.configure(cfg)
    return module.eval()


class TinyEncoder(nn.Module):
    """Only replaces expensive backbones; real reasoner/decision/loss execute."""
    def __init__(self, cfg, dim, num_heads):
        super().__init__()
        self.qwen = tiny_qwen()
        self.projection = nn.Linear(32, dim)

    def shared_qwen_model(self):
        return self.qwen

    def forward(self, batch, ablate_component=None):
        states = self.qwen(input_ids=batch["input_ids"]).last_hidden_state
        evidence = self.projection(states)
        values = dict(zip(("text", "vision", "intrinsic_feature", "interaction"), evidence.unbind(1)))
        zero = evidence[:, 0, 0] * 0
        values["intrinsic"] = {key: zero for key in (
            "text_relation_ambiguity", "event_contradiction", "visual_evidence_inconsistency",
            "multi_image_dispersion", "alignment_attention")}
        values["intrinsic"]["multi_image_available"] = torch.zeros_like(zero, dtype=torch.bool)
        values["per_image"] = evidence[:, 1:2]
        for key in ("event_matched_similarity", "event_mismatched_similarity",
                    "source_matched_similarity", "source_mismatched_similarity"):
            values[key] = zero
        values["source_ranking_valid"] = torch.zeros_like(zero, dtype=torch.bool)
        return values


class AblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(13)
        self.qwen = tiny_qwen().eval()
        self.features = [torch.randn(2, 16) for _ in range(4)]
        self.batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), "labels": torch.tensor([0, 1])}

    def test_all_variants_masks_finite_and_backward(self):
        mask = torch.tensor([[True, False, True, False], [True, False, False, False]])
        for name in tuple(name for name in VARIANTS + CONTROLS if name != "mlp_classifier"):
            with self.subTest(variant=name):
                module = deliberation(name)
                features = [x.clone().requires_grad_() for x in self.features]
                output = module(features, self.qwen, mask)
                self.assertTrue(all(torch.isfinite(x).all() for x in output.values()))
                output["fused_feature"][:, 0].sum().backward()
                for index, feature in enumerate(features):
                    self.assertTrue(torch.isfinite(feature.grad).all())
                    self.assertEqual(feature.grad[~mask[:, index]].count_nonzero(), 0)
                torch.testing.assert_close(output["final_evidence_weights"].sum(1), torch.ones(2))
                self.assertEqual(output["final_evidence_weights"][~mask].count_nonzero(), 0)
                with self.assertRaises(ValueError):
                    module(features, self.qwen, torch.zeros_like(mask))

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_full_exact_output_loss_and_optimizer_step(self):
        from ablation.loss import compute_loss
        cfg = config_for("full")
        torch.manual_seed(7); original = ExplainableMMFND(deepcopy(cfg)).eval()
        torch.manual_seed(7); full = AblationMMFND(deepcopy(cfg)).eval()
        before, after = original(self.batch), full(self.batch)
        for key in before:
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)
        loss, _ = multimodal_loss(before, self.batch["labels"], cfg["loss"], cfg["train"]["label_smoothing"])
        candidate, _ = compute_loss(after, self.batch["labels"], cfg)
        torch.testing.assert_close(loss, candidate, rtol=0, atol=0)
        optimizer1 = torch.optim.AdamW(original.parameters(), lr=1e-4)
        optimizer2 = torch.optim.AdamW(full.parameters(), lr=1e-4)
        loss.backward(); candidate.backward(); optimizer1.step(); optimizer2.step()
        for (name, left), (_, right) in zip(original.named_parameters(), full.named_parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0, msg=name)
        default_cfg = deepcopy(cfg); default_cfg.pop("ablation")
        self.assertEqual(AblationMMFND(default_cfg).runtime_config["ablation"]["name"], "full")

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_shared_initialization_and_rng(self):
        torch.manual_seed(9); full = AblationMMFND(config_for("full"))
        reference = dict(full.named_parameters()); following = torch.rand(4)
        for name in VARIANTS + CONTROLS:
            torch.manual_seed(9); candidate = AblationMMFND(config_for(name))
            torch.testing.assert_close(torch.rand(4), following, rtol=0, atol=0)
            for key, value in candidate.named_parameters():
                if key in reference and reference[key].shape == value.shape:
                    torch.testing.assert_close(value, reference[key], rtol=0, atol=0, msg=name + key)

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_strict_direct_and_whole_mlp_classifier_isolation(self):
        for name in ("direct_only", "path_direct_reference"):
            model = AblationMMFND(config_for(name)).eval()
            self.assertFalse(hasattr(model.lgled, "relation_head"))
            self.assertIsNone(model.decision.calibrator)
            with patch.object(model.lgled, "_run_qwen_last_layers", side_effect=AssertionError("latent Qwen must not run")):
                output = model(self.batch)
            torch.testing.assert_close(output["logits"], output["preliminary_logits"], rtol=0, atol=0)
        self.assertTrue(hasattr(model.lgled, "direct_fusion_head"))
        simple = AblationMMFND(config_for("mlp_classifier")).eval()
        self.assertIsNone(simple.reasoner)
        self.assertEqual(list(simple.lgled.parameters()), [])
        self.assertEqual(simple.lgled.selected_qwen_layers(simple.encoder.shared_qwen_model()), [])
        self.assertIsInstance(simple.decision, MLPClassificationModule)
        self.assertFalse(hasattr(simple.decision, "final_norm"))
        self.assertFalse(hasattr(simple.decision, "classifier"))
        output = simple(self.batch)
        torch.testing.assert_close(output["logits"], output["preliminary_logits"], rtol=0, atol=0)
        self.assertEqual(output["final_evidence_weights"].count_nonzero(), 0)
        torch.manual_seed(19)
        direct = AblationMMFND(config_for("direct_only")).eval()(self.batch)["logits"]
        torch.manual_seed(19)
        alias = AblationMMFND(config_for("path_direct_reference")).eval()(self.batch)["logits"]
        torch.testing.assert_close(direct, alias, rtol=0, atol=0)

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_cpu_bf16_finite_and_full_equivalence(self):
        cfg = config_for("full")
        torch.manual_seed(21); original = ExplainableMMFND(deepcopy(cfg)).eval()
        torch.manual_seed(21); full = AblationMMFND(deepcopy(cfg)).eval()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            left, right = original(self.batch), full(self.batch)
        torch.testing.assert_close(left["logits"], right["logits"], rtol=0, atol=0)
        for name in VARIANTS:
            model = AblationMMFND(config_for(name)).eval()
            with torch.autocast("cpu", dtype=torch.bfloat16):
                output = model(self.batch)
            self.assertTrue(torch.isfinite(output["logits"]).all(), name)

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_uncertainty_perturbation_cannot_change_logits(self):
        model = AblationMMFND(config_for("no_uncertainty")).eval()
        original = model.lgled.relation_head.forward
        def perturbed(states):
            result = original(states)
            result["relation_uncertainty"] = torch.rand_like(result["relation_uncertainty"]) * 100
            return result
        expected = model(self.batch)["logits"]
        with patch.object(model.lgled.relation_head, "forward", side_effect=perturbed):
            actual = model(self.batch)
        torch.testing.assert_close(expected, actual["logits"], rtol=0, atol=0)
        self.assertEqual(actual["evidence_mean_uncertainty"].count_nonzero(), 0)
        self.assertEqual(actual["uncertainty_components"][:, 3].count_nonzero(), 0)
        self.assertTrue((actual["relation_alpha"] >= 1).all())

    def test_global_token_removed_and_masked_pooling(self):
        module = deliberation("no_global_token")
        lengths = []
        hook = self.qwen.layers[-1].register_forward_pre_hook(lambda mod, args: lengths.append(args[0].shape[1]))
        mask = torch.tensor([[True, True, False, False], [True, False, False, False]])
        output = module(self.features, self.qwen, mask)
        hook.remove()
        self.assertEqual(lengths, [10])
        self.assertEqual(module.judge_tokens.embedding.num_embeddings, 6)
        pairs = output["latent_output"][:, 4:10]
        expected = torch.stack([pairs[0, 0], torch.zeros_like(pairs[1, 0])])
        torch.testing.assert_close(output["global_judge_logits"], module.global_judge_head(expected).float())
        self.assertEqual(output["latent_sequence"].shape[1], 10)

    def test_non_llm_strict_mask_slots_and_no_qwen(self):
        from ablation.model import SmallCausalEvaluator
        module = deliberation("non_llm_evaluator")
        with patch.object(module, "_run_qwen_last_layers", side_effect=AssertionError("latent Qwen used")):
            output = module(self.features, None)
        self.assertEqual(output["latent_sequence"].shape[1], 11)
        evaluator = module.evaluator.eval()
        tokens = torch.randn(2, 11, 32)
        before = evaluator(tokens)
        tokens[:, 10] += 100
        after = evaluator(tokens)
        torch.testing.assert_close(before[:, :10], after[:, :10], rtol=0, atol=0)
        mask = SmallCausalEvaluator.causal_mask(11, torch.device("cpu"))
        self.assertFalse(mask.all(1).any())
        visible_slots = [
            [0], [1], [2], [3],
            [0, 1, 4], [0, 2, 5], [0, 3, 6],
            [1, 2, 7], [1, 3, 8], [2, 3, 9], list(range(11)),
        ]
        for row, expected in enumerate(visible_slots):
            self.assertEqual((~mask[row]).nonzero().flatten().tolist(), expected)

    def test_disabled_heads_are_not_computed_and_routing_is_fixed(self):
        minority = deliberation("no_critical_minority")
        self.assertIsNone(minority.minority_head)
        self.assertEqual(minority(self.features, self.qwen)["minority_score"].count_nonzero(), 0)
        fixed = deliberation("fixed_mixture")
        with patch.object(fixed, "dynamic_gate", side_effect=AssertionError("dynamic gate used")):
            result = fixed(self.features, self.qwen)
        torch.testing.assert_close(result["routing_gate"], torch.full((2, 1), 0.5))
        for name in ("deliberation_only", "path_deliberation_reference"):
            module = deliberation(name)
            self.assertIsNone(module.direct_fusion_head)
            self.assertFalse(hasattr(module, "fusion_residual_scale"))
            result = module(self.features, self.qwen)
            torch.testing.assert_close(result["fused_feature"], module.fusion_norm(result["deliberative_feature"]), rtol=0, atol=0)

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_correction_bypass_and_loss_dependencies(self):
        from ablation.loss import compute_loss
        model = AblationMMFND(config_for("no_decision_correction")).eval()
        output = model(self.batch)
        torch.testing.assert_close(output["logits"], output["preliminary_logits"], rtol=0, atol=0)
        cfg = config_for("direct_only")
        output = AblationMMFND(cfg).eval()(self.batch)
        output.pop("relation_strength"); output.pop("uncertainty")
        _, parts = compute_loss(output, self.batch["labels"], cfg)
        self.assertEqual(set(parts), {"classification"})
        cfg["loss"]["contrastive"] = 0.2
        _, parts = compute_loss(output, self.batch["labels"], cfg)
        self.assertIn("contrastive", parts)
        bad = config_for("direct_only"); bad["loss"]["evidential_regularization"] = 1
        with self.assertRaisesRegex(ValueError, "Disabled loss"):
            AblationMMFND(bad)
        bad = config_for("mlp_classifier"); bad["loss"]["causal_probe_classification"] = 1
        with self.assertRaisesRegex(ValueError, "Disabled loss"):
            AblationMMFND(bad)
        bad = config_for("non_llm_evaluator"); bad["ablation"]["non_llm_evaluator"]["hidden_dim"] = 15
        with self.assertRaisesRegex(ValueError, "divisible"):
            AblationMMFND(bad)

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_all_training_loss_graphs(self):
        for name in VARIANTS + CONTROLS:
            model = AblationMMFND(config_for(name)).train()
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                output = model(self.batch)
                self.assertFalse(output["logits"].requires_grad)
                output["training_loss"].backward()
                self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                optimizer.step()

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_checkpoint_isolation_and_roundtrip(self):
        original = engine._checkpoint_contract
        install_contract()
        try:
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "model.pth"
                model = AblationMMFND(config_for("fixed_mixture")).eval()
                engine.save_checkpoint(path, model, None, None, 0, {}, model.runtime_config)
                restored = AblationMMFND(config_for("fixed_mixture")).eval()
                engine.load_checkpoint(path, restored, "cpu")
                torch.testing.assert_close(model(self.batch)["logits"], restored(self.batch)["logits"], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "contract mismatch"):
                    engine.load_checkpoint(path, AblationMMFND(config_for("full")), "cpu")
                other_seed = config_for("fixed_mixture"); other_seed["seed"] += 1
                with self.assertRaisesRegex(ValueError, "ablation_seed"):
                    engine.load_checkpoint(path, AblationMMFND(other_seed), "cpu")
        finally:
            engine._checkpoint_contract = original

    def test_training_seed_source_and_partial_table(self):
        from ablation.run import training_source, write_json
        from ablation.summarize import summarize_suite
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            config_path = folder / "config.json"
            config_path.write_text(json.dumps(config_for("full")))
            source = folder / "plan.json"
            seeds = [1234567, 9876543]
            source.write_text(json.dumps({"datasets": ["weibo21"], "seeds": seeds,
                "jobs": [{"dataset": "weibo21", "seed": seed, "command": ["train.py", "--config", str(config_path), "--epochs", "3"]} for seed in seeds]}))
            base, imported, info = training_source("weibo21", source)
            self.assertEqual(imported, seeds)
            self.assertEqual(base["train"]["epochs"], 3)
            protocol = {"dataset": "weibo21", "seeds": seeds, "smoke_steps": 0,
                        "subset_definition": {"reference_seed": seeds[0], "fraction": 0.25}}
            write_json(folder / "protocol.json", protocol)
            write_json(folder / "runs.json", {f"full/seed{s}": {"dataset": "weibo21", "variant": "full", "seed": s, "status": "planned"} for s in seeds})
            report = summarize_suite(folder)
            results = json.loads((report / "results.json").read_text())
            self.assertTrue(all(r["n"] == 0 for r in results["aggregate"]))
            self.assertEqual(len(results["missing_failed"]), 2)
            self.assertIn("0/2", (report / "results.md").read_text())

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_actual_training_evaluation_and_report_pipeline(self):
        import contextlib
        from types import SimpleNamespace
        import train
        from ablation.entrypoint import configure_entry
        from ablation.run import write_json
        from ablation.summarize import summarize_suite
        from ablation.registry import config_hash
        class Loader(list):
            sampler = None
        batch = {**self.batch, "ids": ["fake", "real"], "image_paths": [["fixture"], ["fixture"]],
                 "texts": ["fixture", "fixture"], "categories": ["", ""]}
        context = SimpleNamespace(device=torch.device("cpu"), is_main=True, distributed=False,
                                  rank=0, local_rank=0, world_size=1)
        with tempfile.TemporaryDirectory() as directory:
            suite = Path(directory)
            protocol = {"dataset": "weibo21", "seeds": [17, 18], "smoke_steps": 0,
                        "subset_definition": {"reference_seed": 17, "fraction": 0.25}}
            write_json(suite / "protocol.json", protocol)
            records = {}
            for name in VARIANTS:
                cfg = config_for(name)
                cfg["train"].update(output_dir=str(suite / name), epochs=2, early_stop_patience=8,
                                    per_gpu_batch_size=2, grad_accum_steps=1, precision="fp32", top_k_checkpoint_average=1)
                cfg["ablation_provenance"] = protocol
                cfg["ablation_resolved_config_hash"] = config_hash(cfg)
                for seed in protocol["seeds"]:
                    records[f"{name}/seed{seed}"] = {"dataset": "weibo21", "variant": name, "seed": seed, "status": "planned"}
                replacements = {
                    "load_config": lambda path, cfg=cfg: cfg,
                    "init_distributed": lambda: context, "cleanup_distributed": lambda: None,
                    "bind_dataset_workspace": lambda *args: suite,
                    "build_processor": lambda *args: None,
                    "build_loader": lambda *args, **kwargs: Loader([batch]),
                    "tqdm": lambda value, **kwargs: value,
                }
                with contextlib.ExitStack() as stack:
                    for key, replacement in replacements.items():
                        stack.enter_context(patch.object(train, key, replacement))
                    for key in ("ExplainableMMFND", "evaluate", "DDP", "multimodal_loss", "reduce_training_stats", "save_checkpoint", "load_checkpoint"):
                        stack.enter_context(patch.object(train, key, getattr(train, key)))
                    stack.enter_context(patch.object(engine, "_checkpoint_contract", engine._checkpoint_contract))
                    stack.enter_context(patch("torch.cuda.is_available", return_value=False))
                    stack.enter_context(patch.object(sys, "argv", ["train.py", "--dataset", "weibo21", "--config", "fixture.json",
                                                                  "--manifest-dir", str(suite), "--run-name", "seed17"]))
                    stack.enter_context(redirect_stdout(io.StringIO()))
                    stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    configure_entry(train, "train")
                    train.main()
                run = suite / name / "seed17"
                self.assertTrue((run / "final_summary.json").is_file())
                last = torch.load(run / "checkpoints/last.pth", weights_only=False)
                self.assertEqual(len(last["ablation_rng_states"]), 1)
                records[f"{name}/seed17"]["status"] = "completed"
            write_json(suite / "runs.json", records)
            with contextlib.redirect_stderr(io.StringIO()):
                output = summarize_suite(suite)
            report = json.loads((output / "results.json").read_text())
            self.assertEqual(len(report["missing_failed"]), 10)
            self.assertTrue(all(row["n"] == 1 for row in report["aggregate"] if row["subset"] == "all"))
            self.assertTrue(report["paired"])
            for item in report["paired"]:
                if item["metric"] == "macro_f1":
                    self.assertAlmostEqual(item["delta_macro_f1_pp"], item["delta_macro_f1"] * 100)

    def test_rng_roundtrip_and_config_overwrite_guard(self):
        import random
        import numpy as np
        from ablation.entrypoint import rng_state, restore_rng
        from ablation.run import write_once
        with patch("torch.cuda.is_available", return_value=False):
            state = rng_state()
            expected = (random.random(), np.random.random(), torch.rand(3))
            restore_rng(state)
            self.assertEqual(expected[0], random.random())
            self.assertEqual(expected[1], np.random.random())
            torch.testing.assert_close(expected[2], torch.rand(3), rtol=0, atol=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_once(path, {"variant": "full"})
            write_once(path, {"variant": "full"})
            with self.assertRaisesRegex(ValueError, "changed"):
                write_once(path, {"variant": "direct_only"})

    @patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder)
    def test_interrupted_epoch_resume_matches_uninterrupted(self):
        import contextlib
        from types import SimpleNamespace
        import train
        from ablation.entrypoint import configure_entry
        class Loader(list):
            sampler = None
        batch = {**self.batch, "ids": ["fake", "real"], "image_paths": [["fixture"], ["fixture"]],
                 "texts": ["fixture", "fixture"], "categories": ["", ""]}
        context = SimpleNamespace(device=torch.device("cpu"), is_main=True, distributed=False,
                                  rank=0, local_rank=0, world_size=1)
        def run_once(root, interrupt=False, resume=False):
            cfg = config_for("full")
            cfg["model"]["lgled"]["dropout"] = 0.2
            cfg["train"].update(output_dir=str(root), epochs=3, early_stop_patience=8,
                                per_gpu_batch_size=2, grad_accum_steps=1, precision="fp32", top_k_checkpoint_average=1)
            checkpoint = root / "seed17/checkpoints/last.pth"
            with contextlib.ExitStack() as stack:
                for key, value in {"load_config": lambda p: cfg, "init_distributed": lambda: context,
                                   "cleanup_distributed": lambda: None, "bind_dataset_workspace": lambda *a: root,
                                   "build_processor": lambda *a: None, "build_loader": lambda *a, **kw: Loader([batch]),
                                   "tqdm": lambda value, **kwargs: value}.items():
                    stack.enter_context(patch.object(train, key, value))
                for key in ("ExplainableMMFND", "evaluate", "DDP", "multimodal_loss", "reduce_training_stats", "save_checkpoint", "load_checkpoint"):
                    stack.enter_context(patch.object(train, key, getattr(train, key)))
                stack.enter_context(patch.object(engine, "_checkpoint_contract", engine._checkpoint_contract))
                stack.enter_context(patch("torch.cuda.is_available", return_value=False))
                args = ["train.py", "--dataset", "weibo21", "--config", "fixture", "--manifest-dir", str(root), "--run-name", "seed17"]
                if resume:
                    args += ["--resume", str(checkpoint)]
                stack.enter_context(patch.object(sys, "argv", args))
                stack.enter_context(redirect_stdout(io.StringIO()))
                stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                configure_entry(train, "train", checkpoint.resolve() if resume else None)
                if interrupt:
                    save = train.save_checkpoint
                    def stop_after_epoch(path, *args, **kwargs):
                        save(path, *args, **kwargs)
                        if path.name == "last.pth":
                            raise RuntimeError("intentional test interruption")
                    train.save_checkpoint = stop_after_epoch
                    with self.assertRaisesRegex(RuntimeError, "intentional test"):
                        train.main()
                else:
                    train.main()
            return checkpoint
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = run_once(root / "baseline")
            run_once(root / "resumed", interrupt=True)
            resumed = run_once(root / "resumed", resume=True)
            expected = torch.load(baseline, weights_only=False)
            actual = torch.load(resumed, weights_only=False)
            self.assertEqual(actual["epoch"], 3)
            for key in expected["model_state_dict"]:
                torch.testing.assert_close(actual["model_state_dict"][key], expected["model_state_dict"][key], rtol=0, atol=0, msg=key)


def ddp_smoke():
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo")
    try:
        batch = {"input_ids": torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]), "labels": torch.tensor([0, 1])}
        with patch("mmfnd.model.MultimodalIntrinsicEvidenceEncoder", TinyEncoder):
            for name in VARIANTS + CONTROLS:
                model = DistributedDataParallel(AblationMMFND(config_for(name)), find_unused_parameters=True, broadcast_buffers=False)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with model.no_sync():
                        (model(batch)["training_loss"] / 2).backward()
                    (model(batch)["training_loss"] / 2).backward()
                    optimizer.step()
                if dist.get_rank() == 0:
                    print(f"DDP {name}: two accumulated optimizer steps passed", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    if sys.argv[1:] == ["--ddp"]:
        ddp_smoke()
    else:
        unittest.main()
