"""Check actual multi-layer information flow, without downloading backbones."""
import unittest

import torch
from transformers import Qwen2Config, Qwen2Model

from ablation.model import SmallCausalEvaluator
from mmfnd.latent_evidence_deliberation import (
    LLMGuidedLatentEvidenceDeliberation, PAIR_EVIDENCE_INDICES,
)


class StrictLatentMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(71)

    def make_models(self, backend, depth):
        config = Qwen2Config(
            vocab_size=32, hidden_size=32, intermediate_size=48,
            num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
            max_position_embeddings=64, attention_dropout=0.0, use_cache=False,
        )
        config._attn_implementation = backend
        qwen = Qwen2Model(config).eval()
        module = LLMGuidedLatentEvidenceDeliberation(16, 32, {
            "latent_judge_num_layers": depth, "head_hidden_dim": 8, "dropout": 0.0,
        }).eval()
        return module, qwen

    def check_visibility(self, run, length, dtype=torch.float32):
        tokens = torch.randn(2, length, 32, dtype=dtype, requires_grad=True)
        baseline = run(tokens)
        self.assertTrue(torch.isfinite(baseline).all())
        for query in range(length):
            allowed = ({query} if query < 4 else
                       {query, *PAIR_EVIDENCE_INDICES[query - 4]} if query < 10 else
                       set(range(length)))
            blocked = sorted(set(range(length)) - allowed)
            changed = tokens.detach().clone()
            changed[:, blocked] = torch.randn_like(changed[:, blocked]) * 7
            torch.testing.assert_close(
                run(changed)[:, query], baseline[:, query], rtol=0, atol=0,
                msg=f"forbidden inputs changed slot {query}",
            )
            gradient, = torch.autograd.grad(
                baseline[:, query, 0].sum(), tokens, retain_graph=True,
            )
            self.assertEqual(gradient[:, blocked].count_nonzero().item(), 0)
            for key in allowed:
                self.assertGreater(gradient[:, key].abs().sum().item(), 0,
                                   f"slot {query} cannot read allowed slot {key}")

    def test_qwen_visibility_all_depths_and_backends(self):
        for backend in ("eager", "sdpa"):
            for depth in (1, 2, 4):
                for length in (10, 11):
                    with self.subTest(backend=backend, depth=depth, length=length):
                        module, qwen = self.make_models(backend, depth)
                        self.check_visibility(
                            lambda x: module._run_qwen_last_layers(x, qwen), length,
                        )

    def test_bfloat16_visibility(self):
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                module, qwen = self.make_models(backend, 4)
                qwen.bfloat16()
                self.check_visibility(
                    lambda x: module._run_qwen_last_layers(x, qwen), 11,
                    torch.bfloat16,
                )

    def test_relation_outputs_depend_only_on_corresponding_evidence(self):
        module, qwen = self.make_models("sdpa", 4)
        features = [torch.randn(2, 16, requires_grad=True) for _ in range(4)]
        baseline = module(features, qwen)
        for pair, members in enumerate(PAIR_EVIDENCE_INDICES):
            changed = [x if i in members else torch.randn_like(x) * 7
                       for i, x in enumerate(features)]
            actual = module(changed, qwen)
            for name in ("relation_probs", "relation_alpha", "relation_uncertainty"):
                torch.testing.assert_close(actual[name][:, pair], baseline[name][:, pair],
                                           rtol=0, atol=0)
            gradients = torch.autograd.grad(
                baseline["relation_probs"][:, pair, 0].sum(), features, retain_graph=True,
            )
            for i, grad in enumerate(gradients):
                if i in members:
                    self.assertGreater(grad.abs().sum().item(), 0)
                else:
                    self.assertEqual(grad.count_nonzero().item(), 0)

    def test_small_transformer_uses_same_visibility(self):
        evaluator = SmallCausalEvaluator(32, {
            "hidden_dim": 16, "num_heads": 4, "feedforward_dim": 32,
            "dropout": 0.0, "num_layers": 4,
        }).eval()
        for length in (10, 11):
            self.check_visibility(evaluator, length)


if __name__ == "__main__":
    unittest.main()
