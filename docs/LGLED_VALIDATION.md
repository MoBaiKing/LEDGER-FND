# LG-LED implementation and validation report

Validation date: 2026-09-09. Hardware: four NVIDIA L20 GPUs with 46,068 MiB each. Environment: Python 3.10, PyTorch 2.4.1+cu124, Transformers 4.46.3, PEFT 0.13.2.

## Audited v2 path

The removed `RelationalEvidenceGraph` stacked `text`, `vision`, `intrinsic_feature`, and `interaction` as `[B,4,384]`, added node-type parameters, built cosine-similarity adjacency, ran one `adjacency @ nodes` update, attentively read out `[B,384]`, and injected a bounded residual before the calibrated two-class decision. Graph adjacency, edge entropy, node weights, graph ambiguity, graph configuration, plotting, and category graph export were removed.

## Actual v3 backbone and sharing

- Repository: `Qwen/Qwen2.5-7B-Instruct`.
- Expected project-relative path: `pretrained_models/Qwen2.5-7B-Instruct`.
- Loaded model class: `Qwen2Model` through `AutoModel` plus `PeftModelForFeatureExtraction`.
- Hidden size: 3,584; decoder layers: 28; attention heads: 28; KV heads: 4.
- LoRA: r=8, alpha=16, dropout=0.05, targets=`q_proj,v_proj`.
- Latent judge: original layer objects 26 and 27; identity test passed.
- The LoRA modules seen by the judge are the original adapters; identity/sharing test passed.
- LG-LED never owns or loads a Qwen module. State dict audit found one Qwen namespace and no `lgled.qwen.*` copy.

## Shapes and parameters

The real BF16 B=1 test produced `[1,4,384]` evidence, `[1,4,3584]` projected evidence, `[1,7,3584]` judge embeddings, `[1,11,3584]` latent input/output, `[1,6,3]` relation beliefs, `[1,6]` relation uncertainty, all four-evidence diagnostics as `[1,4]`, fused evidence `[1,384]`, and logits `[1,2]`.

| Component | Parameters |
|---|---:|
| Loaded shared Qwen module including LoRA | 7,073,142,272 |
| Frozen Qwen base | 7,070,619,136 |
| Qwen LoRA trainable | 2,523,136 |
| One shared projector | 1,387,008 |
| Role embedding | 14,336 |
| Seven judge tokens | 25,088 |
| Evidential relation head | 10,755 |
| Confidence head | 925,185 |
| Critical-minority head | 919,041 |
| Global judge head | 14,340 |
| Adjudication head | 919,297 |
| Direct fusion head | 386 |
| Complete LG-LED including fusion norm/residual | 4,216,205 |
| Complete v3 model | 7,174,226,011 |
| Complete v3 trainable | 24,899,931 |

Four independent projectors would contain 5,548,032 parameters. The default single shared projector saves 4,161,024 parameters and is the only projector instantiated in the main configuration.

## Executed tests

- Tiny-Qwen2 B=1 forward/backward API test: passed.
- Real Qwen2.5-7B + SigLIP + Weibo21 B=1 BF16 forward/backward: passed.
- Shared projector, role, judge, relation, confidence, minority, global, adjudication, direct-fusion, and LoRA gradients: all finite and present.
- Frozen Qwen base gradients: absent as required.
- All checked latent/evidential/routing/logit/loss tensors: finite.
- Compact checkpoint save/load round trip: passed.
- Four-GPU DDP one-step smoke: passed.
- Four-GPU DDP 20-optimizer-step training smoke: passed.
- GossipCop, Weibo21, Twitter, and Weibo first-batch manifest/token/image decoding: passed.
- JSONL case-study diagnostics export: passed.

Single-GPU B=1 allocated memory was 14,607,316,480 bytes before forward, 14,709,963,776 after forward, 14,728,298,496 after backward, and 14,771,035,136 peak. Text-Qwen forward was 0.382 s, the complete LG-LED call was 0.023 s, and whole-model forward was 0.664 s.

The 20-step four-rank peaks were 15,264,192,512; 15,318,716,928; 15,507,271,168; and 15,277,582,336 bytes. Mean relation belief was agreement=0.411, ambiguity=0.211, conflict=0.377; mean strength=6.234, mean uncertainty=0.513, mean routing gate=0.230. No initial all-uniform/single-class relation collapse, evidential-strength explosion, or routing-to-zero/one collapse was observed. This short smoke is a runtime check, not a convergence or accuracy claim.

## Current risks

- Relation and minority quantities have no pair/minority ground truth and learn only through Fake/Real classification; interpret them as learned latent beliefs.
- Last-layer reuse calls the tested Transformers 4.46.3 Qwen2 decoder API directly; the dependency is pinned because later private API changes require a new smoke test.
- A full dataset epoch was not used as a validation shortcut; run the documented seed protocol before reporting accuracy or macro-F1.
- Standard DDP replicates the frozen 7B weights on every GPU. It fits the tested L20 configuration; reduce sequence length/accumulated microbatch pressure before adding FSDP or ZeRO.
