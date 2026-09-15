# Masked R1 implementation

## Model and quantities

`qwen_lora_lgled_masked_r1` is an explicit factory branch. The existing `qwen_lora_lgled_v3` branch and original configs remain legacy. The real workspace name is `sotamodelv3_mask`, not `sotamodelv3_masked`; remote and per-layer implementation establish its identity.

The existing encoder constructs T=text, V=vision, E=intrinsic_feature, X=interaction. E and X already depend on text and vision. These are four internal representations, not four independent external evidence sources. Qwen text encoding and the last-two-layer 11-token deliberation share the same Qwen/LoRA parameters. LoRA remains q_proj/v_proj across all Qwen layers. Hidden size comes from config; default evidence D=384.

The strict sequence is `[T,V,E,X,TV,TE,TX,VE,VX,EX,G]`. Every selected layer receives a per-example additive mask: evidence reads itself; pair reads its endpoints and itself; G reads valid evidence/pairs/self. Invalid queries have self loops; no valid query reads an invalid token. Empty main-model inputs fail. `latent_view_mask` deletes already-generated representations; `raw_modality_intervention` changes input tensors and regenerates E/X.

Reference singleton Fake probabilities define r=2q-1 and t=[max(r_i r_j,0),1-|r_i r_j|,max(-r_i r_j,0)]. These are reference-task agreement/ambiguity/conflict, not human fact relations. Two wrong views may agree. A does not mean correct; M does not mean vacuity.

Relation evidence is softplus(raw logits), alpha=e+1, P=alpha/S, U=3/S, b=(alpha-1)/S. `relation_vacuity` is the main name; `relation_uncertainty` is a compatibility alias only. Conflict belief is b_C=P_C-U/3, not P_C(1-U). All evidence/KL/digamma/loss arithmetic is FP32. The loss is the explicitly designed soft-target extension of supervised EDL: each class-specific concentration has that class replaced by 1, and its digamma loss + beta*Dirichlet KL is weighted by t_r. It is not an unchanged formula from the EDL paper and does not establish calibration of U.

Reference subset masks use integer bits T=1,V=2,E=4,X=8. All 16 probabilities are generated after one encoder call per input batch. q(empty) is the smoothed reference_fit class prior, never an untrained zero-input prediction. Signed LOO Delta and exact Shapley phi use true-class NLL (epsilon=1e-7). Contributions are recomputed on the available-player subgame. phi is divided by OOF-train RMS with a 1e-6 floor; no centering, absolute value or sigmoid is used. Unsigned TV is saved only as a diagnostic.

A single shared EvidenceUtilityHead consumes normalized evidence/global states, detached masked incident means of [P_A,P_M,P_C,U], pair-count/3 and role identity. Uniform prior gives w=masked_softmax(u/tau), default tau=1. Aggregation is exactly one weighted sum of original D-dimensional views, then LayerNorm/Dropout/Linear(D,2). No confidence/minority/adjudication/direct/global-preference/IURD heads are instantiated in R1. There is no conflict subtraction or fusion residual.

## Gradient table

| Loss | Updated paths | Blocked dedicated parameters |
|---|---|---|
| CE | classifier and encoder through original z | utility and relation heads through weights |
| SmoothL1 utility | utility, evidence/global states and shared parameters | relation-head exclusive parameters through detached P/U |
| EDL relation | relation, corresponding pairs, projector/shared Qwen | offline teacher |
| Reference CE | reference_fit encoder and subset probe only | student and target-fold labels |

Default total = CE + .2 L_relation + 1 L_utility; beta ramps from 0 to .01 over 3 epochs. These are unvalidated starting values. Utility's final linear layer has small nonzero initialization; gradient checks span several steps. `allow_task_grad_into_routing` is an explicit alternate control.

## Controls implemented

Configs under `configs/revisions/v3_masked_r1/ablations` cover encoder mean, ordinary encoder scorer, no Global, no explicit P/U, no explicit U, supervised softmax, uniform fusion, MLP judge with the same targets, signed LOO, task-routing gradients, task-trained matched scorer, causal visibility, and legacy matched replay. Uniform fusion removes the unused utility head and Global token. Encoder-only controls instantiate no deliberation/relation heads and do not require an OOF cache. Supervised softmax applies softmax to raw logits and has no alpha/S/U output.

`frozen_judge_controls.py` plus export/train CLIs provide fixed-Z pretrained/random judge comparisons with full training of the same last two judge layers and heads. The text encoder is outside that experiment and is never randomized. A cache-export encoder must be chosen at a predeclared train epoch; provenance is recorded, and the caller's predeclaration cannot itself be verified by code. This is a conditional fixed-input comparison, not the shared end-to-end model. Dataset-wide seeded role permutations preserve marginals and work with inference batch_size=1. They are explicitly eval-only sensitivity, not a retrained ablation.

## Diagnostic scope

Student predictions are written without labels before minority grouping and loss-based analysis. Groups use independent reference singleton predictions at .5, never student utility. Reports include counts, labels/roles, small/empty group flags, accuracy/NLL/eligible Macro-F1, weights, phi, utility, actual student latent-deletion effects; utility MAE/Spearman/sign coverage/top-1; and relationship soft CE/Brier/confusion/risk-coverage using U, entropy and 1-max(P). Small single-class groups produce explicit undefined AUC warnings, not fabricated values.

Existing Gaussian image noise and random token-typo perturbations are exposed in the R1 diagnostics CLI. They re-encode inputs and E/X. The legacy typo procedure protects special tokens but does not guarantee preservation of people, dates or other entities; its label-preservation limitation is reported. Robustness grouping remains clean-reference grouping and is explicitly sensitivity analysis, not matched corrupted-target training.

## Sources and claim limits

The [EDL paper](https://arxiv.org/abs/1806.01768), [predictive-power feature contribution work](https://arxiv.org/abs/2004.00668), and [temperature-scaling paper](https://proceedings.mlr.press/v70/guo17a.html) motivate underlying components. They do not validate the R1 combination, its soft relation target, or any benchmark improvement.
