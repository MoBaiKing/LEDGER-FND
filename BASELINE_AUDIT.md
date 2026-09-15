# Masked R1 baseline audit — 2026-09-15

- Requested logical name: sotamodelv3_masked. Actual workspace: /data/dyl/sotamodelv3_mask.
- origin: https://github.com/MoBaiKing/sotamodelv3_; original branch main.
- BASE: 7649c557b5d36bbd205ae353c96590b682a30402; initial git status and diff empty.
- Working branch: feature/v3-masked-r1. No reset, cleanup, data or checkpoint deletion.
- Original tracked code/config snapshot: /data/dyl/sotamodelv3_mask_7649c557_pre_r1.tar.
- Original architecture qwen_lora_lgled_v3 is retained; R1 is explicitly qwen_lora_lgled_masked_r1.
- Strict locality confirmed by code: latent_evidence_deliberation.py:41–58, 325–358 passes additive mask to EVERY selected layer. Existing availability is absent from attention mask.
- One PEFT Qwen; q_proj/v_proj LoRA registered over ALL Qwen layers (model.py:395–405), last TWO reused for deliberation. No role LoRA.
- T=text; V=vision; E=intrinsic_feature depends on text+vision; X=interaction depends on text+vision. Mask guarantees begin at these four vectors.
- Existing last token sum(mask)-1 is correct only for right padding; factory enforces right padding. R1 additionally uses general last-valid index.
- Existing threshold grid [0.20,0.80], step .01; validation Macro-F1 checkpoint selection already present. train.py automatically runs final test: R1 must disable that.
- Existing checkpoint format is trainable-only with version/key checks. R1 will validate exact trainable key set and frozen backbone fingerprints, then merge into full state with strict=True.
- Existing environment (unchanged): torch 2.4.1+cu124, transformers 4.46.3, peft 0.13.2; local .venv. All immediate Python runs have CUDA_VISIBLE_DEVICES=''.
- GPU execution requested for next 01:00 Asia/Shanghai = 2026-09-16 01:00; four NEW seeds, one seed pipeline per GPU, FIFO refill. This overrides attachment's default five seeds for scheduled run; five-seed CLI remains provided.

## Issue tracking (initial execution status NOT RUN)

| Issue | Legacy location | R1 resolution / test group |
|---|---|---|
| A/M exchange symmetry | relation head; multimodal_loss strength.mean | operational soft targets + EDL; 1–6 |
| U confused with correctness / M | relation_uncertainty, IURD | vacuity identities and scope; 4–6,26 |
| P_C(1-U) vs belief | latent_evidence_deliberation forward | b_C=P_C-U/3, diagnostic only; 4–5 |
| unsigned sensitivity / disabled probe CE | CausalReliabilityGate, loss config | offline supervised subset probe; 7–11 |
| minority name without target | CriticalMinorityHead | signed Shapley utility; 7–9,16 |
| pair slots/missing availability | strict mask / forward | per-sample mask every layer; 12–15 |
| compensating multiple scores / dual fusion | LGLED forward | one supervised score, uniform prior; 18–20 |
| detached IURD weights / erased dissent | IntrinsicUncertaintyResidualDisentangler | absent in R1; 18,22 |
| checkpoint/threshold/cost confounds | train/evaluate/engine | locked validation protocol, explicit test; 23–26 |
| E/X not independent causal evidence | intrinsic encoder | separate raw intervention and latent deletion; 9,13 |

Final execution statuses are in docs/MASKED_R1_VALIDATION.md; this audit does not claim PASS.

## 2026-09-15 checkpoint cleanup amendment

After the baseline audit, the user explicitly authorized deleting all old checkpoints in this masked project. Deleted 396 historical training checkpoint files (41,188,183,245 logical bytes), including synthetic R1 training checkpoints. Preserved data, pretrained backbones, source backup, and validation logs. The scheduled 12 jobs use fresh reference/run directories, pretrained-only reference initialization, and no resume argument. Exact deletion inventory: `docs/r1/checkpoint_cleanup_20260915.json`. Earlier checkpoint-preservation statements describe the initial audit, before this authorization.
