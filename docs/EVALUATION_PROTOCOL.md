# Evaluation protocol audit — 2026-09-13

## Scope and label contract

The user's latest instruction takes precedence over the attachment's opposite
label convention: retain **0=Fake, 1=Real**, `positive_label=0`, and
`fake_prob = softmax(logits.float(), dim=-1)[:, 0]` for all current v3 datasets.
Dataset records, class order, logits, loss targets and exported label/prediction
IDs were NOT relabeled. The briefly considered evaluation-layer remapping was
removed before any real experiment started.

No architecture, LG-LED evidence/token construction, Qwen/LoRA, attention mask,
loss, optimizer, data split, learning rate, batch size, maximum epoch or patience
was changed by this evaluation repair. Checkpoint selection is now Macro-F1, as
requested. Top-k weight averaging remains enabled at the existing configured k.

## Findings and changes

| Area | Before | Now |
| --- | --- | --- |
| Threshold objective | Positive/Fake binary F1 | Validation Macro-F1, explicit `average="macro", labels=[0,1], zero_division=0` |
| Threshold ties | Nearest 0.5 on a near-optimal plateau, possibly below the maximum | Exact highest Macro-F1; nearest 0.5; then lower threshold |
| Search grid | 0.20–0.80, step 0.01 | Same 61-point grid; obsolete plateau delta removed |
| Reported Accuracy | Two-logit argmax; F1 used tuned threshold | Same threshold predictions as all discrete metrics |
| Epoch validation | Fixed threshold 0.5 | One inference pass, then Macro-F1 threshold search using cached probabilities |
| Selection/early stop | Macro-F1 at 0.5 for Weibo/Weibo21; AUC for Gossip/Twitter/FineFake/Fakeddit | Tuned validation Macro-F1 everywhere |
| Epoch checkpoints | Threshold defaulted to 0.5 | `best.pth`, each top-k epoch and `last.pth` carry their own validation threshold and epoch |
| Final model | Top-k averaging; Fake-F1 threshold search | Same weight averaging; a fresh validation Macro-F1 threshold for that averaged model |
| Test | Existing train path already calibrated on validation only | Still validation-only; explicit split/provenance guards reject test-time tuning |
| Standalone evaluation | Missing threshold silently defaulted to 0.5 | Refuses uncalibrated/legacy thresholds unless explicit validation recalibration is requested |
| Auxiliary exports | Missing threshold silently defaulted to 0.5 | Display/explanation/diagnostic predictions require validated checkpoint thresholds |
| AUC | Continuous Fake probability; silent `None` for single-class splits | Same continuous probability; explicit warning and NaN if undefined |
| NLL/Brier | Probability-based | Still original probabilities; NLL uses original true-class probability with log(0) protection only |
| ECE | Not reported in the central evaluator | Standard top-label ECE, 15 equal-width bins; independent of tuned threshold |
| Seed aggregation | No evaluation-protocol identity checks | Same-protocol validation/test provenance checks, mean and sample std (n−1), ECE included |

## Changed files

- `mmfnd/evaluation.py`: shared threshold search, metric calculation, class/provenance checks, checkpoint validation, result metadata and logging. No model dependencies.
- `mmfnd/engine.py`: original-label probability collection; one-pass validation tuning; frozen test threshold; shared predictions; checkpoint/threshold pairing and source validation.
- `train.py`: per-epoch tuned Macro-F1 selection/early stop; paired best/top-k/last saves; fresh calibration for averaged weights; reload the final saved checkpoint before the single final test; reject old-protocol/non-resumable checkpoints before rewriting run metadata.
- `evaluate.py`: fail-closed standalone evaluation; explicit `--recalibrate-on-val` migration that saves a new calibrated checkpoint and separate results, never overwriting the source checkpoint.
- `aggregate_seeds.py`: check protocol, validation/test threshold agreement and confusion-matrix Accuracy; keep all four seeds; calculate mean/sample std; propagate undefined AUC as NaN instead of dropping seeds.
- `explain.py`, `display.py`, `export_lgled_diagnostics.py`: remove implicit 0.5 fallbacks and record threshold provenance in saved prediction records.
- `configs/datasets/*.json`: all six datasets select on Macro-F1; remove obsolete `threshold_plateau_delta`. No label or optimizer/training-size changes.
- `scripts/scheduled_random4.py`: newly prepared schedules explicitly set `monitor=macro_f1`.
- The pending timer's three config snapshots: remove plateau delta and switch Gossip's AUC monitor to Macro-F1, retaining its original schedule/seeds/epochs/patience/batches.
- `tests/test_evaluation_protocol.py`: CPU-only numerical, persistence, training-flow, legacy migration and aggregation regression tests.
- `README.md` and this document: protocol and migration documentation.

## Checkpoint and threshold binding

Each validation result includes `threshold`, `decision_threshold`,
`threshold_source="validation"`, `threshold_objective="macro_f1"`,
`val_best_macro_f1`, `evaluation_protocol="validation_macro_f1_threshold_v1"`,
original label semantics and a structured `threshold_selection` record.

Epoch checkpoint provenance identifies the exact epoch. Saving/loading an epoch
with another epoch's threshold is rejected. Final averaged provenance records
the contributing epochs; it is **not** claimed to be one epoch's model. The
averaged model's own validation predictions select its threshold. Its `epoch`
field is the last training epoch, while `checkpoint_reference.kind` and `epochs`
identify the actual averaged weights. Test uses that saved pair, not a threshold
from the last epoch or an arithmetic mean of epoch/seed thresholds.

Every seed chooses independently. Aggregation reads frozen per-seed final-test
metrics; it never accesses test labels/probabilities to optimize a pooled threshold.
`best_val_macro_f1` in the final summary identifies the best individual epoch;
`val_best_macro_f1` identifies the final averaged model's calibration score;
`test_macro_f1` is its frozen-threshold test result.

## Global audit / remaining argmax

Audited `argmax`, `f1_score`, `macro_f1`, `best_f1`, `threshold`,
`best_threshold`, `precision_recall_fscore_support`, `accuracy_score`,
`roc_auc_score`, `log_loss`, `brier`, `ece`, plus every evaluator/checkpoint caller.
There is no remaining positive-F1 threshold search, plateau search, implicit 0.5
checkpoint fallback, or argmax-based reported Accuracy in production evaluators.

Two intentional production argmax uses remain:

- Standard top-label ECE chooses its probability-confidence class independently of the classification threshold.
- The optional standalone image occlusion helper can choose an argmax target if none is supplied. Both current CLI callers explicitly pass the threshold-derived predicted class; this fallback never contributes to classification metrics.

The Brier-like quantity in `mmfnd/model.py` is an existing training-loss term,
not the reported probability metric, and was intentionally left unchanged.
No test-label threshold tuning was found in the previous training path; the main
problems were objective/metric/selection inconsistency and silent fallback thresholds.

## CPU verification

From `/data/dyl/sotamodelv3`:

```bash
CUDA_VISIBLE_DEVICES='' /data/dyl/sotamodelv5/.venv/bin/python -m unittest discover -s tests -v
/data/dyl/sotamodelv5/.venv/bin/python scripts/test_scheduled_random4.py
```

The evaluation suite covers 15 tests, including a complete tiny CPU-model training
loop with early stopping and best/top-k/final saves. It does NOT load Qwen/SigLIP,
train a real dataset or allocate GPU tensors. Additional four scheduler tests
cover FIFO refill, busy-card avoidance, failure continuation and timer cancellation.
Threshold selection is deterministic for identical input probabilities; this does
not claim bitwise reproducibility of arbitrary GPU training kernels.

## Legacy results

Historical checkpoints/results have not been rewritten or deleted. They must not
be mixed with new-protocol seeds for a new-protocol mean/std report. Recalibration
can repair threshold/metric reporting for an existing fixed model, but cannot
retroactively change its old checkpoint-selection or early-stopping decisions.

For an existing checkpoint, pass `--recalibrate-on-val` to the regular evaluator:

```bash
/data/dyl/sotamodelv5/.venv/bin/python evaluate.py \
  --config configs/datasets/weibo21.json \
  --checkpoint /absolute/path/to/existing/checkpoint.pth \
  --split test --recalibrate-on-val
```

This command is an example only and was NOT run on any real model during this
repair. It evaluates validation first, saves `validation_calibrated.pth` under
a new `manual_evaluation` directory, then evaluates test with the saved threshold.
It performs model inference and will use the normal inference device when run.

## Pending server timer

The existing one-shot tmux timer is retained, not duplicated or rescheduled:

- Session: `v3_timer_20260914_0145`.
- Start: **2026-09-14 01:45:00 Asia/Shanghai**.
- Order: Weibo21 → GossipCop → Weibo; four random seeds each, 12 FIFO jobs.
- GPUs 0–3, one job per card, immediate refill across dataset boundaries.
- Maximum epoch 30, patience 8; batches 16/32/16, accumulation 1.
- Training commands start fresh processes from `/data/dyl/sotamodelv3/train.py`, so they load this revised protocol at launch.
- Dataset manifest hashes and original label semantics remain unchanged.
- Closing SSH is safe; tmux does not survive a server/container restart.

```bash
tail -f /data/dyl/sotamodelv3/workspaces/scheduled/v3_random4_20260914_014500/scheduler.log
```

## Final Evaluation Protocol

```text
Threshold source     : Validation set, independently per seed/model
Threshold objective  : Macro-F1
Positive class       : Fake (0); Real remains 1
Discrete metrics     : Shared threshold-based predictions
Accuracy             : Threshold-based
Macro-F1             : Threshold-based
Class-wise P/R/F1    : Threshold-based
AUC                  : Raw Fake probabilities
NLL                  : Raw true-class probabilities
Brier                : Raw Fake probabilities, binary target (label == 0)
ECE                  : Raw probabilities, standard top-label ECE
Test threshold tuning: Disabled
Checkpoint/threshold : Bound to the same epoch or freshly calibrated averaged model
```
