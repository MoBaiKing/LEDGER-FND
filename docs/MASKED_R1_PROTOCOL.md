# Reproducible protocol and executable commands

Run from `/data/dyl/sotamodelv3_mask`. Existing `.venv`, pretrained files and official manifests are reused. No environment upgrade or full model download was performed. Current immediate execution uses CPU only (`CUDA_VISIBLE_DEVICES=''`). The scheduled GPU suite is separate.

## Reference isolation and replay

Each reference seed equals its student seed. Three target folds use grouped stratification: union real event_id/source_id/near_duplicate_group/original_id when present and whitespace-normalized casefold exact text duplicates. No synthetic event labels are invented. Variants share the original fold. Inside train minus target fold, a grouped stratified inner split supplies reference_fit and reference_cal. Every encoder, E/X generator, projection, adapter and probe starts from the same pretrained/fresh initialization and trains on fit labels only. Calibration labels select the reference checkpoint/temperature, never encoder gradients. Temperature candidates [.5,1,2,4] are internal engineering choices, not verified optimum values.

The pool is clean plus one deterministic augmentation for datasets with augmentation enabled; disabled datasets have only clean. Augmentation seeds depend on reference/student seed, sample_id and augmentation_id; Python RNG state is restored around sample generation. The original crop/brightness/contrast/JPEG/image ordering routines are retained. This changes the old infinitely randomized augmentation protocol. All matched controls must use this same pool and label_smoothing=0.

OOF probabilities are saved before the separate target-building stage reads train labels. Cache fingerprints cover model/backbone/tokenizer/processor bytes, preprocessing, original image contents, train manifest, folds, seed, encoder/reference source, runtime library versions and reference optimizer settings. Every row includes original ID, augmentation ID, availability, fold, fit/cal hashes, reference checkpoint hash and q in [Fake,Real] order. Exact cache lookups fail on mismatches. Student inference needs its checkpoint/config and original local frozen backbones, never teacher models/caches.

## Training/evaluation

Fake=0, Real=1, positive_label=0. Thresholded prediction is Fake iff p_fake>=threshold. Original grid .20:.01:.80 is retained for all three datasets. All discrete metrics share that prediction; NLL, binary Brier, top-label ECE-15 and AUC use continuous probabilities. ECE uses argmax correctness, independent of the reporting threshold.

Early stopping, best checkpoints and top-k averaging all use validation Macro-F1 after threshold selection. Ties prefer closest to .5, then lower threshold. Averaged weights are evaluated again and get their own threshold. Missing/mismatched thresholds fail. `all` never performs final test. Explicit `test --frozen` uses the exact final weights/threshold. Main results remain raw, T=1. Optional scalar-temperature fitting requires predefined disjoint val_cal/val_select; raw evaluation rejects calibrated metadata. The helper is not enabled in the main run.

Validation uses one complete, unsampled loader on rank 0; other ranks wait at symmetric barriers. Prediction rows are keyed/deduplicated by sample_id and inconsistent duplicates fail. No padded distributed validation sampler is used. CPU/Gloo four-process execution has been run; GPU/NCCL remains pending. Reference folds unload models AND temporary full state dictionaries before the next fold.

## Unified stages

```bash
cd /data/dyl/sotamodelv3_mask
CUDA_VISIBLE_DEVICES='' bash scripts/run_masked_r1.sh --dataset weibo21 --stage preflight --seed 20260916
# GPU commands below are for the scheduled/explicit GPU window only.
bash scripts/run_masked_r1.sh --dataset weibo21 --stage reference --seed 20260916 --nproc 4
bash scripts/run_masked_r1.sh --dataset weibo21 --stage targets --seed 20260916
bash scripts/run_masked_r1.sh --dataset weibo21 --stage train --seed 20260916 --nproc 4
bash scripts/run_masked_r1.sh --dataset weibo21 --stage diagnostics --seed 20260916 --split val
bash scripts/run_masked_r1.sh --dataset weibo21 --stage test --seed 20260916 --frozen
```

Reference directories must be new (or contain only preflight.json); existing completed references are never silently overwritten. To reuse a matching completed reference, invoke `train` directly with `--reference-dir`. To continue a student last checkpoint, use `--resume /absolute/path/checkpoints/last.pth` with the same seed/config/reference directory and process count. Per-rank RNG, optimizer, scheduler and training selection history are restored. Model-only best/final checkpoints cannot resume training. Interrupted reference fitting currently requires a new reference output directory; its incomplete files remain preserved.

## Three datasets / five explicit paired seeds

This is the formal five-seed interface; it is NOT the requested four-seed scheduled suite. Run the same list on every method/dataset. No claims of measured performance accompany these commands.

```bash
for dataset in gossipcop weibo21 weibo; do
  bash scripts/run_masked_r1.sh --dataset "$dataset" --stage all --nproc 4 \
    --seeds 202609161 202609162 202609163 202609164 202609165
done
# Only after the weights and validation decisions from all are frozen:
for dataset in gossipcop weibo21 weibo; do
  bash scripts/run_masked_r1.sh --dataset "$dataset" --stage test --frozen \
    --seeds 202609161 202609162 202609163 202609164 202609165
done
```

## Ablations and strong baselines

```bash
bash scripts/run_masked_r1.sh --dataset weibo21 --stage train --seed 20260916 --nproc 4 \
  --config configs/revisions/v3_masked_r1/ablations/weibo21_mlp_judge.json \
  --reference-dir workspaces/r1_reference/weibo21/seed20260916 --run-name mlp_seed20260916
bash scripts/run_masked_r1.sh --dataset weibo21 --stage train --seed 20260916 --nproc 4 \
  --config configs/revisions/v3_masked_r1/ablations/weibo21_encoder_mean.json --run-name mean_seed20260916
# Original masked, unchanged old protocol replay:
.venv/bin/python train.py --dataset weibo21 --config configs/datasets/weibo21.json \
  --manifest-dir datasets/weibo21/ready --seed 20260916 --run-name legacy_replay_seed20260916
# Legacy architecture, matched replay pool and zero smoothing (separate result table):
bash scripts/run_masked_r1.sh --dataset weibo21 --stage train --seed 20260916 --nproc 4 \
  --config configs/revisions/v3_masked_r1/ablations/weibo21_legacy_matched.json --run-name legacy_matched_seed20260916
```

Use other ablation JSON names in the same manner. Ordinary encoder mean/scorer need no reference; uniform fusion has no utility head; task_scorer sets L_utility=0 and enables CE routing so it is actually trained. Reference generation cost is separately recorded, never hidden inside a cheaper baseline budget. Optimizer lora/vision/head parameter grouping is preserved. Queue configs multiply accumulation by four when running one GPU/seed, preserving original four-rank effective batches (GossipCop 32, Weibo21/Weibo 16); this must also be used for matched single-GPU controls.

Fixed-input controls have separate executable exporters/trainers:

```bash
.venv/bin/python scripts/export_masked_r1_fixed_views.py --help
.venv/bin/python scripts/run_frozen_judge_control.py --help
# Export with explicit config, checkpoint, reference-dir, output and
# --encoder-selection predeclared_train_epoch, then use the SAME cache twice:
.venv/bin/python scripts/run_frozen_judge_control.py --cache /absolute/fixed_views.pt \
  --config configs/revisions/v3_masked_r1/weibo21.json --judge-init pretrained --seed 20260916 \
  --epochs 3 --output /absolute/pretrained_control
.venv/bin/python scripts/run_frozen_judge_control.py --cache /absolute/fixed_views.pt \
  --config configs/revisions/v3_masked_r1/weibo21.json --judge-init random --seed 20260916 \
  --epochs 3 --eval-shuffle --output /absolute/random_control
```

## Robustness/statistics

`evaluate_masked_r1_mechanisms.py` supports the existing frozen-test Gaussian/typo interfaces, explicit strengths/seeds and recomputation of E/X. Use separate output directories per strength. Entity/label preservation of token typos cannot be guaranteed and is disclosed. `compare_masked_r1_runs.py` accepts paired per-seed JSONL predictions; it resamples the same news/group indices across all seeds, not seed-times-news as independent observations. It reports seed differences and a paired bootstrap interval, with explicit resampling seed/count/unit and Bonferroni comparison count.

## Costs and interpretation

Per-reference-fold time, steps, checkpoint temperature, parameters and peak GPU allocation are recorded in reference_manifest.json. Student final_summary.json records parameters, elapsed training/validation time, throughput, actual effective batch and rank-0 peak GPU memory. No measured FLOPs are claimed. Rank-specific GPU maxima for smoke are available in train.py; full-training peak reporting currently names rank 0 explicitly. Reference/statistical/robustness results on the real datasets have NOT RUN at delivery. Neither CPU correctness nor synthetic diagnostics demonstrate accuracy improvements, useful dissent detection, semantic relation calibration, or a causal claim.
