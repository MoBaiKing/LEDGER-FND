# Experiment results

Model variant: **sotamodelv3_masked**.

This directory contains per-sample validation/test predictions, metrics,
threshold metadata, run configuration, epoch history, final summaries and
multi-seed aggregates. Checkpoints, optimizer state, pretrained weights and
training logs are intentionally excluded.

Prediction files are the original JSONL bytes stored with deterministic gzip
compression. Decompress with gzip -dc predictions.jsonl.gz; SHA-256 values
for both compressed and original bytes are recorded in manifest.json.

Labels use 0=fake and 1=real.

| Dataset | Experiment | Seeds | Status | Accuracy | Macro-F1 | AUC |
|---|---|---:|---|---:|---:|---:|
| gossipcop | sotav3mask_random4_ep30_20260915_020000_gossipcop | 4 | complete | 87.8561% | 81.3582% | 90.4498% |
| weibo | sotav3mask_random4_ep30_20260915_020000_weibo | 4 | partial (2/4) | — | — | — |
| weibo21 | sotav3mask_random4_ep30_20260915_020000_weibo21 | 4 | complete | 96.3008% | 96.3006% | 98.9137% |

A partial experiment has no multi-seed aggregate. Its completed runs retain
their final predictions and metrics; interrupted runs retain configuration
and epoch history only.
