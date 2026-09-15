# CUTE-FND: Strict Masked R1
The current implementation is strict masked R1 in the masked workspace (`sotamodelv3_mask`), selected explicitly as `qwen_lora_lgled_masked_r1`. Historical configurations remain available for named legacy comparisons. The original tracked tree is preserved at `/data/dyl/sotamodelv3_mask_7649c557_pre_r1.tar` (base `7649c557b5d36bbd205ae353c96590b682a30402`).

Legacy: four views → shared Qwen strict deliberation → confidence/minority/global/direct scores → two-path fusion/IURD.

R1: same T/V/E/X → same shared Qwen last two layers with sample-specific strict mask → supervised reference-task A/M/C and vacuity → one supervised signed utility → one masked softmax → one weighted sum → LayerNorm/Dropout/Linear(D,2). The offline reference trains the entire existing encoder plus a small subset probe in three disjoint folds and produces all 16 probabilities from one encoding per input. It never becomes an inference dependency.

- [Method, output definitions, gradient routes, controls and limitations](docs/MASKED_R1_METHOD.md)
- [Data isolation, caching, five-seed commands, ablation/robustness/statistical interfaces and cost reporting](docs/MASKED_R1_PROTOCOL.md)
- [Actual validation results and the 26-contract checklist](docs/MASKED_R1_VALIDATION.md)
- [Baseline audit](BASELINE_AUDIT.md), [mathematical notes](docs/MASKED_R1_THEORY_NOTES.md)
- [Actual scheduled four-seed job](docs/MASKED_R1_SCHEDULE.md)

```bash
cd /data/dyl/sotamodelv3_mask
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests -q
bash scripts/run_masked_r1.sh --help
# GPU window only; all does not run test:
bash scripts/run_masked_r1.sh --dataset weibo21 --stage all --seed 20260916 --nproc 4
# Explicit final frozen test:
bash scripts/run_masked_r1.sh --dataset weibo21 --stage test --seed 20260916 --frozen
```

Checkpoint loading checks the architecture, exact trained parameter set, local frozen backbone identity and student source fingerprint, and loads a complete merged state with `strict=True` in R1. Changed cache/model/preprocessing/fold/augmentation identities fail closed. E/X are derived representations; internal deletion is not raw-modality causality. CPU engineering checks do not establish mechanism or performance gains. No new real benchmark improvement has been measured at delivery.

## Verification and fresh-start policy

Executed CPU checks: **66 pytest cases and 14 subtests passed**, plus synthetic single-process and four-process Gloo pipelines. Real-data GPU training, mechanism acceptance and performance acceptance are **not run** at publication; the four-seed suite is scheduled for 2026-09-16 01:00 Asia/Shanghai. Historical results below are not R1 results.

At the user's request, all 396 existing task training checkpoints in this masked workspace were deleted. Data, pretrained Qwen/SigLIP backbones and validation logs remain. All 12 scheduled jobs use new reference and main-model run directories without resume arguments; each starts from pretrained backbone weights and freshly initialized task parameters. See the [cleanup inventory](docs/r1/checkpoint_cleanup_20260915.json) and [formula-to-code map](docs/MASKED_R1_CODE_MAP.md).

## Historical v3 documentation

The following describes the previous architecture, commands and published results. For R1 use the configuration and entry point above.

# Historical CUTE-FND v3: LG-LED

CUTE-FND v3 is a multimodal fake-news classifier whose fourth stage is **LLM-Guided Latent Evidence Deliberation (LG-LED)**, with the subtitle *Uncertainty-Aware Cross-Evidence Adjudication*.

V2 used a similarity-driven Relational Evidence Graph: four 384-D nodes were connected by cosine-derived adjacency, updated by graph message passing, attentively pooled, and injected as a residual. V3 removes that graph path. It uses one shared Qwen2.5-7B-Instruct backbone for both textual encoding and latent cross-evidence adjudication. No second Qwen, four-model agent ensemble, natural-language debate, RAG, knowledge graph, GNN, or plain MHA fusion is used.

Agreement, ambiguity, and conflict are **task-driven learned latent relation beliefs**. They are not claims that Qwen establishes true logical contradiction because the supported datasets provide only Fake/Real labels, not pair-level relation labels.

## Evaluation protocol (updated 2026-09-13)

Original labels stay unchanged: **0=Fake, 1=Real**. Every epoch/seed independently
maximizes validation Macro-F1 over the configured Fake-probability threshold grid.
Accuracy, Macro-F1, class-wise P/R/F1 and confusion counts use that same threshold.
Early stopping/top-k selection use tuned validation Macro-F1. The averaged final
model gets its own validation threshold, stored inside its checkpoint and frozen
for test. AUC/NLL/Brier/ECE use original probabilities, not thresholded labels.

See [the evaluation audit and migration guide](docs/EVALUATION_PROTOCOL.md) for
changed files, tie-breaking, old-checkpoint handling, CPU tests and timer details.

## Architecture

The four real code-level evidence representations are:

1. `text`: final non-padding Qwen text state projected to D.
2. `vision`: mean SigLIP image representation projected to D and aggregated per post.
3. `intrinsic_feature`: intrinsic/event evidence derived from text self-relations and local/global text-image alignment.
4. `interaction`: cross-modal representation of `|text-vision|` and `text*vision`.

`D` is configurable and currently 384. Qwen hidden size and layer count are read dynamically from `qwen.config`.

```text
Four 384-D Evidence
        ↓
ONE Shared 384→H Projector
        ↓
Four Evidence Role Embeddings
        ↓
4 Evidence Tokens + 6 Pair Judges + 1 Global Judge
        ↓
11-token causal latent sequence
        ↓
The same Qwen2.5-7B Last-2 Decoder Layers
        ↓
Dirichlet Agreement / Ambiguity / Conflict + Uncertainty
        ↓
Confidence → Uncertainty-Aware Deviation → Critical Minority
        ↓
Global Preference + Shared Evidence Adjudication Head
        ↓
Selective Direct / Deliberative Fusion
        ↓
384-D Final Evidence → Existing Calibrated Binary Classifier
```

The pair order is fixed as text–vision, text–intrinsic, text–interaction, vision–intrinsic, vision–interaction, and intrinsic–interaction. Evidence precedes all judge tokens; the global judge is last.

This mask version applies strict visibility in **every** reused Qwen layer: each evidence token attends only to itself; each of the six pair judges attends only to its two corresponding evidence tokens and itself; G attends to all 11 tokens. Pair judges cannot attend to other judges or G. Isolating evidence tokens also prevents indirect leakage across layers. The same mask applies to the non-LLM Transformer ablation (and the 10-token sequence without G). Qwen eager and SDPA attention are supported; unsupported attention backends fail explicitly. This restriction starts at the four LG-LED input evidence vectors; upstream evidence construction and downstream fusion are unchanged.

Simplified formulation:

```text
z_k = LN(P_shared(e_k) + r_k)
S = [z_T,z_V,z_I,z_C,J_TV,J_TI,J_TC,J_VI,J_VC,J_IC,J_G]
H = Qwen_last-N(S)

e_ij = softplus(W_e h_ij)
alpha_ij = e_ij + 1
p_ij,k = alpha_ij,k / sum_c alpha_ij,c
u_ij = 3 / sum_c alpha_ij,c

o_i = (1/3) sum_{j != i} p_ij,conflict (1-u_ij)
g_delib = sum_i a_i e_i
D_sample = mean_pairs[p_conflict (1-u)]
lambda = sigmoid(scale (D_sample-tau))
g = (1-lambda) g_direct + lambda g_delib
```

High deviation does not cause evidence deletion. The differentiable critical-minority head can preserve a confident, low-uncertainty minority evidence item when it contains a useful misinformation clue.

## Setup

Python 3.10, PyTorch 2.4.1, Transformers 4.46.3, and PEFT 0.13 are the tested API contract.

```bash
git clone https://github.com/MoBaiKing/sotamodelvqwen.git
cd sotamodelvqwen
bash scripts/setup_environment.sh
```

Download/reuse the exact local backbones:

```bash
.venv/bin/python scripts/download_pretrained_models.py
```

Set `HF_SHARED_CACHE=/path/to/cache` (or pass `--shared-cache`) to reuse
complete local model directories instead of downloading another copy.

Equivalent standard Hugging Face command for the Qwen model is:

```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir pretrained_models/Qwen2.5-7B-Instruct
```

The model code uses `local_files_only=True`; an incomplete path fails rather than silently downloading during DDP startup. The expected locations are:

```text
pretrained_models/Qwen2.5-7B-Instruct
pretrained_models/siglip-base-patch16-224
```

One-command bootstrap:

```bash
DATASET_SOURCE_ROOT=/path/to/prepared/datasets bash scripts/bootstrap.sh
```

## V2 datasets prepared for V3

Expose validated V2 GossipCop, Weibo21, Twitter, and Weibo datasets inside this
project with space-saving `datasets/<name>/ready` links:

```bash
.venv/bin/python scripts/prepare_v2_datasets.py \
  --source-root /path/to/prepared/datasets
```

Use `--copy` only when a physically independent copy is required. The script validates dataset identity, Fake=0/Real=1 semantics, manifests, and all three JSONL splits before creating anything.

Raw-data adapters remain available through `prepare_dataset.py` for a new copy of a dataset.

## Smoke tests

Fast Transformers/PEFT API test without downloading 7B:

```bash
.venv/bin/python smoke_lgled_unit.py
```

Real Qwen2.5-7B, SigLIP, data, forward/backward, BF16, memory, runtime, gradient, layer-sharing, LoRA-sharing, state-dict, and checkpoint audit:

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python smoke_intrinsic.py \
  --config configs/datasets/weibo21.json \
  --manifest-dir datasets/weibo21/ready
```

Four-rank DDP smoke training:

```bash
bash scripts/run_dataset.sh weibo21 --smoke-steps 2 --run-name weibo21_lgled_ddp_smoke
```

## Training commands

All four commands retain the original `train.py`/`torchrun` interface and use four GPUs by default:

```bash
bash scripts/run_dataset.sh gossipcop
bash scripts/run_dataset.sh weibo21
bash scripts/run_dataset.sh twitter
bash scripts/run_dataset.sh weibo
```

The expanded Weibo21 command is:

```bash
.venv/bin/python -m torch.distributed.run \
  --standalone --nproc_per_node=4 train.py \
  --dataset weibo21 \
  --config configs/datasets/weibo21.json \
  --manifest-dir datasets/weibo21/ready
```

Five seeds:

```bash
bash run_5seeds.sh --dataset gossipcop
bash run_5seeds.sh --dataset weibo21
bash run_5seeds.sh --dataset twitter
bash run_5seeds.sh --dataset weibo
```

The first-round memory-safe defaults for four 46-GB L20 GPUs are BF16, per-GPU batch size 1, gradient accumulation 4 (Weibo/Weibo21) or 8 (GossipCop/Twitter), text length 192, LoRA r=8/alpha=16/dropout=0.05 on `q_proj` and `v_proj`, `use_cache=false`, gradient checkpointing, and Last-2 latent judge layers. Reduce text length or batch size before considering FSDP/ZeRO; 4-bit QLoRA is not enabled by default.

## Diagnostics and case studies

Validation/test metrics contain aggregate LG-LED beliefs, strength, uncertainty, confidence, deviation, minority scores, global/direct/deliberative/final weights, sample disagreement, and routing gate. Set `model.lgled.export_diagnostics=true` to include sample-level diagnostics in normal evaluation JSONL, or export explicitly:

```bash
.venv/bin/python export_lgled_diagnostics.py \
  --config configs/datasets/weibo21.json \
  --manifest-dir datasets/weibo21/ready \
  --checkpoint workspaces/weibo21/runs/training/RUN/checkpoints/final_averaged.pth \
  --split test
```

## Ablations

Recommended order: Last-1/Last-2/Last-4; shared versus independent projector; Qwen latent versus `mlp_pair`; without role embedding; without uncertainty; without critical minority; without global judge; without selective routing; then mean pooling/plain MHA external baselines. The main configuration always uses the single shared projector and `qwen_latent` judge. Independent projectors and the MLP judge are instantiated only when explicitly selected.

## Published experiment outputs

See [results/README.md](results/README.md) for the strict-mask 4-seed suite.
The export includes per-sample validation/test predictions, metrics, selected
thresholds, run metadata, epoch histories and multi-seed aggregates. The two
Weibo runs interrupted by the scheduled cutoff retain their histories and
continuation metadata. Prediction JSONL files use lossless gzip compression;
model weights and optimizer checkpoints are excluded.
