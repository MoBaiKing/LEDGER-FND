# CUTE-FND v3: LG-LED

CUTE-FND v3 is a multimodal fake-news classifier whose fourth stage is **LLM-Guided Latent Evidence Deliberation (LG-LED)**, with the subtitle *Uncertainty-Aware Cross-Evidence Adjudication*.

V2 used a similarity-driven Relational Evidence Graph: four 384-D nodes were connected by cosine-derived adjacency, updated by graph message passing, attentively pooled, and injected as a residual. V3 removes that graph path. It uses one shared Qwen2.5-7B-Instruct backbone for both textual encoding and latent cross-evidence adjudication. No second Qwen, four-model agent ensemble, natural-language debate, RAG, knowledge graph, GNN, or plain MHA fusion is used.

Agreement, ambiguity, and conflict are **task-driven learned latent relation beliefs**. They are not claims that Qwen establishes true logical contradiction because the supported datasets provide only Fake/Real labels, not pair-level relation labels.

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

The pair order is fixed as text–vision, text–intrinsic, text–interaction, vision–intrinsic, vision–interaction, and intrinsic–interaction. Evidence precedes all judge tokens because Qwen is causal; the global judge is last.

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
