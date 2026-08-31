# CUTE-FND v2

CUTE-FND v2 is a multimodal fake-news classification model. This repository contains the model implementation, training and evaluation entry points, dataset adapters, and configuration files. Datasets and model parameter files are not included.

## Structure

```text
.
├── configs/datasets/   # dataset and training configurations
├── dataset_tools/      # dataset preprocessing adapters
├── datasets/           # local dataset directories
├── mmfnd/              # model, data loader, metrics, and runtime utilities
├── prepare_dataset.py
├── train.py
└── evaluate.py
```

`explain.py`, `display.py`, `export_domain_graph.py`, and `smoke_intrinsic.py` provide optional inspection and smoke-test utilities.

## Environment

Python 3.10 is recommended.

```bash
conda env create -f environment.yml
conda activate cute-fnd-py310
python -m pip install -r requirements.txt
```

## Data preparation

The repository supports Fakeddit, FineFake, GossipCop, Twitter, Weibo, and Weibo21. Place local files under `datasets/<dataset>/raw/`, then adjust the corresponding file in `configs/datasets/` if its layout differs.

Each preprocessing adapter writes `train.jsonl`, `val.jsonl`, `test.jsonl`, and `dataset_manifest.json` to the configured `processed_dir`.

```bash
python prepare_dataset.py \
  --dataset weibo21 \
  --config configs/datasets/weibo21.json \
  --verify-decode
```

Set `model.text_backbone` and `model.vision_backbone` in the dataset configuration to local model directories before training.

## Training

```bash
python train.py \
  --dataset weibo21 \
  --config configs/datasets/weibo21.json \
  --manifest-dir workspaces/weibo21/processed
```

For distributed training, launch the same command with `torchrun` and the required process count.

## Evaluation

```bash
python evaluate.py \
  --config configs/datasets/weibo21.json \
  --checkpoint workspaces/weibo21/runs/training/RUN_NAME/checkpoints/final_averaged.pth \
  --split test
```

Training and evaluation outputs are written under `workspaces/` and are ignored by Git.
