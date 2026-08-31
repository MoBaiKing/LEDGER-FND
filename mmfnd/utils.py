from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
import os
import random
from pathlib import Path



def load_config(path: str | Path) -> dict:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as file:
        if path.suffix.lower() == ".json":
            return json.load(file)
        try:
            import yaml
        except ImportError as error:
            raise ImportError("读取 YAML 配置需要 PyYAML；也可改用 config.json") from error
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    world_size: int
    local_rank: int
    device: object

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedContext:
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        backend = "nccl"
    else:
        device = torch.device("cpu") if world_size > 1 else get_device()
        backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        # Non-main ranks intentionally wait while rank 0 evaluates the complete
        # validation/test split, so use an explicit generous collective timeout.
        dist.init_process_group(
            backend=backend, init_method="env://", timeout=timedelta(hours=2)
        )
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    return DistributedContext(rank, world_size, local_rank, device)


def cleanup_distributed() -> None:
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def resolve_path(project_root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def dump_json(data: object, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def get_device(local_rank: int | None = None):
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda", 0 if local_rank is None else local_rank)
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
