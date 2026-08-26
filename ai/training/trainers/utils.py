"""Trainer utilities: config load, seed, device, optimizers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "ai" / "training" / "runs"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be a mapping: {path}")
    return data


def load_config(
    task: str = "text",
    *,
    config_path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Load common.yaml + task yaml (text.yaml / vision.yaml), then apply overrides.

    Hyperparameters live only in YAML — trainers must not hardcode them.
    """
    common = load_yaml(CONFIGS_DIR / "common.yaml")
    task_file = {
        "text": "text.yaml",
        "vision": "vision.yaml",
        "image": "vision.yaml",
    }.get(task, f"{task}.yaml")
    task_cfg = load_yaml(CONFIGS_DIR / task_file)
    cfg = deep_merge(common, task_cfg)
    if config_path is not None:
        cfg = deep_merge(cfg, load_yaml(config_path))
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(device: str | None = None) -> str:
    if device and device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def ensure_run_dirs(cfg: dict[str, Any], *, run_name: str | None = None) -> dict[str, Path]:
    paths_cfg = cfg.get("paths") or {}
    root = Path(paths_cfg.get("output_root") or DEFAULT_OUTPUT_ROOT)
    if not root.is_absolute():
        root = REPO_ROOT / root
    name = run_name or cfg.get("run_name") or cfg.get("task", "run")
    run_dir = root / name
    dirs = {
        "run": run_dir,
        "logs": run_dir / (paths_cfg.get("logs_dirname") or "logs"),
        "models": run_dir / (paths_cfg.get("models_dirname") or "models"),
        "reports": run_dir / (paths_cfg.get("reports_dirname") or "reports"),
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def build_optimizer(model: Any, cfg: dict[str, Any]):
    import torch

    train_cfg = cfg.get("train") or {}
    name = str(train_cfg.get("optimizer", "adamw")).lower()
    lr = float(train_cfg.get("learning_rate", 2e-5))
    wd = float(train_cfg.get("weight_decay", 0.01))
    params = [p for p in model.parameters() if p.requires_grad]
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            weight_decay=wd,
            momentum=float(train_cfg.get("momentum", 0.9)),
        )
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def build_scheduler(optimizer: Any, cfg: dict[str, Any], *, steps_per_epoch: int):
    import torch

    train_cfg = cfg.get("train") or {}
    name = str(train_cfg.get("scheduler", "none")).lower()
    epochs = int(train_cfg.get("epochs", 1))
    if name in ("none", "", "null"):
        return None
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    if name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(train_cfg.get("scheduler_step_size", 1)),
            gamma=float(train_cfg.get("scheduler_gamma", 0.1)),
        )
    if name == "linear":
        total_steps = max(1, epochs * max(1, steps_per_epoch))
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=float(train_cfg.get("scheduler_end_factor", 0.0)),
            total_iters=total_steps,
        )
    raise ValueError(f"Unknown scheduler: {name}")


def get_lr(optimizer: Any) -> float:
    return float(optimizer.param_groups[0]["lr"])
