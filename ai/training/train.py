"""
ConsentShield training entrypoint (Phase 1).

Default mode: validate datasets, dataloaders, configs, and run directories.
Does NOT fine-tune MiniLM or CLIP.

Usage:
  python -m ai.training.train --task text --dry-run
  python -m ai.training.train --task vision --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from ai.training.datasets.collate import (
    collate_image_batch,
    collate_text_batch,
    create_dataloader,
)
from ai.training.datasets.image_dataset import ImageDataset
from ai.training.datasets.text_dataset import TextDataset
from ai.training.datasets.unified_dataset import UnifiedDataset
from ai.training.trainers.utils import (
    REPO_ROOT,
    ensure_run_dirs,
    load_config,
    resolve_device,
    set_seed,
)


def _data_filters(cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data") or {}
    return {
        "sources": data.get("sources"),
        "modalities": data.get("modalities"),
        "label_binary": data.get("label_binary"),
        "label_fine": data.get("label_fine"),
        "label_consentshield": data.get("label_consentshield"),
        "max_samples": data.get("max_samples"),
    }


def build_datasets(task: str, cfg: dict[str, Any]) -> dict[str, UnifiedDataset]:
    data = cfg.get("data") or {}
    paths = cfg.get("paths") or {}
    data_dir = paths.get("data_dir") or "datasets/unified"
    data_path = Path(data_dir)
    if not data_path.is_absolute():
        data_path = REPO_ROOT / data_path

    label_field = str(data.get("label_field", "label_binary"))
    filters = _data_filters(cfg)
    common = {
        "data_dir": data_path,
        "label_field": label_field,
        "repo_root": REPO_ROOT,
        **{k: v for k, v in filters.items() if v is not None},
    }

    if task in ("text", "minilm"):
        min_chars = int(data.get("min_text_chars", 1))
        train_ds = TextDataset("train", min_text_chars=min_chars, **common)
        vocab = train_ds.label_vocab
        val_ds = TextDataset("val", min_text_chars=min_chars, label_vocab=vocab, **common)
        test_ds = TextDataset("test", min_text_chars=min_chars, label_vocab=vocab, **common)
    elif task in ("vision", "image", "clip"):
        require = bool(data.get("require_image_exists", True))
        train_ds = ImageDataset("train", require_image_exists=require, **common)
        vocab = train_ds.label_vocab
        val_ds = ImageDataset("val", require_image_exists=require, label_vocab=vocab, **common)
        test_ds = ImageDataset("test", require_image_exists=require, label_vocab=vocab, **common)
    else:
        train_ds = UnifiedDataset("train", **common)
        vocab = train_ds.label_vocab
        val_ds = UnifiedDataset("val", label_vocab=vocab, **common)
        test_ds = UnifiedDataset("test", label_vocab=vocab, **common)

    return {"train": train_ds, "val": val_ds, "test": test_ds}


def build_loaders(
    task: str,
    datasets: dict[str, UnifiedDataset],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    data = cfg.get("data") or {}
    train_cfg = cfg.get("train") or {}
    batch_size = int(train_cfg.get("batch_size", 16))
    num_workers = int(data.get("num_workers", 0))
    pin_memory = bool(data.get("pin_memory", False))
    persistent_workers = bool(data.get("persistent_workers", False))
    drop_last = bool(data.get("drop_last", False))
    shuffle_train = bool(train_cfg.get("shuffle_train", True))

    if task in ("text", "minilm"):
        collate = collate_text_batch
    elif task in ("vision", "image", "clip"):
        collate = collate_image_batch
    else:
        collate = None

    loaders = {}
    for split, ds in datasets.items():
        loaders[split] = create_dataloader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle_train if split == "train" else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            drop_last=drop_last if split == "train" else False,
            collate_fn=collate,
        )
    return loaders


def dry_run(task: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Validate framework wiring without training any model."""
    set_seed(int((cfg.get("reproducibility") or {}).get("seed", 42)))
    device = resolve_device((cfg.get("hardware") or {}).get("device"))
    run_dirs = ensure_run_dirs(cfg)

    datasets = build_datasets(task, cfg)

    batch_preview: dict[str, Any] = {}
    dataloader_status = "ok"
    try:
        loaders = build_loaders(task, datasets, cfg)
        train_loader = loaders["train"]
        if len(datasets["train"]) > 0:
            batch = next(iter(train_loader))
            if isinstance(batch, dict):
                batch_preview = {
                    "keys": sorted(batch.keys()),
                    "batch_size": len(batch.get("ids") or batch.get("label_ids") or []),
                }
            else:
                batch_preview = {"type": type(batch).__name__, "len": len(batch)}
    except ImportError as exc:
        # Allow dataset/config validation without the optional `[ai]` torch extra.
        dataloader_status = f"skipped ({exc})"
        if len(datasets["train"]) > 0:
            sample = datasets["train"][0]
            batch_preview = {
                "sample_keys": sorted(sample.keys()),
                "sample_id": sample.get("id"),
                "note": "DataLoader requires PyTorch; sample[0] verified without torch",
            }

    report = {
        "phase": 1,
        "mode": "dry-run",
        "task": task,
        "device": device,
        "training_enabled": False,
        "dataloader_status": dataloader_status,
        "run_dirs": {k: str(v) for k, v in run_dirs.items()},
        "config_train": cfg.get("train"),
        "datasets": {name: ds.summary() for name, ds in datasets.items()},
        "batch_preview": batch_preview,
        "note": "No model weights were loaded or updated. Phase 2/3 required for training.",
    }

    out_path = run_dirs["reports"] / "phase1_dry_run.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConsentShield training framework (Phase 1)")
    p.add_argument("--task", choices=["text", "vision", "unified"], default="text")
    p.add_argument("--config", type=str, default=None, help="Optional YAML override file")
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Validate data/config pipeline only (default in Phase 1)",
    )
    p.add_argument(
        "--allow-train",
        action="store_true",
        help="Attempt real training (blocked until Phase 2/3 model builders exist)",
    )
    p.add_argument("--run-name", type=str, default=None)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.run_name:
        overrides["run_name"] = args.run_name

    cfg = load_config(args.task if args.task != "unified" else "text", config_path=args.config, overrides=overrides)
    if args.task == "unified":
        cfg["task"] = "unified"

    if args.allow_train:
        print(
            "ERROR: --allow-train is not available in Phase 1.\n"
            "TextTrainer/VisionTrainer.build_model are intentionally unimplemented.\n"
            "Approve Phase 2 (MiniLM) or Phase 3 (CLIP) before enabling training.",
            file=sys.stderr,
        )
        return 2

    report = dry_run(args.task, cfg)
    print(json.dumps(report, indent=2))
    print(f"\nPhase 1 dry-run OK -> {report['report_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
