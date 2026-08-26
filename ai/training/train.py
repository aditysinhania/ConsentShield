"""
ConsentShield training entrypoint.

Default: dry-run (datasets / dataloaders / config).
Phase 2 MiniLM: ``--allow-train`` with ``--task text`` runs real fine-tuning.

Usage:
  python -m ai.training.train --task text --dry-run
  python -m ai.training.train --config ai/training/configs/text.yaml --allow-train
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

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
    deep_merge,
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
        "label_field": label_field,
        "repo_root": REPO_ROOT,
        **{k: v for k, v in filters.items() if v is not None},
    }

    if task in ("text", "minilm"):
        min_chars = int(data.get("min_text_chars", 1))
        text_train = data_path / "text_train.jsonl"
        if text_train.is_file():
            train_ds = TextDataset(
                "train",
                path=text_train,
                min_text_chars=min_chars,
                **common,
            )
            vocab = train_ds.label_vocab
            val_ds = TextDataset(
                "val",
                path=data_path / "text_val.jsonl",
                min_text_chars=min_chars,
                label_vocab=vocab,
                **common,
            )
            test_ds = TextDataset(
                "test",
                path=data_path / "text_test.jsonl",
                min_text_chars=min_chars,
                label_vocab=vocab,
                **common,
            )
        else:
            train_ds = TextDataset(
                "train",
                data_dir=data_path,
                min_text_chars=min_chars,
                **common,
            )
            vocab = train_ds.label_vocab
            val_ds = TextDataset(
                "val",
                data_dir=data_path,
                min_text_chars=min_chars,
                label_vocab=vocab,
                **common,
            )
            test_ds = TextDataset(
                "test",
                data_dir=data_path,
                min_text_chars=min_chars,
                label_vocab=vocab,
                **common,
            )
    elif task in ("vision", "image", "clip"):
        require = bool(data.get("require_image_exists", True))
        train_ds = ImageDataset("train", data_dir=data_path, require_image_exists=require, **common)
        vocab = train_ds.label_vocab
        val_ds = ImageDataset(
            "val",
            data_dir=data_path,
            require_image_exists=require,
            label_vocab=vocab,
            **common,
        )
        test_ds = ImageDataset(
            "test",
            data_dir=data_path,
            require_image_exists=require,
            label_vocab=vocab,
            **common,
        )
    else:
        train_ds = UnifiedDataset("train", data_dir=data_path, **common)
        vocab = train_ds.label_vocab
        val_ds = UnifiedDataset("val", data_dir=data_path, label_vocab=vocab, **common)
        test_ds = UnifiedDataset("test", data_dir=data_path, label_vocab=vocab, **common)

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
        "note": "Dry-run only. Use --allow-train for MiniLM Phase 2 training.",
    }

    out_path = run_dirs["reports"] / "phase1_dry_run.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def _finalize_text_artifacts(trainer: Any, cfg: dict[str, Any], history: list[dict[str, Any]]) -> dict[str, str]:
    models_dir = trainer.run_dirs["models"]
    best_src = models_dir / str((cfg.get("checkpoint") or {}).get("best_name", "best.pt"))
    latest_src = models_dir / str((cfg.get("checkpoint") or {}).get("latest_name", "latest.pt"))
    mapping = {
        "best_model.pt": best_src if best_src.is_file() else latest_src,
        "last_model.pt": latest_src if latest_src.is_file() else best_src,
    }
    paths: dict[str, str] = {}
    for name, src in mapping.items():
        if src.is_file():
            dst = models_dir / name
            if dst.resolve() != src.resolve():
                if dst.exists():
                    dst.unlink()
                shutil.copy2(src, dst)
            paths[name] = str(dst)
    config_path = models_dir / "training_config.yaml"
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    paths["training_config.yaml"] = str(config_path)
    if history:
        trainer.save_checkpoint(
            models_dir / "training_state.pt",
            epoch=int(history[-1].get("epoch") or len(history)),
            metrics=(history[-1].get("metrics") or {}),
        )
        paths["training_state.pt"] = str(models_dir / "training_state.pt")
    return paths


def run_text_training(cfg: dict[str, Any], *, max_batches: int | None = None) -> dict[str, Any]:
    """Phase 2 MiniLM fine-tuning via TextTrainer (no PhaseGate)."""
    from ai.training.trainers.text_trainer import TextTrainer

    set_seed(int((cfg.get("reproducibility") or {}).get("seed", 42)))

    if max_batches is not None and max_batches > 0:
        batch_size = int((cfg.get("train") or {}).get("batch_size", 32))
        cfg = deep_merge(
            cfg,
            {
                "train": {"epochs": 1},
                "data": {"max_samples": max(batch_size * max_batches, batch_size)},
            },
        )

    datasets = build_datasets("text", cfg)
    loaders = build_loaders("text", datasets, cfg)

    trainer = TextTrainer(cfg, allow_train=True)
    trainer.setup(
        num_labels=len(datasets["train"].label_vocab),
        label_vocab=datasets["train"].label_vocab,
    )

    fit_result = trainer.fit(loaders["train"], loaders["val"])
    history = fit_result["history"]
    artifacts = _finalize_text_artifacts(trainer, cfg, history)

    best_path = Path(artifacts.get("best_model.pt") or trainer.run_dirs["models"] / "best.pt")
    if best_path.is_file():
        trainer.load_checkpoint(best_path)

    return {
        "phase": 2,
        "mode": "train",
        "device": trainer.device,
        "amp": trainer.use_amp,
        "freeze_layers": trainer.freeze_layers,
        "train_samples": len(datasets["train"]),
        "val_samples": len(datasets["val"]),
        "history": history,
        "run_dirs": {k: str(v) for k, v in trainer.run_dirs.items()},
        "artifacts": artifacts,
        "label_vocab": trainer.label_vocab,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConsentShield training framework")
    p.add_argument("--task", choices=["text", "vision", "unified"], default="text")
    p.add_argument("--config", type=str, default=None, help="Optional YAML override file")
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Validate data/config pipeline only (no weight updates)",
    )
    p.add_argument(
        "--allow-train",
        action="store_true",
        help="Run real MiniLM training (text task). Vision remains Phase-3 gated.",
    )
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Cap dataset size (smoke tests)")
    p.add_argument(
        "--smoke-batches",
        type=int,
        default=None,
        help="If set with --allow-train, run a 1-epoch smoke using this many batches",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {}
    if args.run_name:
        overrides["run_name"] = args.run_name
    if args.max_samples is not None:
        overrides = deep_merge(overrides, {"data": {"max_samples": args.max_samples}})

    task_key = args.task if args.task != "unified" else "text"
    cfg = load_config(task_key, config_path=args.config, overrides=overrides or None)
    if args.task == "unified":
        cfg["task"] = "unified"

    # Default to dry-run unless --allow-train is set
    if not args.allow_train or args.dry_run:
        if args.allow_train and args.dry_run:
            print("Note: --dry-run overrides --allow-train", file=sys.stderr)
        report = dry_run(args.task, cfg)
        print(json.dumps(report, indent=2))
        print(f"\nDry-run OK -> {report['report_path']}")
        return 0

    if args.task in ("vision", "image", "clip"):
        print(
            "ERROR: Vision/CLIP training is still Phase-3 gated.\n"
            "Use --task text --allow-train for MiniLM Phase 2.",
            file=sys.stderr,
        )
        return 2

    if args.task == "unified":
        print("ERROR: --allow-train requires --task text for MiniLM.", file=sys.stderr)
        return 2

    result = run_text_training(cfg, max_batches=args.smoke_batches)
    print(json.dumps(result, indent=2, default=str))
    print(f"\nPhase 2 training finished. Models -> {result['run_dirs'].get('models')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
