"""
Reusable evaluation pipeline for ConsentShield.

Usage:
  python -m ai.training.evaluate --baseline majority --split val
  python -m ai.training.evaluate --checkpoint runs/minilm/models/best_model.pt --split test
  python -m ai.training.evaluate --predictions path/to/preds.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from ai.training.datasets.collate import collate_text_batch, create_dataloader
from ai.training.datasets.text_dataset import TextDataset
from ai.training.metrics.report import build_evaluation_payload, write_evaluation_reports
from ai.training.trainers.utils import REPO_ROOT, ensure_run_dirs, load_config, resolve_device


def load_predictions_csv(path: Path) -> tuple[list[str], list[Any], list[Any]]:
    ids: list[str] = []
    y_true: list[Any] = []
    y_pred: list[Any] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(str(row.get("id", len(ids))))
            y_true.append(row["y_true"])
            y_pred.append(row["y_pred"])
    return ids, y_true, y_pred


def build_text_datasets(cfg: dict[str, Any]) -> dict[str, TextDataset]:
    paths = cfg.get("paths") or {}
    data = cfg.get("data") or {}
    data_dir = Path(paths.get("data_dir") or "datasets/training")
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    label_field = str(data.get("label_field", "label_binary"))
    min_chars = int(data.get("min_text_chars", 1))
    common: dict[str, Any] = {
        "label_field": label_field,
        "repo_root": REPO_ROOT,
        "min_text_chars": min_chars,
    }
    if data.get("sources"):
        common["sources"] = data["sources"]
    if data.get("max_samples") is not None:
        common["max_samples"] = data["max_samples"]

    train_ds = TextDataset("train", path=data_dir / "text_train.jsonl", **common)
    vocab = train_ds.label_vocab
    val_ds = TextDataset("val", path=data_dir / "text_val.jsonl", label_vocab=vocab, **common)
    test_ds = TextDataset("test", path=data_dir / "text_test.jsonl", label_vocab=vocab, **common)
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ConsentShield evaluation framework")
    p.add_argument("--task", choices=["text", "vision", "unified"], default="text")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--baseline", choices=["majority"], default=None)
    p.add_argument("--predictions", type=str, default=None, help="CSV with id,y_true,y_pred")
    p.add_argument("--checkpoint", type=str, default=None, help="MiniLM checkpoint (.pt)")
    p.add_argument("--run-name", type=str, default="minilm")
    p.add_argument("--output-stem", type=str, default="evaluation")
    return p.parse_args(argv)


def _eval_checkpoint(args: argparse.Namespace, cfg: dict[str, Any], run_dirs: dict[str, Path]) -> int:
    from ai.training.checkpoint_loader import load_minilm_classifier, resolve_minilm_checkpoint
    from ai.training.trainers.text_trainer import TextTrainer

    ckpt_path = Path(args.checkpoint) if args.checkpoint else resolve_minilm_checkpoint(None)
    if ckpt_path is None:
        print(
            "No checkpoint found. Use --baseline majority, --predictions CSV, "
            "or --checkpoint path/to/best_model.pt",
            file=sys.stderr,
        )
        return 2

    datasets = build_text_datasets(cfg)
    device = resolve_device((cfg.get("hardware") or {}).get("device"))
    model, meta = load_minilm_classifier(ckpt_path, device=device)
    trainer = TextTrainer(cfg, allow_train=True)
    trainer.model = model
    trainer.label_vocab = meta.get("label_vocab") or datasets["train"].label_vocab
    trainer.device = device

    data = cfg.get("data") or {}
    train_cfg = cfg.get("train") or {}
    loader = create_dataloader(
        datasets[args.split],
        batch_size=int(train_cfg.get("batch_size", 32)),
        shuffle=False,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", False)),
        collate_fn=collate_text_batch,
    )
    details = trainer.evaluate_detailed(loader)
    id2label = details["id2label"]
    y_true_labels = [id2label[i] for i in details["y_true"]]
    y_pred_labels = [id2label[i] for i in details["y_pred"]]
    payload = build_evaluation_payload(
        y_true_labels,
        y_pred_labels,
        ids=details["ids"],
        label_names={lab: lab for lab in id2label.values()},
        extra={"numeric_metrics": details["metrics"], "checkpoint": str(ckpt_path)},
    )
    paths = write_evaluation_reports(payload, run_dirs["reports"], stem=args.output_stem)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    print(payload["classification_report"])
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {
        "run_name": args.run_name,
        "paths": {"data_dir": "datasets/training", "output_root": "runs"},
        "data": {
            "sources": ["hf_synthetic", "hf_manual", "contextdp"],
            "label_field": "label_binary",
        },
    }
    cfg = load_config("text" if args.task != "vision" else "vision", config_path=args.config, overrides=overrides)
    run_dirs = ensure_run_dirs(cfg)

    if args.baseline is None and args.predictions is None:
        return _eval_checkpoint(args, cfg, run_dirs)

    if args.predictions:
        ids, y_true, y_pred = load_predictions_csv(Path(args.predictions))
        extra: dict[str, Any] = {}
        label_names = None
    elif args.baseline == "majority":
        datasets = build_text_datasets(cfg)
        train = datasets["train"]
        eval_ds = datasets[args.split]
        counts = Counter(str(s[train.label_field]) for s in train.samples)
        majority = counts.most_common(1)[0][0]
        ids, y_true, y_pred = [], [], []
        for i in range(len(eval_ds)):
            item = eval_ds[i]
            ids.append(item["id"])
            y_true.append(item["label"])
            y_pred.append(majority)
        extra = {"baseline": "majority", "majority_label": majority, "train_counts": dict(counts)}
        label_names = {lab: lab for lab in train.label_vocab}
    else:
        print("Use --baseline majority, --predictions CSV, or --checkpoint path", file=sys.stderr)
        return 2

    payload = build_evaluation_payload(y_true, y_pred, ids=ids, label_names=label_names, extra=extra)
    paths = write_evaluation_reports(payload, run_dirs["reports"], stem=args.output_stem)
    print(json.dumps({k: str(v) for k, v in paths.items()}, indent=2))
    print(payload["classification_report"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
