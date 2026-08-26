"""
Phase 2 MiniLM fine-tuning entrypoint.

Usage:
  python -m ai.training.train_minilm
  python -m ai.training.train_minilm --epochs 3 --batch-size 16
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from ai.training.callbacks.checkpoint import save_checkpoint
from ai.training.datasets.collate import collate_text_batch, create_dataloader
from ai.training.datasets.text_dataset import TextDataset
from ai.training.metrics.report import build_evaluation_payload, write_evaluation_reports
from ai.training.trainers.text_trainer import TextTrainer
from ai.training.trainers.utils import REPO_ROOT, deep_merge, load_config, set_seed


def _plot_curves(history: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback CSV for curves if matplotlib missing
        csv_path = out_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_f1,val_accuracy,lr\n")
            for row in history:
                m = row.get("metrics") or {}
                f.write(
                    f"{row.get('epoch')},{row.get('train_loss')},{row.get('val_loss')},"
                    f"{m.get('val_f1')},{m.get('val_accuracy')},{row.get('learning_rate')}\n"
                )
        return

    epochs = [r.get("epoch") for r in history]
    train_loss = [r.get("train_loss") for r in history]
    val_loss = [r.get("val_loss") for r in history]
    val_f1 = [(r.get("metrics") or {}).get("val_f1") for r in history]
    val_acc = [(r.get("metrics") or {}).get("val_accuracy") for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train_loss", marker="o")
    axes[0].plot(epochs, val_loss, label="val_loss", marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss curves")

    axes[1].plot(epochs, val_f1, label="val_f1", marker="o")
    axes[1].plot(epochs, val_acc, label="val_accuracy", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].legend()
    axes[1].set_title("Validation metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def build_text_datasets(cfg: dict[str, Any]) -> dict[str, TextDataset]:
    paths = cfg.get("paths") or {}
    data = cfg.get("data") or {}
    data_dir = Path(paths.get("data_dir") or "datasets/training")
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    label_field = str(data.get("label_field", "label_binary"))
    min_chars = int(data.get("min_text_chars", 1))
    sources = data.get("sources")
    max_samples = data.get("max_samples")

    common: dict[str, Any] = {
        "label_field": label_field,
        "repo_root": REPO_ROOT,
        "min_text_chars": min_chars,
    }
    if sources:
        common["sources"] = sources
    if max_samples is not None:
        common["max_samples"] = max_samples

    train_ds = TextDataset(
        "train",
        path=data_dir / "text_train.jsonl",
        **common,
    )
    vocab = train_ds.label_vocab
    val_ds = TextDataset(
        "val",
        path=data_dir / "text_val.jsonl",
        label_vocab=vocab,
        **common,
    )
    test_ds = TextDataset(
        "test",
        path=data_dir / "text_test.jsonl",
        label_vocab=vocab,
        **common,
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def build_loaders(datasets: dict[str, TextDataset], cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data") or {}
    train_cfg = cfg.get("train") or {}
    batch_size = int(train_cfg.get("batch_size", 32))
    kwargs = dict(
        batch_size=batch_size,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", False)),
        persistent_workers=bool(data.get("persistent_workers", False)),
        collate_fn=collate_text_batch,
    )
    return {
        "train": create_dataloader(
            datasets["train"],
            shuffle=bool(train_cfg.get("shuffle_train", True)),
            drop_last=bool(data.get("drop_last", False)),
            **kwargs,
        ),
        "val": create_dataloader(datasets["val"], shuffle=False, drop_last=False, **kwargs),
        "test": create_dataloader(datasets["test"], shuffle=False, drop_last=False, **kwargs),
    }


def save_phase2_artifacts(
    trainer: TextTrainer,
    *,
    history: list[dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Path]:
    models_dir = trainer.run_dirs["models"]
    paths: dict[str, Path] = {}

    latest_src = models_dir / str((cfg.get("checkpoint") or {}).get("latest_name", "latest.pt"))
    best_src = models_dir / str((cfg.get("checkpoint") or {}).get("best_name", "best.pt"))

    best_dst = models_dir / "best_model.pt"
    last_dst = models_dir / "last_model.pt"
    state_dst = models_dir / "training_state.pt"
    config_dst = models_dir / "training_config.yaml"

    def _replace_copy(src: Path, dst: Path) -> None:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)

    if best_src.is_file():
        _replace_copy(best_src, best_dst)
    elif latest_src.is_file():
        _replace_copy(latest_src, best_dst)
    else:
        save_checkpoint(
            best_dst,
            model=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            epoch=len(history),
            metrics=(history[-1].get("metrics") if history else {}) or {},
            config=cfg,
            label_vocab=trainer.label_vocab,
        )

    if latest_src.is_file():
        _replace_copy(latest_src, last_dst)
    else:
        _replace_copy(best_dst, last_dst)

    save_checkpoint(
        state_dst,
        model=trainer.model,
        optimizer=trainer.optimizer,
        scheduler=trainer.scheduler,
        epoch=int((history[-1].get("epoch") if history else 0) or len(history)),
        metrics=(history[-1].get("metrics") if history else {}) or {},
        config=cfg,
        label_vocab=trainer.label_vocab,
        extra={"history": history},
    )
    config_dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    paths["best_model"] = best_dst
    paths["last_model"] = last_dst
    paths["training_state"] = state_dst
    paths["training_config"] = config_dst
    return paths


def evaluate_split(
    trainer: TextTrainer,
    loader: Any,
    *,
    reports_dir: Path,
    stem: str,
) -> dict[str, Any]:
    details = trainer.evaluate_detailed(loader)
    id2label = details["id2label"]
    y_true_labels = [id2label[i] for i in details["y_true"]]
    y_pred_labels = [id2label[i] for i in details["y_pred"]]
    payload = build_evaluation_payload(
        y_true_labels,
        y_pred_labels,
        ids=details["ids"],
        label_names={lab: lab for lab in id2label.values()},
        extra={"numeric_metrics": details["metrics"], "confusion_matrix": details["confusion_matrix"]},
    )
    paths = write_evaluation_reports(payload, reports_dir, stem=stem)
    return {"metrics": details["metrics"], "confusion_matrix": details["confusion_matrix"], "paths": paths, "payload": payload}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 2 MiniLM fine-tuning")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Optional cap for smoke tests")
    p.add_argument("--run-name", type=str, default="minilm")
    p.add_argument("--device", type=str, default=None)
    p.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume from checkpoint path, or 'auto' for runs/<run>/models/best.pt",
    )
    return p.parse_args(argv)


def _resolve_resume_path(args: argparse.Namespace, cfg: dict[str, Any]) -> Path | None:
    if args.resume is None:
        return None
    if args.resume != "auto":
        path = Path(args.resume)
        return path if path.is_file() else None
    run_name = args.run_name or cfg.get("run_name") or "minilm"
    candidates = [
        REPO_ROOT / "runs" / run_name / "models" / "training_state.pt",
        REPO_ROOT / "runs" / run_name / "models" / "best.pt",
        REPO_ROOT / "runs" / run_name / "models" / "best_model.pt",
        REPO_ROOT / "runs" / run_name / "models" / "latest.pt",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {
        "run_name": args.run_name,
        "paths": {
            "data_dir": "datasets/training",
            "output_root": "runs",
        },
        "data": {
            "sources": ["hf_synthetic", "hf_manual", "contextdp"],
            "label_field": "label_binary",
        },
        "checkpoint": {
            "best_name": "best.pt",
            "latest_name": "latest.pt",
            "monitor": "val_f1",
            "mode": "max",
        },
        "train": {
            "mixed_precision": True,
        },
    }
    if args.epochs is not None:
        overrides = deep_merge(overrides, {"train": {"epochs": args.epochs}})
    if args.batch_size is not None:
        overrides = deep_merge(overrides, {"train": {"batch_size": args.batch_size}})
    if args.max_samples is not None:
        overrides = deep_merge(overrides, {"data": {"max_samples": args.max_samples}})
    if args.device is not None:
        overrides = deep_merge(overrides, {"hardware": {"device": args.device}})

    cfg = load_config("text", config_path=args.config, overrides=overrides)
    set_seed(int((cfg.get("reproducibility") or {}).get("seed", 42)))

    datasets = build_text_datasets(cfg)
    loaders = build_loaders(datasets, cfg)

    trainer = TextTrainer(cfg, allow_train=True)
    trainer.setup(num_labels=len(datasets["train"].label_vocab), label_vocab=datasets["train"].label_vocab)

    start_epoch = 1
    resume_path = _resolve_resume_path(args, cfg)
    prior_history: list[dict[str, Any]] = []
    if resume_path is not None:
        start_epoch = trainer.load_training_state(resume_path)
        # Keep prior epoch-1 metrics in the final report when available
        summary_prev = trainer.run_dirs["reports"] / "summary_report.json"
        if summary_prev.is_file():
            try:
                prior_history = list(json.loads(summary_prev.read_text(encoding="utf-8")).get("history") or [])
            except Exception:
                prior_history = []
        print(json.dumps({"resuming_from": str(resume_path), "start_epoch": start_epoch}, indent=2))

    print(
        json.dumps(
            {
                "device": trainer.device,
                "amp": trainer.use_amp,
                "train_samples": len(datasets["train"]),
                "val_samples": len(datasets["val"]),
                "test_samples": len(datasets["test"]),
                "label_vocab": datasets["train"].label_vocab,
                "start_epoch": start_epoch,
                "epochs": (cfg.get("train") or {}).get("epochs"),
            },
            indent=2,
        )
    )

    fit_result = trainer.fit(loaders["train"], loaders["val"], start_epoch=start_epoch)
    history = prior_history + fit_result["history"]

    artifact_paths = save_phase2_artifacts(trainer, history=history, cfg=cfg)
    reports_dir = trainer.run_dirs["reports"]
    _plot_curves(history, reports_dir / "training_curves.png")

    # Reload best weights for final val/test evaluation when available
    best_path = artifact_paths["best_model"]
    if best_path.is_file() and trainer.model is not None:
        from ai.training.callbacks.checkpoint import load_checkpoint

        ckpt = load_checkpoint(best_path, map_location=trainer.device)
        if "model_state_dict" in ckpt:
            trainer.model.load_state_dict(ckpt["model_state_dict"])

    val_eval = evaluate_split(trainer, loaders["val"], reports_dir=reports_dir, stem="val_evaluation")
    test_eval = evaluate_split(trainer, loaders["test"], reports_dir=reports_dir, stem="test_evaluation")

    summary = {
        "phase": 2,
        "model": (cfg.get("model") or {}).get("name"),
        "device": trainer.device,
        "mixed_precision": trainer.use_amp,
        "datasets": {k: v.summary() for k, v in datasets.items()},
        "history": history,
        "validation_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "val_confusion_matrix": val_eval["confusion_matrix"],
        "test_confusion_matrix": test_eval["confusion_matrix"],
        "best_checkpoint": str(best_path),
        "artifacts": {k: str(v) for k, v in artifact_paths.items()},
        "report_paths": {
            "val": {k: str(v) for k, v in val_eval["paths"].items()},
            "test": {k: str(v) for k, v in test_eval["paths"].items()},
        },
    }
    summary_path = reports_dir / "summary_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        "# MiniLM Phase 2 — summary report",
        "",
        f"- Device: `{trainer.device}` (AMP={trainer.use_amp})",
        f"- Best checkpoint: `{best_path}`",
        "",
        "## Validation metrics",
        "",
        "```json",
        json.dumps(val_eval["metrics"], indent=2),
        "```",
        "",
        "## Test metrics",
        "",
        "```json",
        json.dumps(test_eval["metrics"], indent=2),
        "```",
        "",
        "## Test confusion matrix",
        "",
        "```json",
        json.dumps(test_eval["confusion_matrix"], indent=2),
        "```",
        "",
    ]
    (reports_dir / "summary_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "best_checkpoint": str(best_path), "test_metrics": test_eval["metrics"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
