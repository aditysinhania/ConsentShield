"""
Phase 3 CLIP fine-tuning entrypoint (Colab / local).

Usage:
  python -m ai.training.train_clip
  python -m ai.training.train_clip --epochs 5 --batch-size 32
  python -m ai.training.train_clip --max-samples 64 --epochs 1   # smoke
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import yaml

from ai.training.callbacks.checkpoint import load_checkpoint, save_checkpoint
from ai.training.datasets.collate import collate_image_batch, create_dataloader
from ai.training.datasets.image_dataset import ImageDataset
from ai.training.datasets.image_transforms import build_eval_transform, build_train_transform
from ai.training.metrics.classification import sklearn_classification_report
from ai.training.metrics.report import build_evaluation_payload, write_evaluation_reports
from ai.training.models.clip_classifier import build_clip_processor, clip_image_mean_std
from ai.training.trainers.utils import REPO_ROOT, deep_merge, load_config, set_seed
from ai.training.trainers.vision_trainer import VisionTrainer

VISION_CLASS_ORDER = (
    "No Dark Pattern",
    "Visual Manipulation",
    "Hidden Choice",
    "Forced Action",
    "Privacy Friction",
)


def _plot_curves(history: list[dict[str, Any]], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        csv_path = out_path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8") as f:
            f.write("epoch,train_loss,val_loss,val_macro_f1,val_accuracy,lr\n")
            for row in history:
                m = row.get("metrics") or {}
                f.write(
                    f"{row.get('epoch')},{row.get('train_loss')},{row.get('val_loss')},"
                    f"{m.get('val_macro_f1')},{m.get('val_accuracy')},{row.get('learning_rate')}\n"
                )
        return

    epochs = [r.get("epoch") for r in history]
    train_loss = [r.get("train_loss") for r in history]
    val_loss = [r.get("val_loss") for r in history]
    val_f1 = [(r.get("metrics") or {}).get("val_macro_f1") for r in history]
    val_acc = [(r.get("metrics") or {}).get("val_accuracy") for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train_loss", marker="o")
    axes[0].plot(epochs, val_loss, label="val_loss", marker="o")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].set_title("Loss curves")

    axes[1].plot(epochs, val_f1, label="val_macro_f1", marker="o")
    axes[1].plot(epochs, val_acc, label="val_accuracy", marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("score")
    axes[1].legend()
    axes[1].set_title("Validation metrics")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _plot_confusion(cm: dict[str, Any], out_path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    matrix = np.asarray(cm.get("matrix") or [], dtype=float)
    names = list(cm.get("label_names") or [])
    if matrix.size == 0:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046)
    ax.set(
        xticks=range(len(names)),
        yticks=range(len(names)),
        xticklabels=names,
        yticklabels=names,
        ylabel="True",
        xlabel="Predicted",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fixed_label_vocab(train_ds: ImageDataset) -> dict[str, int]:
    """Prefer stable 5-class order; fall back to dataset vocab."""
    present = set(train_ds.label_vocab.keys())
    ordered = [c for c in VISION_CLASS_ORDER if c in present]
    for c in sorted(present):
        if c not in ordered:
            ordered.append(c)
    return {lab: i for i, lab in enumerate(ordered)}


def build_vision_datasets(cfg: dict[str, Any]) -> dict[str, ImageDataset]:
    paths = cfg.get("paths") or {}
    data = cfg.get("data") or {}
    model_cfg = cfg.get("model") or {}
    data_dir = Path(paths.get("data_dir") or "datasets/training/vision")
    if not data_dir.is_absolute():
        data_dir = REPO_ROOT / data_dir

    image_size = int(model_cfg.get("image_size", 224))
    processor = build_clip_processor(str(model_cfg.get("name", "openai/clip-vit-base-patch32")))
    mean, std = clip_image_mean_std(processor)
    train_tf = build_train_transform(image_size=image_size, mean=mean, std=std)
    eval_tf = build_eval_transform(image_size=image_size, mean=mean, std=std)

    label_field = str(data.get("label_field", "label_vision"))
    max_samples = data.get("max_samples")
    require = bool(data.get("require_image_exists", True))
    common: dict[str, Any] = {
        "label_field": label_field,
        "repo_root": REPO_ROOT,
        "require_image_exists": require,
    }
    if max_samples is not None:
        common["max_samples"] = max_samples

    train_ds = ImageDataset("train", path=data_dir / "train.jsonl", transform=train_tf, **common)
    vocab = _fixed_label_vocab(train_ds)
    # Rebuild train with fixed vocab so label_ids align
    train_ds = ImageDataset(
        "train",
        path=data_dir / "train.jsonl",
        transform=train_tf,
        label_vocab=vocab,
        **common,
    )
    val_ds = ImageDataset(
        "val",
        path=data_dir / "val.jsonl",
        transform=eval_tf,
        label_vocab=vocab,
        **common,
    )
    test_ds = ImageDataset(
        "test",
        path=data_dir / "test.jsonl",
        transform=eval_tf,
        label_vocab=vocab,
        **common,
    )
    return {"train": train_ds, "val": val_ds, "test": test_ds}


def build_loaders(datasets: dict[str, ImageDataset], cfg: dict[str, Any]) -> dict[str, Any]:
    data = cfg.get("data") or {}
    train_cfg = cfg.get("train") or {}
    batch_size = int(train_cfg.get("batch_size", 32))
    kwargs: dict[str, Any] = dict(
        batch_size=batch_size,
        num_workers=int(data.get("num_workers", 0)),
        pin_memory=bool(data.get("pin_memory", False)) and str(cfg.get("hardware", {}).get("device", "")).startswith("cuda"),
        persistent_workers=bool(data.get("persistent_workers", False)),
        collate_fn=collate_image_batch,
    )

    sampler = None
    if bool(data.get("use_sample_weights", True)):
        import torch
        from torch.utils.data import WeightedRandomSampler

        weights = datasets["train"].sample_weights()
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(weights, dtype=torch.double),
            num_samples=len(weights),
            replacement=True,
        )

    return {
        "train": create_dataloader(
            datasets["train"],
            shuffle=sampler is None and bool(train_cfg.get("shuffle_train", True)),
            drop_last=bool(data.get("drop_last", False)),
            sampler=sampler,
            **kwargs,
        ),
        "val": create_dataloader(datasets["val"], shuffle=False, drop_last=False, **kwargs),
        "test": create_dataloader(datasets["test"], shuffle=False, drop_last=False, **kwargs),
    }


def save_phase3_artifacts(
    trainer: VisionTrainer,
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


def export_public_artifacts(
    *,
    best_model: Path,
    reports_dir: Path,
    cfg: dict[str, Any],
    test_eval: dict[str, Any],
    history: list[dict[str, Any]],
    datasets: dict[str, ImageDataset],
) -> dict[str, Path]:
    paths_cfg = cfg.get("paths") or {}
    export_dir = Path(paths_cfg.get("export_dir") or "models/checkpoints/clip")
    if not export_dir.is_absolute():
        export_dir = REPO_ROOT / export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    dst_best = export_dir / "best_model.pt"
    if dst_best.exists() or dst_best.is_symlink():
        dst_best.unlink()
    shutil.copy2(best_model, dst_best)
    out["best_model"] = dst_best

    metrics = {
        "history": history,
        "test_metrics": test_eval.get("metrics") or {},
        "label_vocab": datasets["train"].label_vocab,
    }
    best_epoch = 0
    best_f1 = -1.0
    for row in history:
        f1 = float((row.get("metrics") or {}).get("val_macro_f1") or -1)
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = int(row.get("epoch") or 0)
    metrics["best_epoch"] = best_epoch
    metrics["best_val_macro_f1"] = best_f1

    metrics_path = export_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out["metrics"] = metrics_path

    cm = test_eval.get("confusion_matrix") or {}
    cm_png = export_dir / "confusion_matrix.png"
    _plot_confusion(cm, cm_png)
    if cm_png.is_file():
        out["confusion_matrix"] = cm_png

    curves_src = reports_dir / "training_curves.png"
    curves_dst = export_dir / "training_curves.png"
    if curves_src.is_file():
        shutil.copy2(curves_src, curves_dst)
        out["training_curves"] = curves_dst

    id2label = {i: lab for lab, i in datasets["train"].label_vocab.items()}
    y_true = test_eval.get("y_true_labels") or []
    y_pred = test_eval.get("y_pred_labels") or []
    report_txt = sklearn_classification_report(
        y_true,
        y_pred,
        labels=list(id2label.values()),
        target_names=[id2label[i] for i in sorted(id2label)],
    ) if y_true else (test_eval.get("payload") or {}).get("classification_report", "")
    report_md = export_dir / "classification_report.md"
    report_md.write_text(
        "# CLIP classification report\n\n```\n" + str(report_txt) + "\n```\n",
        encoding="utf-8",
    )
    out["classification_report"] = report_md

    # Top-level training report
    report_md_path = Path(paths_cfg.get("report_md") or "runs/clip_training/report.md")
    if not report_md_path.is_absolute():
        report_md_path = REPO_ROOT / report_md_path
    report_md_path.parent.mkdir(parents=True, exist_ok=True)
    ds_stats = {k: v.summary() for k, v in datasets.items()}
    vision_stats_path = REPO_ROOT / "datasets" / "training" / "vision" / "stats.json"
    vision_stats = {}
    if vision_stats_path.is_file():
        vision_stats = json.loads(vision_stats_path.read_text(encoding="utf-8"))

    lines = [
        "# ConsentShield Phase 3 — CLIP Training Report",
        "",
        f"- Model: `{(cfg.get('model') or {}).get('name')}`",
        f"- Best epoch (val Macro F1): **{best_epoch}**",
        f"- Best val Macro F1: **{best_f1:.4f}**",
        f"- Test accuracy: **{float((test_eval.get('metrics') or {}).get('accuracy') or 0):.4f}**",
        f"- Test Macro F1: **{float((test_eval.get('metrics') or {}).get('macro_f1') or 0):.4f}**",
        f"- Checkpoint: `{dst_best}`",
        "",
        "## Dataset statistics",
        "",
        "```json",
        json.dumps(vision_stats.get("class_table") or ds_stats, indent=2),
        "```",
        "",
        "## Test metrics",
        "",
        "```json",
        json.dumps(test_eval.get("metrics") or {}, indent=2),
        "```",
        "",
        "## Confusion matrix",
        "",
        "```json",
        json.dumps(cm, indent=2),
        "```",
        "",
        "## Notes",
        "",
        "- Text encoder + projection frozen; vision encoder + head trained.",
        "- Train sampling uses `sample_weight` (WeightedRandomSampler).",
        "- Fine-tuned CLIP is **not** enabled in inference by default "
        "(set `CLIP_FINETUNED_ENABLED=true` after evaluation).",
        "",
    ]
    report_md_path.write_text("\n".join(lines), encoding="utf-8")
    out["report_md"] = report_md_path
    return out


def evaluate_split(
    trainer: VisionTrainer,
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
    return {
        "metrics": details["metrics"],
        "confusion_matrix": details["confusion_matrix"],
        "paths": paths,
        "payload": payload,
        "y_true_labels": y_true_labels,
        "y_pred_labels": y_pred_labels,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 CLIP fine-tuning")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None, help="Cap samples for smoke tests")
    p.add_argument("--run-name", type=str, default="clip")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--grad-accum", type=int, default=None, help="Gradient accumulation steps")
    p.add_argument(
        "--resume",
        type=str,
        nargs="?",
        const="auto",
        default=None,
        help="Resume from checkpoint path, or 'auto'",
    )
    return p.parse_args(argv)


def _resolve_resume_path(args: argparse.Namespace, cfg: dict[str, Any]) -> Path | None:
    if args.resume is None:
        return None
    if args.resume != "auto":
        path = Path(args.resume)
        return path if path.is_file() else None
    run_name = args.run_name or cfg.get("run_name") or "clip"
    for path in [
        REPO_ROOT / "ai" / "training" / "runs" / run_name / "models" / "training_state.pt",
        REPO_ROOT / "ai" / "training" / "runs" / run_name / "models" / "best_model.pt",
        REPO_ROOT / "runs" / "clip" / "models" / "training_state.pt",
        REPO_ROOT / "models" / "checkpoints" / "clip" / "best_model.pt",
    ]:
        if path.is_file():
            return path
    return None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    overrides: dict[str, Any] = {"run_name": args.run_name}
    if args.epochs is not None:
        overrides = deep_merge(overrides, {"train": {"epochs": args.epochs}})
    if args.batch_size is not None:
        overrides = deep_merge(overrides, {"train": {"batch_size": args.batch_size}})
    if args.max_samples is not None:
        overrides = deep_merge(overrides, {"data": {"max_samples": args.max_samples}})
    if args.device is not None:
        overrides = deep_merge(overrides, {"hardware": {"device": args.device}})
    if args.grad_accum is not None:
        overrides = deep_merge(
            overrides, {"train": {"gradient_accumulation_steps": args.grad_accum}}
        )

    cfg = load_config("vision", config_path=args.config, overrides=overrides)
    set_seed(int((cfg.get("reproducibility") or {}).get("seed", 42)))

    datasets = build_vision_datasets(cfg)
    loaders = build_loaders(datasets, cfg)

    trainer = VisionTrainer(cfg, allow_train=True)
    trainer.setup(
        num_labels=len(datasets["train"].label_vocab),
        label_vocab=datasets["train"].label_vocab,
    )

    start_epoch = 1
    resume_path = _resolve_resume_path(args, cfg)
    prior_history: list[dict[str, Any]] = []
    if resume_path is not None:
        start_epoch = trainer.load_training_state(resume_path)
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
                "epochs": (cfg.get("train") or {}).get("epochs"),
                "batch_size": (cfg.get("train") or {}).get("batch_size"),
            },
            indent=2,
        )
    )

    fit_result = trainer.fit(loaders["train"], loaders["val"], start_epoch=start_epoch)
    history = prior_history + fit_result["history"]

    artifact_paths = save_phase3_artifacts(trainer, history=history, cfg=cfg)
    reports_dir = trainer.run_dirs["reports"]
    _plot_curves(history, reports_dir / "training_curves.png")

    best_path = artifact_paths["best_model"]
    if best_path.is_file() and trainer.model is not None:
        ckpt = load_checkpoint(best_path, map_location=trainer.device)
        if "model_state_dict" in ckpt:
            trainer.model.load_state_dict(ckpt["model_state_dict"])

    val_eval = evaluate_split(trainer, loaders["val"], reports_dir=reports_dir, stem="val_evaluation")
    test_eval = evaluate_split(trainer, loaders["test"], reports_dir=reports_dir, stem="test_evaluation")

    exported = export_public_artifacts(
        best_model=best_path,
        reports_dir=reports_dir,
        cfg=cfg,
        test_eval=test_eval,
        history=history,
        datasets=datasets,
    )

    summary = {
        "phase": 3,
        "model": (cfg.get("model") or {}).get("name"),
        "device": trainer.device,
        "mixed_precision": trainer.use_amp,
        "datasets": {k: v.summary() for k, v in datasets.items()},
        "history": history,
        "validation_metrics": val_eval["metrics"],
        "test_metrics": test_eval["metrics"],
        "best_checkpoint": str(exported["best_model"]),
        "exports": {k: str(v) for k, v in exported.items()},
    }
    (reports_dir / "summary_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "best_checkpoint": str(exported["best_model"]),
                "test_metrics": test_eval["metrics"],
                "report": str(exported.get("report_md")),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
