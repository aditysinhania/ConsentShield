"""Evaluation report writers (JSON / Markdown / CSV)."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from ai.training.metrics.classification import (
    compute_classification_metrics,
    per_class_accuracy,
    sklearn_classification_report,
)
from ai.training.metrics.confusion import confusion_matrix_dict


def build_evaluation_payload(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    ids: Sequence[str] | None = None,
    label_names: dict[Any, str] | None = None,
    labels: Sequence[Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    name_map = label_names or {}
    ordered = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    target_names = [name_map.get(i, str(i)) for i in ordered]
    metrics = compute_classification_metrics(y_true, y_pred, labels=ordered)
    cm = confusion_matrix_dict(
        y_true,
        y_pred,
        labels=ordered,
        label_names=target_names,
    )
    report_txt = sklearn_classification_report(
        y_true,
        y_pred,
        labels=ordered,
        target_names=target_names,
    )
    payload: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "num_samples": len(list(y_true)),
        "metrics": metrics,
        "per_class_accuracy": per_class_accuracy(y_true, y_pred, label_names=name_map),
        "confusion_matrix": cm,
        "classification_report": report_txt,
    }
    if ids is not None:
        payload["predictions"] = [
            {"id": i, "y_true": t, "y_pred": p}
            for i, t, p in zip(ids, y_true, y_pred, strict=False)
        ]
    if extra:
        payload["extra"] = extra
    return payload


def write_evaluation_reports(
    payload: dict[str, Any],
    output_dir: str | Path,
    *,
    stem: str = "evaluation",
) -> dict[str, Path]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    json_path = out / f"{stem}.json"
    md_path = out / f"{stem}.md"
    csv_path = out / f"{stem}_predictions.csv"

    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    metrics = payload.get("metrics") or {}
    lines = [
        f"# Evaluation report — `{stem}`",
        "",
        f"Generated: `{payload.get('generated_at')}`",
        f"Samples: **{payload.get('num_samples', 0)}**",
        "",
        "## Summary metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key in ("accuracy", "precision", "recall", "f1", "macro_f1", "weighted_f1"):
        if key in metrics:
            lines.append(f"| {key} | {metrics[key]:.4f} |")
    lines.extend(["", "## Per-class accuracy", ""])
    for name, val in (payload.get("per_class_accuracy") or {}).items():
        lines.append(f"- `{name}`: {val:.4f}")
    lines.extend(["", "## Classification report", "", "```", payload.get("classification_report", ""), "```", ""])
    cm = payload.get("confusion_matrix") or {}
    lines.extend(["## Confusion matrix", "", f"Labels: {cm.get('label_names')}", "", "```"])
    for row in cm.get("matrix") or []:
        lines.append("  ".join(str(x) for x in row))
    lines.extend(["```", ""])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    preds = payload.get("predictions") or []
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "y_true", "y_pred"])
        writer.writeheader()
        for row in preds:
            writer.writerow(row)

    return {"json": json_path, "markdown": md_path, "predictions_csv": csv_path}
