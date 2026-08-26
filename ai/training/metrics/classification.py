"""Classification metrics for binary and multiclass tasks."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)


def _as_arrays(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(y_true), np.asarray(y_pred)


def compute_classification_metrics(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    average: str | None = None,
) -> dict[str, float]:
    """
    Accuracy, precision, recall, F1 (macro/weighted/binary-aware).

    If ``average`` is None, returns macro + weighted F1 plus overall accuracy.
    """
    yt, yp = _as_arrays(y_true, y_pred)
    if yt.size == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
        }

    uniq = sorted(set(yt.tolist()) | set(yp.tolist()))
    is_binary = len(uniq) <= 2
    avg = average or ("binary" if is_binary else "macro")
    if avg == "binary" and not is_binary:
        avg = "macro"

    pos_label = uniq[-1] if is_binary and avg == "binary" else 1

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(yt, yp)),
        "macro_f1": float(f1_score(yt, yp, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(f1_score(yt, yp, average="weighted", labels=labels, zero_division=0)),
    }

    kwargs: dict[str, Any] = {"average": avg, "zero_division": 0}
    if labels is not None:
        kwargs["labels"] = labels
    if avg == "binary":
        kwargs["pos_label"] = pos_label

    metrics["precision"] = float(precision_score(yt, yp, **kwargs))
    metrics["recall"] = float(recall_score(yt, yp, **kwargs))
    metrics["f1"] = float(f1_score(yt, yp, **kwargs))
    return metrics


def per_class_accuracy(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    label_names: dict[Any, str] | None = None,
) -> dict[str, float]:
    yt, yp = _as_arrays(y_true, y_pred)
    out: dict[str, float] = {}
    for lab in sorted(set(yt.tolist())):
        mask = yt == lab
        name = label_names.get(lab, str(lab)) if label_names else str(lab)
        if not np.any(mask):
            out[name] = 0.0
        else:
            out[name] = float(np.mean(yp[mask] == lab))
    return out


def sklearn_classification_report(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    target_names: Sequence[str] | None = None,
    labels: Sequence[Any] | None = None,
) -> str:
    return classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
