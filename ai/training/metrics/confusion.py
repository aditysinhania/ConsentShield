"""Confusion matrix helpers."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def confusion_matrix_dict(
    y_true: Sequence[Any],
    y_pred: Sequence[Any],
    *,
    labels: Sequence[Any] | None = None,
    label_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    labs = list(labels) if labels is not None else sorted(set(y_true) | set(y_pred))
    cm = compute_confusion_matrix(y_true, y_pred, labels=labs)
    names = list(label_names) if label_names is not None else [str(x) for x in labs]
    return {
        "labels": labs,
        "label_names": names,
        "matrix": cm.tolist(),
    }
