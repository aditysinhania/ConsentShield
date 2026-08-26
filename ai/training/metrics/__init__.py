"""Metrics package."""

from __future__ import annotations

from ai.training.metrics.classification import compute_classification_metrics, per_class_accuracy
from ai.training.metrics.confusion import compute_confusion_matrix, confusion_matrix_dict
from ai.training.metrics.report import build_evaluation_payload, write_evaluation_reports

__all__ = [
    "compute_classification_metrics",
    "per_class_accuracy",
    "compute_confusion_matrix",
    "confusion_matrix_dict",
    "build_evaluation_payload",
    "write_evaluation_reports",
]
