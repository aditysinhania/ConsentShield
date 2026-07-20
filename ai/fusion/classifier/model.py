"""Fusion classifier stub — XGBoost / sklearn later."""

from __future__ import annotations

from typing import Any


class FusionClassifier:
    """Learned fusion head. Unloaded until models/checkpoints provides weights."""

    name = "fusion_classifier"
    version = "0.1.0"

    def __init__(self) -> None:
        self._loaded = False

    def predict_proba(self, features: list[float]) -> dict[str, Any]:
        return {
            "status": "not_loaded",
            "message": "Learned fusion classifier not loaded; using rule-based aggregator.",
            "features": features,
        }

    def is_ready(self) -> bool:
        return self._loaded
