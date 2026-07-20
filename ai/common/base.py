"""Base detector protocol — every AI module implements this contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ai.common.types import EvalMetrics, PredictionResult, TrainResult

TInput = TypeVar("TInput")
TOutput = TypeVar("TOutput", bound=PredictionResult)


class ModelNotLoadedError(RuntimeError):
    """Raised when predict() is called before a model artifact is loaded."""


class StubPredictionError(RuntimeError):
    """Raised by stub modules that must not invent predictions."""


class BaseDetector(ABC, Generic[TInput, TOutput]):
    """
    Common interface for Vision, Text, Rules, Fusion, and Explanation modules.

    Implementations must not fabricate dark-pattern labels. If a model is not
    loaded, raise ModelNotLoadedError or return an explicit stub status.
    """

    name: str = "base"
    version: str = "0.1.0"

    @abstractmethod
    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        """Train or fine-tune the underlying model."""

    @abstractmethod
    def predict(self, input_data: TInput) -> TOutput:
        """Run inference. Must not invent labels when models are unloaded."""

    @abstractmethod
    def evaluate(self, dataset: Any) -> EvalMetrics:
        """Evaluate on a labeled dataset."""

    @abstractmethod
    def load_model(self, path: str) -> None:
        """Load model weights / config from disk."""

    @abstractmethod
    def save_model(self, path: str) -> None:
        """Persist model weights / config to disk."""

    def is_ready(self) -> bool:
        """Return True when the module can produce real predictions."""
        return False
