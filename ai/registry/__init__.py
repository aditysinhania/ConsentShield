"""Dependency-injection model registry."""

from ai.registry.interfaces import EmbeddingEngine, ReasoningEngine, TextClassifier, VisionClassifier
from ai.registry.model_registry import ModelRegistry

__all__ = [
    "ModelRegistry",
    "TextClassifier",
    "VisionClassifier",
    "EmbeddingEngine",
    "ReasoningEngine",
]
