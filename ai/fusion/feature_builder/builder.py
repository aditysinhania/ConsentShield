"""Build a numeric feature vector from multimodal inputs."""

from __future__ import annotations

from ai.common.types import FusionInput


def build_features(inputs: FusionInput) -> list[float]:
    """Deterministic feature builder used by fusion (rules-dominant until ML loads)."""
    rules = inputs.rules
    vision = inputs.vision
    text = inputs.text

    risk = float(rules.normalized_risk) if rules else 0.0
    hit_count = float(len(rules.hits)) if rules else 0.0
    vision_ready = 1.0 if vision and vision.status == "ready" else 0.0
    text_ready = 1.0 if text and text.status == "ready" else 0.0
    text_conf = float(text.confidence) if text and text.confidence is not None else 0.0
    banner = 1.0 if vision and vision.banner_detected else 0.0

    return [risk / 100.0, hit_count, vision_ready, text_ready, text_conf, banner]
