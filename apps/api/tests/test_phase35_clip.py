"""Phase 3.5 — Fine-tuned CLIP inference integration tests."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import pytest

from ai.common.types import ScanPayload
from ai.datasets.fixtures.validation_sites import guardian_payload
from ai.inference.pipeline import InferencePipeline
from ai.models.cache import clear_cache
from ai.models.config import Phase4ModelConfig
from ai.registry.model_registry import ModelRegistry
from ai.training.checkpoint_loader import resolve_clip_checkpoint
from ai.vision.detectors.vision_detector import VisionDetector

CHECKPOINT = Path("models/checkpoints/clip/best_model.pt")


def _checkpoint_available() -> bool:
    return resolve_clip_checkpoint(CHECKPOINT) is not None


pytestmark = pytest.mark.skipif(
    not _checkpoint_available(),
    reason="models/checkpoints/clip/best_model.pt not present",
)


def _tiny_png_b64() -> str:
    from PIL import Image

    img = Image.new("RGB", (64, 64), color=(240, 240, 250))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture(scope="module")
def clip_registry() -> ModelRegistry:
    clear_cache()
    cfg = Phase4ModelConfig(
        models_root=Path("./models"),
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vision_model_name="openai/clip-vit-base-patch32",
        device="cpu",
        embedding_batch_size=8,
        cache_models=True,
        stub_mode=False,
        backend="pretrained",
        minilm_checkpoint="__disabled__",
        clip_checkpoint=str(CHECKPOINT),
        clip_finetuned_enabled=True,
    )
    return ModelRegistry.from_config(cfg)


@pytest.fixture(scope="module")
def clip_pipeline(clip_registry: ModelRegistry) -> InferencePipeline:
    return InferencePipeline(registry=clip_registry)


def test_checkpoint_loads(clip_registry: ModelRegistry):
    status = clip_registry.finetuned_clip_status()
    assert status["loaded"] is True
    assert status["status"] == "loaded"
    assert status["enabled"] is True
    assert status["checkpoint"]
    assert "best_model.pt" in str(status["checkpoint"]).replace("\\", "/")
    assert status["model_version"]
    assert clip_registry.vision.is_ready()
    inner = getattr(clip_registry.vision, "_inner", None)
    assert inner is not None
    assert getattr(inner, "_finetuned", None) is not None


def test_inference_returns_vision_class(clip_registry: ModelRegistry):
    detector = VisionDetector(
        config=clip_registry.config,
        finetuned_bundle=clip_registry._finetuned_clip_bundle,
    )
    payload = ScanPayload(
        url="https://example.com",
        screenshot_base64=_tiny_png_b64(),
        css_snapshot={
            "banner": {"width": 800, "height": 120, "text": "We use cookies"},
            "buttons": [{"text": "Accept all"}, {"text": "Reject"}],
        },
    )
    feats = detector.extract_features(payload)
    assert feats.status == "ready"
    assert feats.backend == "finetuned_clip"
    assert feats.predicted_class
    assert feats.confidence is not None
    assert feats.class_probabilities
    assert abs(sum(feats.class_probabilities.values()) - 1.0) < 1e-2
    layout = feats.layout or {}
    assert layout.get("backend") == "finetuned_clip"
    assert len(layout.get("top_3_class_probabilities") or []) == 3


def test_report_contains_finetuned_clip(clip_pipeline: InferencePipeline):
    payload = guardian_payload()
    # Attach a tiny screenshot so finetuned path can run
    payload = payload.model_copy(update={"screenshot_base64": _tiny_png_b64()})
    report = clip_pipeline.run(payload)
    assert report.vision is not None
    assert report.vision.status == "ready"
    assert report.vision.backend == "finetuned_clip"
    assert "Vision (Fine-tuned CLIP)" in (report.models_used or [])
    assert report.ai_analysis is not None
    vis = report.ai_analysis["vision"]
    assert vis["name"] == "Fine-tuned CLIP"
    assert vis["predicted_class"]
    assert vis["confidence"] is not None
    assert len(vis["top_3_class_probabilities"]) == 3


def test_rules_remain_dominant(clip_pipeline: InferencePipeline):
    payload = guardian_payload().model_copy(update={"screenshot_base64": _tiny_png_b64()})
    report = clip_pipeline.run(payload)
    assert report.rules is not None
    assert len(report.rules.hits) >= 1
    assert 30 <= report.risk_score <= 65
    assert report.fusion is not None
    assert "rule" in (report.fusion.message or "").lower()


def test_fallback_when_checkpoint_disabled():
    clear_cache()
    cfg = Phase4ModelConfig(
        models_root=Path("./models"),
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vision_model_name="openai/clip-vit-base-patch32",
        device="cpu",
        embedding_batch_size=8,
        cache_models=True,
        stub_mode=False,
        backend="lexical",
        minilm_checkpoint="__disabled__",
        clip_checkpoint="__disabled__",
        clip_finetuned_enabled=True,
    )
    reg = ModelRegistry.from_config(cfg)
    assert reg.finetuned_clip_status()["loaded"] is False
    det = VisionDetector(config=cfg)
    feats = det.extract_features(
        ScanPayload(
            url="https://example.com",
            css_snapshot={"banner": {"width": 100, "height": 40}},
        )
    )
    assert feats.status == "ready"
    assert feats.backend == "lexical"
