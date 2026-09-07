"""Phase 2.5 — Fine-tuned MiniLM inference integration tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from ai.common.types import ScanPayload
from ai.datasets.fixtures.validation_sites import guardian_payload
from ai.inference.pipeline import InferencePipeline
from ai.models.cache import clear_cache
from ai.models.config import Phase4ModelConfig
from ai.registry.model_registry import ModelRegistry
from ai.text.classifier.finetuned_minilm import (
    REPORT_CLASSES,
    distribute_binary_to_report_classes,
    resolve_checkpoint,
)
from ai.text.classifier.text_classifier import TextClassifier


CHECKPOINT = Path("models/checkpoints/minilm/best_model.pt")


def _checkpoint_available() -> bool:
    return resolve_checkpoint(CHECKPOINT) is not None


pytestmark = pytest.mark.skipif(
    not _checkpoint_available(),
    reason="models/checkpoints/minilm/best_model.pt not present",
)


@pytest.fixture(scope="module")
def minilm_registry() -> ModelRegistry:
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
        minilm_checkpoint=str(CHECKPOINT),
    )
    return ModelRegistry.from_config(cfg)


@pytest.fixture(scope="module")
def minilm_pipeline(minilm_registry: ModelRegistry) -> InferencePipeline:
    return InferencePipeline(registry=minilm_registry)


def test_checkpoint_loads(minilm_registry: ModelRegistry):
    status = minilm_registry.finetuned_status()
    assert status["loaded"] is True
    assert status["status"] == "loaded"
    assert status["checkpoint"]
    assert "best_model.pt" in str(status["checkpoint"]).replace("\\", "/")
    assert status["model_version"]
    assert status["inference_device"] in ("cpu", "cuda")
    assert minilm_registry.text.is_ready()


def test_inference_returns_class_probabilities(minilm_pipeline: InferencePipeline):
    report = minilm_pipeline.run(guardian_payload())
    assert report.text is not None
    assert report.text.status == "ready"
    assert report.text.backend == "finetuned_minilm"
    assert report.text.predicted_class in REPORT_CLASSES
    probs = report.text.class_probabilities
    assert set(REPORT_CLASSES) <= set(probs.keys())
    assert abs(sum(probs.values()) - 1.0) < 1e-3
    assert report.text.confidence is not None
    assert report.ai_analysis is not None
    nlp = report.ai_analysis["nlp"]
    assert nlp["predicted_class"]
    assert nlp["confidence"] is not None
    assert len(nlp["top_3_class_probabilities"]) == 3


def test_report_contains_finetuned_minilm(minilm_pipeline: InferencePipeline):
    report = minilm_pipeline.run(guardian_payload())
    assert "NLP (Fine-tuned MiniLM)" in (report.models_used or [])
    assert report.ai_analysis is not None
    assert "Fine-tuned MiniLM" in str(report.ai_analysis.get("models_used"))
    assert report.ai_analysis["nlp"]["name"] == "Fine-tuned MiniLM"


def test_rule_findings_preserved_with_minilm(minilm_pipeline: InferencePipeline):
    report = minilm_pipeline.run(guardian_payload())
    assert report.rules is not None
    assert len(report.rules.hits) >= 1
    assert 30 <= report.risk_score <= 60
    # Fusion stays rule-dominant
    assert report.fusion is not None
    assert "rule" in (report.fusion.message or "").lower()


def test_fallback_when_checkpoint_missing():
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
        minilm_checkpoint="models/checkpoints/minilm/does_not_exist.pt",
    )
    reg = ModelRegistry.from_config(cfg)
    assert reg.finetuned_status()["loaded"] is False
    clf = TextClassifier(config=cfg, skip_finetuned_autoload=True)
    clf.bind_finetuned(None)
    clf._finetuned_error = "checkpoint missing"
    pred = clf.classify(
        ScanPayload(
            url="https://example.com",
            visible_text="Accept all cookies. Manage preferences.",
            css_snapshot={
                "banner": {"text": "We use cookies", "width": 100, "height": 40},
                "buttons": [{"text": "Accept all"}],
            },
        )
    )
    assert pred.status == "ready"
    assert pred.backend == "exemplar_similarity"


def test_distribute_binary_sums_to_one():
    probs = distribute_binary_to_report_classes(
        0.8,
        0.2,
        "Accept all cookies. Free trial will auto-renew. No thanks I hate privacy.",
    )
    assert set(probs.keys()) == set(REPORT_CLASSES)
    assert abs(sum(probs.values()) - 1.0) < 1e-5
    assert probs["no_dark_pattern"] == pytest.approx(0.2, abs=1e-5)
