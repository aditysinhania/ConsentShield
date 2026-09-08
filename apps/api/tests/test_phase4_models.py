"""Phase 4 — pretrained NLP/vision integration tests (lexical backend, no downloads)."""

import pytest

from ai.common.types import Category, ScanPayload
from ai.datasets.fixtures.validation_sites import (
    adobe_payload,
    bbc_payload,
    cookielawinfo_payload,
    guardian_payload,
    mozilla_payload,
    reuters_payload,
    spotify_payload,
)
from ai.inference.pipeline import InferencePipeline
from ai.models.cache import clear_cache
from ai.models.config import Phase4ModelConfig
from ai.registry.model_registry import ModelRegistry
from ai.report.document import build_report_document
from ai.report.exporters import export_html, export_markdown


@pytest.fixture
def phase4_registry():
    clear_cache()
    cfg = Phase4ModelConfig(
        models_root=__import__("pathlib").Path("./models"),
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vision_model_name="openai/clip-vit-base-patch32",
        device="cpu",
        embedding_batch_size=8,
        cache_models=True,
        stub_mode=False,
        backend="lexical",
        minilm_checkpoint="__disabled__",
        clip_checkpoint="__disabled__",
        clip_finetuned_enabled=False,
    )
    return ModelRegistry.from_config(cfg)


@pytest.fixture
def phase4_pipeline(phase4_registry) -> InferencePipeline:
    return InferencePipeline(registry=phase4_registry)


@pytest.fixture
def stub_pipeline() -> InferencePipeline:
    clear_cache()
    cfg = Phase4ModelConfig(
        models_root=__import__("pathlib").Path("./models"),
        text_model_name="sentence-transformers/all-MiniLM-L6-v2",
        vision_model_name="openai/clip-vit-base-patch32",
        device="cpu",
        embedding_batch_size=8,
        cache_models=True,
        stub_mode=True,
        backend="auto",
        minilm_checkpoint="__disabled__",
        clip_checkpoint="__disabled__",
        clip_finetuned_enabled=False,
    )
    return InferencePipeline(registry=ModelRegistry.from_config(cfg))


def test_registry_lazy_interfaces(phase4_registry):
    desc = phase4_registry.describe()
    assert len(desc) >= 5
    assert phase4_registry.text.is_ready()
    assert phase4_registry.vision.is_ready()
    assert phase4_registry.embedding.is_ready()


def test_stub_mode_preserves_not_loaded(stub_pipeline):
    report = stub_pipeline.run(guardian_payload())
    assert report.vision is not None
    assert report.vision.status == "not_loaded"
    assert report.text is not None
    assert report.text.status == "not_loaded"
    assert 30 <= report.risk_score <= 55


@pytest.mark.parametrize(
    "name,fn,lo,hi",
    [
        ("Guardian", guardian_payload, 30, 58),
        ("BBC", bbc_payload, 20, 58),
        ("Adobe", adobe_payload, 28, 58),
        ("Spotify", spotify_payload, 0, 25),
        ("Mozilla", mozilla_payload, 12, 45),
        ("Reuters", reuters_payload, 20, 58),
        ("CookieLawInfo", cookielawinfo_payload, 20, 55),
    ],
)
def test_phase4_site_bands_with_ai(name, fn, lo, hi, phase4_pipeline, stub_pipeline):
    deterministic = stub_pipeline.run(fn())
    assisted = phase4_pipeline.run(fn())
    assert lo <= assisted.risk_score <= hi, f"{name} assisted {assisted.risk_score}"
    # Rules remain source of truth: AI must not invent risk when rules found nothing
    if deterministic.risk_score == 0:
        assert assisted.risk_score == 0
    # With rule hits, AI may boost slightly but stay near deterministic
    if deterministic.risk_score > 0:
        assert assisted.risk_score >= deterministic.risk_score - 0.01
        assert assisted.risk_score <= deterministic.risk_score + 8.0
    assert assisted.ai_analysis is not None
    assert assisted.ai_analysis["fusion_contribution"]["rules"] >= 0
    assert assisted.confidence_breakdown is not None
    assert assisted.confidence_breakdown.rules >= 0
    assert assisted.confidence_breakdown.nlp >= 0
    assert assisted.confidence_breakdown.vision_model >= 0


def test_ai_does_not_invent_without_dom(phase4_pipeline):
    payload = ScanPayload(
        url="https://example.com/",
        visible_text="Hello world welcome to our blog",
        viewport={"width": 1280, "height": 800},
        css_snapshot={"buttons": [{"text": "Sign in", "width": 80, "height": 32}]},
    )
    report = phase4_pipeline.run(payload)
    assert report.risk_score == 0
    assert report.category == Category.NO_DARK_PATTERN


def test_nlp_outputs_similarity_structure(phase4_pipeline):
    report = phase4_pipeline.run(guardian_payload())
    assert report.text is not None
    assert report.text.status == "ready"
    assert report.ai_analysis is not None
    assert "nlp" in report.ai_analysis
    assert "similarity_scores" in report.ai_analysis


def test_vision_outputs_detections(phase4_pipeline):
    report = phase4_pipeline.run(guardian_payload())
    assert report.vision is not None
    assert report.vision.status == "ready"
    assert report.vision.banner_detected is True
    layout = report.vision.layout or {}
    assert layout.get("backend") in ("lexical", "pretrained", "pretrained-dom")


def test_report_includes_ai_analysis_section(phase4_pipeline):
    report = phase4_pipeline.run(reuters_payload())
    doc = build_report_document(
        scan_id="phase4",
        url="https://www.reuters.com/",
        title="Reuters",
        risk_score=report.risk_score,
        category=report.category.value,
        confidence=report.confidence,
        report=report.model_dump(),
    )
    html = export_html(doc)
    md = export_markdown(doc)
    assert "AI Analysis" in html
    assert "## AI Analysis" in md
    assert "fusion_contribution" in (doc.appendix.get("ai_analysis") or {})


def test_before_after_documentation_table(phase4_pipeline, stub_pipeline):
    """Document where AI assists vs pure rules (no site hardcoding of scores)."""
    rows = []
    for name, fn in [
        ("Guardian", guardian_payload),
        ("BBC", bbc_payload),
        ("Reuters", reuters_payload),
        ("Adobe", adobe_payload),
        ("Mozilla", mozilla_payload),
        ("Spotify", spotify_payload),
        ("CookieLawInfo", cookielawinfo_payload),
    ]:
        d = stub_pipeline.run(fn())
        a = phase4_pipeline.run(fn())
        improved = (
            a.confidence >= d.confidence
            and (a.ai_analysis or {}).get("nlp", {}).get("status") == "ready"
            and (a.ai_analysis or {}).get("vision", {}).get("status") == "ready"
        )
        rows.append(
            {
                "site": name,
                "rules_risk": d.risk_score,
                "ai_risk": a.risk_score,
                "delta": round(a.risk_score - d.risk_score, 2),
                "ai_channels_ready": improved,
            }
        )
    assert len(rows) == 7
    # Spotify: rules empty → AI must not invent risk
    spotify = next(r for r in rows if r["site"] == "Spotify")
    assert spotify["ai_risk"] == 0
    # Sites with rule hits may get a small AI confidence/risk assist
    with_hits = [r for r in rows if r["rules_risk"] > 0]
    assert all(r["ai_channels_ready"] for r in with_hits)
