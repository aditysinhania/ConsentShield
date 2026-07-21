"""Unit tests for Phase 3 explanation, enrichment, traces, and confidence."""

from __future__ import annotations

from ai.common.types import ScanPayload
from ai.evidence.enrichment import enrich_evidence_item
from ai.explanation.catalog import explanation_for_rule
from ai.fusion.aggregator.fusion_engine import FusionEngine
from ai.common.types import FusionInput, EvidenceItem
from ai.inference.pipeline import InferencePipeline
from ai.rules import RuleEngine
from ai.rules.evidence.builder import evidence_from_rules


def _accept_only_payload(**extra) -> ScanPayload:
    css = {
        "buttons": [
            {
                "text": "Accept all",
                "width": 160,
                "height": 40,
                "fontWeight": 700,
                "backgroundColor": "rgb(0,100,0)",
                "fontSizePx": 16,
                "xpath": "/html[1]/body[1]/button[1]",
                "cssSelector": "button.accept-all",
                "domPath": "html > body > button.accept-all",
                "htmlSnippet": '<button class="accept-all">Accept all</button>',
                "boundingRect": {
                    "x": 10,
                    "y": 20,
                    "width": 160,
                    "height": 40,
                    "top": 20,
                    "left": 10,
                    "right": 170,
                    "bottom": 60,
                },
                "computedStyles": {
                    "fontSizePx": 16,
                    "fontWeight": 700,
                    "backgroundColor": "rgb(0,100,0)",
                    "color": "rgb(255,255,255)",
                    "display": "block",
                    "visibility": "visible",
                    "opacity": 1,
                },
            }
        ],
        "banner": {
            "width": 800,
            "height": 400,
            "xpath": "/html[1]/body[1]/div[1]",
            "cssSelector": "#cookie-banner",
            "domPath": "html > body > div#cookie-banner",
        },
    }
    return ScanPayload(
        url="https://example.com/consent",
        title="Example Consent Page",
        visible_text="We use cookies. Accept all",
        css_snapshot=css,
        viewport={"width": 1200, "height": 800},
        scroll_position={"x": 0, "y": 120},
        collected_at="2026-07-21T10:00:00Z",
        **extra,
    )


def test_hidden_reject_triggers():
    engine = RuleEngine()
    result = engine.evaluate_payload(_accept_only_payload())
    ids = {h.rule_id for h in result.hits}
    assert "cookie.hidden_reject" in ids
    assert result.normalized_risk > 0


def test_structured_explanation_on_evidence():
    engine = RuleEngine()
    result = engine.evaluate_payload(_accept_only_payload())
    items = evidence_from_rules(result, _accept_only_payload())
    hidden = next(i for i in items if i.rule_id == "cookie.hidden_reject")
    assert hidden.explanation
    assert hidden.user_impact
    assert "GDPR" in (hidden.gdpr_relevance or "")
    assert hidden.recommendation
    catalog = explanation_for_rule("cookie.hidden_reject")
    assert catalog.explanation == hidden.explanation


def test_evidence_enrichment_uses_real_values_only():
    payload = _accept_only_payload()
    engine = RuleEngine()
    result = engine.evaluate_payload(payload)
    items = evidence_from_rules(result, payload)
    hidden = next(i for i in items if i.rule_id == "cookie.hidden_reject")
    assert hidden.url == "https://example.com/consent"
    assert hidden.page_title == "Example Consent Page"
    assert hidden.timestamp == "2026-07-21T10:00:00Z"
    assert hidden.viewport == {"width": 1200, "height": 800}
    assert hidden.scroll_position == {"x": 0, "y": 120}
    assert hidden.xpath == "/html[1]/body[1]/button[1]"
    assert hidden.css_selector == "button.accept-all"
    assert hidden.dom_path == "html > body > button.accept-all"
    assert hidden.html_snippet and "Accept all" in hidden.html_snippet
    assert hidden.bounding_rect is not None
    assert hidden.bounding_rect.width == 160
    assert hidden.computed_styles is not None
    assert hidden.computed_styles["fontWeight"] == 700


def test_enrichment_skips_missing_locators():
    item = EvidenceItem(
        id="rule:test",
        statement="test",
        severity=0.5,
        source="rules",
        rule_id="cookie.hidden_reject",
        metadata={},
    )
    payload = ScanPayload(url="https://x.test", title="T")
    enriched = enrich_evidence_item(item, payload)
    assert enriched.url == "https://x.test"
    assert enriched.page_title == "T"
    assert enriched.xpath is None
    assert enriched.css_selector is None
    assert enriched.bounding_rect is None
    assert enriched.scroll_position is None


def test_rule_traceability():
    engine = RuleEngine()
    result = engine.evaluate_payload(_accept_only_payload())
    assert result.traces
    trace = next(t for t in result.traces if t.rule_id == "cookie.hidden_reject")
    assert trace.features_used
    assert trace.risk_contribution > 0
    assert 0 <= trace.visual_score <= 1
    assert 0 <= trace.text_score <= 1
    assert 0 <= trace.layout_score <= 1
    assert trace.confidence_breakdown.final > 0
    hit = next(h for h in result.hits if h.rule_id == "cookie.hidden_reject")
    assert hit.features_used
    assert hit.text_score > 0 or hit.visual_score > 0


def test_confidence_breakdown_keeps_final():
    payload = _accept_only_payload()
    engine = RuleEngine()
    rules = engine.evaluate_payload(payload)
    fusion = FusionEngine().fuse(
        FusionInput(rules=rules, dom_features=payload.css_snapshot or {})
    )
    assert fusion.confidence > 0
    assert fusion.confidence_breakdown is not None
    assert fusion.confidence_breakdown.final == fusion.confidence
    assert 0 <= fusion.confidence_breakdown.text <= 1
    assert 0 <= fusion.confidence_breakdown.visual <= 1
    assert 0 <= fusion.confidence_breakdown.layout <= 1
    assert 0 <= fusion.confidence_breakdown.cmp <= 1
    assert 0 <= fusion.confidence_breakdown.agreement <= 1
    assert fusion.confidence_breakdown.cmp > 0


def test_pipeline_wires_phase3_fields():
    report = InferencePipeline().run(_accept_only_payload())
    assert report.confidence_breakdown is not None
    assert report.confidence_breakdown.final == report.confidence
    assert report.rule_traces
    assert any(e.explanation for e in report.evidence)
    assert any(e.url for e in report.evidence)
    assert report.severity is not None
    assert report.accessibility is not None
    assert report.timeline is not None
    assert report.performance is not None
    dumped = report.model_dump()
    assert "confidence_breakdown" in dumped
    assert "rule_traces" in dumped
    assert "severity" in dumped
    assert "pattern_clusters" in dumped
    assert "accessibility" in dumped
    assert "timeline" in dumped
    assert "performance" in dumped
    assert dumped["evidence"][0].get("explanation")
