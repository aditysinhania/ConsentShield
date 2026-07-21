"""Phase 3.4 — LLM narrator tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from ai.datasets.fixtures.validation_sites import guardian_payload
from ai.inference.pipeline import InferencePipeline
from ai.narrator.context import build_narrator_context
from ai.narrator.narrator import LLMNarrator


def _report_dict():
    report = InferencePipeline().run(guardian_payload())
    return report.model_dump()


def test_narrator_context_only_rule_facts():
    data = _report_dict()
    ctx = build_narrator_context(data)
    assert "triggered_rules" in ctx
    assert "evidence" in ctx
    assert "risk_score" in ctx
    assert "confidence" in ctx
    assert "severity" in ctx
    assert "timeline" in ctx
    assert "accessibility" not in ctx
    assert "vision" not in ctx
    assert all(e.get("source") != "vision" for e in ctx.get("evidence", []))


def test_offline_mode_uses_deterministic_summary():
    narrator = LLMNarrator(provider="offline")
    result = narrator.narrate(_report_dict(), url="https://www.theguardian.com/")
    assert result.source == "deterministic"
    assert result.status == "offline"
    assert result.summary
    assert "rule engine" in result.summary.lower() or "ConsentShield" in result.summary


def test_provider_unavailable_no_key_fallback():
    narrator = LLMNarrator(provider="gemini", gemini_api_key="")
    result = narrator.narrate(_report_dict(), url="https://example.com")
    assert result.source == "deterministic"
    assert result.status == "fallback"
    assert result.message


def test_invalid_key_fallback():
    narrator = LLMNarrator(provider="openai", openai_api_key="bad-key")
    mock_resp = MagicMock()
    mock_resp.raise_for_status.side_effect = httpx.HTTPStatusError(
        "Unauthorized",
        request=MagicMock(),
        response=MagicMock(status_code=401),
    )
    with patch("ai.narrator.providers.openai.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        result = narrator.narrate(_report_dict(), url="https://example.com")
    assert result.source == "deterministic"
    assert result.status == "fallback"
    assert result.message


def test_provider_configured_openai():
    narrator = LLMNarrator(provider="openai", openai_api_key="sk-test")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "LLM executive summary from triggered rules only."}}]
    }
    with patch("ai.narrator.providers.openai.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        result = narrator.narrate(_report_dict(), url="https://example.com")
    assert result.source == "llm:openai"
    assert result.status == "ready"
    assert "LLM executive summary" in result.summary
    assert result.duration_ms >= 0


def test_provider_configured_gemini():
    narrator = LLMNarrator(provider="gemini", gemini_api_key="gem-key")
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "candidates": [{"content": {"parts": [{"text": "Gemini summary of rule findings."}]}}]
    }
    with patch("ai.narrator.providers.gemini.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        result = narrator.narrate(_report_dict(), url="https://example.com")
    assert result.source == "llm:gemini"
    assert result.status == "ready"
    assert "Gemini summary" in result.summary


def test_narrator_wired_into_report_document():
    from ai.report.document import build_report_document

    data = _report_dict()
    narrator = LLMNarrator(provider="offline").narrate(data, url=guardian_payload().url)
    doc = build_report_document(
        scan_id="s1",
        url=guardian_payload().url,
        title="Guardian",
        risk_score=data["risk_score"],
        category=data["category"],
        confidence=data["confidence"],
        report=data,
        narrator=narrator.model_dump(),
    )
    assert doc.executive_summary == narrator.summary
    assert doc.narrator is not None
