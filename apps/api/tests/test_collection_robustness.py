"""Collection robustness + consent vocabulary tests (pre-Phase 4)."""

from __future__ import annotations

import pytest

from ai.common.types import ScanPayload
from ai.datasets.fixtures.validation_sites import (
    adobe_payload,
    bbc_payload,
    cookielawinfo_payload,
    guardian_payload,
    mozilla_payload,
    reuters_payload,
    spotify_payload,
)
from ai.debug.zero_risk import build_zero_risk_debug
from ai.inference.pipeline import InferencePipeline
from ai.rules import RuleEngine
from ai.rules.evaluators.registry import ACCEPT_PATTERNS, REJECT_PATTERNS, SETTINGS_PATTERNS


@pytest.fixture(scope="module")
def pipeline() -> InferencePipeline:
    return InferencePipeline()


@pytest.mark.parametrize(
    "label,kind",
    [
        ("Yes, I accept", "accept"),
        ("Accept cookies", "accept"),
        ("Accept all", "accept"),
        ("Allow all", "accept"),
        ("I agree", "accept"),
        ("Agree", "accept"),
        ("No, thank you", "reject"),
        ("No thanks", "reject"),
        ("Reject all", "reject"),
        ("Decline", "reject"),
        ("Disagree", "reject"),
        ("Only necessary", "reject"),
        ("Essential only", "reject"),
        ("Manage cookies", "settings"),
        ("Privacy settings", "settings"),
        ("Cookie settings", "settings"),
        ("Customize", "settings"),
        ("Save preferences", "settings"),
    ],
)
def test_consent_vocabulary_labels(label: str, kind: str):
    if kind == "accept":
        assert ACCEPT_PATTERNS.search(label)
    elif kind == "reject":
        assert REJECT_PATTERNS.search(label)
    else:
        assert SETTINGS_PATTERNS.search(label)


def test_guardian_live_labels_trigger_rules():
    result = RuleEngine().evaluate_payload(guardian_payload())
    ids = {h.rule_id for h in result.hits}
    assert result.normalized_risk >= 30
    assert "cookie.accept_visually_dominant" in ids or "cookie.unequal_emphasis" in ids
    assert "cookie.banner_obstruction" in ids


def test_bbc_fixture_triggers_rules():
    result = RuleEngine().evaluate_payload(bbc_payload())
    assert result.normalized_risk >= 20
    assert result.hits


def test_zero_risk_debug_when_no_banner():
    payload = ScanPayload(
        url="https://example.com/",
        visible_text="Hello world",
        viewport={"width": 1280, "height": 800},
        css_snapshot={"buttons": [{"text": "Sign in", "width": 80, "height": 32}]},
    )
    report = InferencePipeline().run(payload)
    assert report.risk_score == 0
    assert report.debug is not None
    assert report.debug["rules_matched"] == 0
    assert report.debug["rules_evaluated"] > 0
    assert "No consent banner detected" in report.debug["reasons"] or report.debug["summary"]


def test_zero_risk_debug_unmatched_labels():
    payload = ScanPayload(
        url="https://example.com/",
        visible_text="cookies",
        viewport={"width": 1280, "height": 800},
        css_snapshot={
            "banner": {"width": 800, "height": 200, "xpath": "/div"},
            "buttons": [{"text": "Proceed", "width": 100, "height": 40}],
            "consent_state": {"banner_visible": True},
            "cmp": {"detected": True, "vendor": "Sourcepoint", "confidence": 0.9},
        },
    )
    report = InferencePipeline().run(payload)
    assert report.risk_score == 0
    assert report.debug is not None
    assert any("unmatched" in r.lower() or "labels" in r.lower() for r in report.debug["reasons"])


def test_zero_risk_debug_inaccessible_iframe():
    payload = ScanPayload(
        url="https://example.com/",
        viewport={"width": 1280, "height": 800},
        css_snapshot={
            "buttons": [],
            "iframes": [
                {
                    "src": "https://cdn.privacy-mgmt.com/message",
                    "cross_origin": True,
                    "likely_cmp": True,
                    "cmp_vendor": "Sourcepoint",
                    "accessible": False,
                }
            ],
            "cmp": {"detected": True, "vendor": "Sourcepoint", "confidence": 0.85},
        },
    )
    debug = build_zero_risk_debug(
        payload,
        RuleEngine().evaluate_payload(payload),
        InferencePipeline().run(payload),
        rules_catalog_count=12,
    )
    assert debug is not None
    assert any("iframe" in r.lower() for r in debug["reasons"])


@pytest.mark.parametrize(
    "name,fn,lo,hi",
    [
        ("Guardian", guardian_payload, 30, 55),
        ("BBC", bbc_payload, 20, 55),
        ("Adobe", adobe_payload, 28, 55),
        ("Spotify", spotify_payload, 0, 25),
        ("Mozilla", mozilla_payload, 12, 40),
        ("Reuters", reuters_payload, 20, 55),
        ("CookieLawInfo", cookielawinfo_payload, 20, 50),
    ],
)
def test_site_risk_bands_after_collection_hardening(name, fn, lo, hi, pipeline):
    report = pipeline.run(fn())
    assert lo <= report.risk_score <= hi, f"{name} risk {report.risk_score} outside [{lo},{hi}]"
    css = fn().css_snapshot or {}
    assert css.get("cmp", {}).get("detected") is True
    if report.risk_score == 0:
        assert report.debug is not None
