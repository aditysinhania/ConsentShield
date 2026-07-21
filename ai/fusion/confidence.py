"""Deterministic fusion confidence breakdown (keeps existing confidence field)."""

from __future__ import annotations

from typing import Any

from ai.common.types import ConfidenceBreakdown, FusionInput, RuleResult


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _cmp_certainty(dom: dict[str, Any]) -> float:
    """Estimate CMP presence certainty from collected DOM/CSS hints."""
    score = 0.0
    banner = dom.get("banner") or {}
    if isinstance(banner, dict) and (banner.get("width") or banner.get("height") or banner.get("xpath")):
        score += 0.45
    cmp_meta = dom.get("cmp") or dom.get("cmpMetadata") or {}
    if isinstance(cmp_meta, dict) and cmp_meta:
        if cmp_meta.get("name") or cmp_meta.get("id") or cmp_meta.get("detected"):
            score += 0.4
        if cmp_meta.get("version"):
            score += 0.1
    buttons = list(dom.get("buttons", []) or [])
    cookie_like = sum(
        1
        for b in buttons
        if any(
            tok in str(b.get("text", "") or b.get("ariaLabel", "")).lower()
            for tok in ("accept", "reject", "cookie", "consent", "agree")
        )
    )
    if cookie_like >= 1:
        score += 0.15
    return _clamp01(score)


def compute_confidence_breakdown(
    inputs: FusionInput,
    *,
    final_confidence: float,
) -> ConfidenceBreakdown:
    """
    Build text/visual/layout/cmp/agreement/final breakdown.
    `final` is set to the existing fusion confidence (not a second invent).
    """
    rules: RuleResult | None = inputs.rules
    hits = rules.hits if rules else []

    if hits:
        text = _clamp01(sum(h.text_score for h in hits) / len(hits))
        visual = _clamp01(sum(h.visual_score for h in hits) / len(hits))
        layout = _clamp01(sum(h.layout_score for h in hits) / len(hits))
    else:
        text = visual = layout = 0.0

    # Vision stub contributes only when ready
    if inputs.vision and inputs.vision.status == "ready" and inputs.vision.banner_detected:
        visual = _clamp01(max(visual, 0.6))

    if inputs.text and inputs.text.status == "ready" and inputs.text.confidence is not None:
        text = _clamp01(max(text, float(inputs.text.confidence)))

    cmp = _cmp_certainty(inputs.dom_features or {})

    active = [c for c in (text, visual, layout, cmp) if c > 0]
    if len(active) >= 2:
        agreement = _clamp01(1.0 - (max(active) - min(active)))
    elif len(active) == 1:
        agreement = 0.65
    else:
        agreement = 0.5

    return ConfidenceBreakdown(
        text=text,
        visual=visual,
        layout=layout,
        cmp=cmp,
        agreement=agreement,
        final=_clamp01(final_confidence),
    )
