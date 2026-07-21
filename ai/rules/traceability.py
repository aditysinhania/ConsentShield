"""Build per-rule execution traces for API exposure."""

from __future__ import annotations

from typing import Any

from ai.common.types import ConfidenceBreakdown, RuleHit, RuleResult, RuleTrace

# Feature signals typically used by each evaluator
_RULE_FEATURES: dict[str, list[str]] = {
    "cookie.hidden_reject": [
        "buttons.text",
        "buttons.ariaLabel",
        "buttons.display",
        "buttons.visibility",
        "buttons.opacity",
        "buttons.fontSizePx",
        "visible_text",
    ],
    "cookie.multi_click_reject": [
        "buttons.text",
        "buttons.ariaLabel",
        "visible_text",
        "settings_patterns",
    ],
    "cookie.accept_visually_dominant": [
        "buttons.width",
        "buttons.height",
        "buttons.fontWeight",
        "accept_reject_area_ratio",
    ],
    "cookie.unequal_emphasis": [
        "buttons.backgroundColor",
        "buttons.textDecoration",
    ],
    "cookie.preselected_marketing": [
        "checkboxes.label",
        "checkboxes.checked",
        "checkboxes.required",
        "toggles.label",
        "toggles.checked",
    ],
    "cookie.banner_obstruction": [
        "banner.width",
        "banner.height",
        "viewport.width",
        "viewport.height",
        "coverage_ratio",
    ],
    "sub.auto_renew": ["visible_text", "billingSection.fontSizePx", "billingSection.nearPrimaryCta"],
    "sub.hidden_billing": ["billingSection.hidden", "billingSection.opacity", "layoutHints"],
    "sub.small_font_disclaimer": ["billingSection.fontSizePx", "body.fontSizePx"],
    "sub.misleading_free_trial": ["visible_text", "primaryCta.text"],
    "sub.preselected_plan": ["subscriptionOptions.checked", "subscriptionOptions.label"],
    "sub.confirmshaming": ["visible_text"],
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, round(value, 4)))


def _features_for_hit(hit: RuleHit) -> list[str]:
    if hit.features_used:
        return list(hit.features_used)
    base = list(_RULE_FEATURES.get(hit.rule_id, []))
    # Include metadata keys that were actually present
    for key in sorted(hit.metadata.keys()):
        feat = f"metadata.{key}"
        if feat not in base:
            base.append(feat)
    return base


def _component_scores(hit: RuleHit) -> tuple[float, float, float]:
    """Derive visual / text / layout component scores from hit data."""
    visual = hit.visual_score
    text = hit.text_score
    layout = hit.layout_score

    meta = hit.metadata
    # Infer from metadata when evaluators did not set component scores
    if visual == 0.0 and any(
        k in meta for k in ("area_ratio", "accept_weight", "reject_weight", "accept_bg", "reject_bg", "opacity", "fontSizePx")
    ):
        visual = _clamp01(hit.score)
    if text == 0.0 and any(k in meta for k in ("match", "accept_text", "label", "cta", "settings_path")):
        text = _clamp01(hit.score)
    if layout == 0.0 and any(k in meta for k in ("coverage", "display", "visibility")):
        layout = _clamp01(hit.score)

    # If still unset, distribute from overall score using feature types
    features = _features_for_hit(hit)
    if visual == 0.0 and any("button" in f or "font" in f or "weight" in f or "bg" in f for f in features):
        visual = _clamp01(hit.score * 0.85)
    if text == 0.0 and any("text" in f or "label" in f or "match" in f or "pattern" in f for f in features):
        text = _clamp01(hit.score * 0.85)
    if layout == 0.0 and any("banner" in f or "viewport" in f or "coverage" in f or "layout" in f for f in features):
        layout = _clamp01(hit.score * 0.85)

    # Guarantee at least one non-zero component when rule fired
    if visual == 0.0 and text == 0.0 and layout == 0.0:
        text = _clamp01(hit.score)

    return visual, text, layout


def build_rule_trace(hit: RuleHit, *, total_weighted: float, normalized_risk: float) -> RuleTrace:
    visual, text, layout = _component_scores(hit)
    weighted = hit.score * max(hit.severity, 0.1)
    if total_weighted > 0 and normalized_risk > 0:
        risk_contribution = round(normalized_risk * (weighted / total_weighted), 2)
    else:
        risk_contribution = 0.0

    # Per-rule confidence: severity-weighted score with component agreement
    components = [c for c in (visual, text, layout) if c > 0]
    agreement = _clamp01(1.0 - (max(components) - min(components)) if len(components) > 1 else 0.7)
    cmp_signal = 0.0  # CMP certainty filled at fusion level; per-rule default 0
    final = _clamp01(0.4 * hit.score + 0.3 * hit.severity + 0.2 * agreement + 0.1 * max(components or [0]))

    return RuleTrace(
        rule_id=hit.rule_id,
        features_used=_features_for_hit(hit),
        visual_score=visual,
        text_score=text,
        layout_score=layout,
        confidence_breakdown=ConfidenceBreakdown(
            text=text,
            visual=visual,
            layout=layout,
            cmp=cmp_signal,
            agreement=agreement,
            final=final,
        ),
        risk_contribution=risk_contribution,
    )


def attach_traces(result: RuleResult) -> RuleResult:
    """Populate result.traces and sync component scores onto hits."""
    if not result.hits:
        result.traces = []
        return result

    total_weighted = sum(h.score * max(h.severity, 0.1) for h in result.hits)
    traces: list[RuleTrace] = []
    updated_hits: list[RuleHit] = []
    for hit in result.hits:
        visual, text, layout = _component_scores(hit)
        updated = hit.model_copy(
            update={
                "features_used": _features_for_hit(hit),
                "visual_score": visual,
                "text_score": text,
                "layout_score": layout,
            }
        )
        updated_hits.append(updated)
        traces.append(
            build_rule_trace(
                updated,
                total_weighted=total_weighted,
                normalized_risk=result.normalized_risk,
            )
        )
    result.hits = updated_hits
    result.traces = traces
    return result


def traces_as_dicts(result: RuleResult) -> list[dict[str, Any]]:
    return [t.model_dump() for t in result.traces]
