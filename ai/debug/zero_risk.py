"""Deterministic diagnostics when a scan yields risk score 0."""

from __future__ import annotations

from typing import Any

from ai.common.types import ExplainableReport, RuleResult, ScanPayload
from ai.rules.evaluators.registry import ACCEPT_PATTERNS, REJECT_PATTERNS, SETTINGS_PATTERNS


def _button_labels(css: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    for btn in css.get("buttons") or []:
        if not isinstance(btn, dict):
            continue
        text = str(btn.get("text") or btn.get("ariaLabel") or "").strip()
        if text:
            labels.append(text)
    return labels


def build_zero_risk_debug(
    payload: ScanPayload,
    rules: RuleResult | None,
    report: ExplainableReport,
    *,
    rules_catalog_count: int = 0,
) -> dict[str, Any] | None:
    """
    Return a structured debug block when risk is 0.
    Explains collection / matching gaps without changing scores.
    """
    if report.risk_score > 0:
        return None

    css = payload.css_snapshot or {}
    banner = css.get("banner") if isinstance(css.get("banner"), dict) else None
    cmp = css.get("cmp") if isinstance(css.get("cmp"), dict) else {}
    consent_state = css.get("consent_state") if isinstance(css.get("consent_state"), dict) else {}
    iframes = list(css.get("iframes") or [])
    buttons = list(css.get("buttons") or [])
    labels = _button_labels(css)

    accept_hits = [t for t in labels if ACCEPT_PATTERNS.search(t)]
    reject_hits = [t for t in labels if REJECT_PATTERNS.search(t)]
    settings_hits = [t for t in labels if SETTINGS_PATTERNS.search(t)]

    reasons: list[str] = []
    banner_visible = bool(
        consent_state.get("banner_visible")
        or (banner and (banner.get("width") or banner.get("height")))
    )
    banner_dismissed = bool(consent_state.get("banner_dismissed") or consent_state.get("banner_hidden"))

    if not banner_visible and not banner:
        reasons.append("No consent banner detected")
    elif banner_dismissed:
        reasons.append("Banner dismissed or hidden")
    elif banner_visible or banner:
        if not buttons:
            reasons.append("Banner detected but no buttons found")
        elif not accept_hits and not reject_hits:
            reasons.append("Buttons found but labels unmatched")
        else:
            vw = float((payload.viewport or {}).get("width") or 0)
            vh = float((payload.viewport or {}).get("height") or 0)
            bw = float((banner or {}).get("width") or 0)
            bh = float((banner or {}).get("height") or 0)
            if vw > 0 and vh > 0 and bw > 0 and bh > 0:
                coverage = (bw * bh) / (vw * vh)
                if coverage < 0.35 and not accept_hits:
                    reasons.append("Banner below obstruction threshold")
                elif accept_hits and reject_hits:
                    reasons.append(
                        "Accept and Reject both matched; visual/emphasis thresholds not met"
                    )
                elif accept_hits and not reject_hits and settings_hits:
                    reasons.append(
                        "Accept + settings found but multi-click / hidden-reject did not trigger"
                    )

    cross_origin_cmp = [
        f
        for f in iframes
        if isinstance(f, dict) and f.get("cross_origin") and (f.get("cmp_vendor") or f.get("likely_cmp"))
    ]
    if cross_origin_cmp and not accept_hits:
        reasons.append("CMP detected but inaccessible iframe")

    if cmp.get("detected") and not banner_visible and not accept_hits:
        reasons.append(
            f"CMP vendor '{cmp.get('vendor') or 'unknown'}' detected but consent UI not collected"
        )

    if not reasons:
        reasons.append("No dark-pattern rules matched on collected features")

    rules_evaluated = rules_catalog_count
    if not rules_evaluated and rules and rules.traces:
        rules_evaluated = len(rules.traces)

    return {
        "status": "zero_risk",
        "reasons": reasons,
        "banner_detected": bool(banner),
        "banner_visible": banner_visible,
        "cmp_detected": bool(cmp.get("detected")),
        "cmp_vendor": cmp.get("vendor"),
        "cmp_confidence": cmp.get("confidence"),
        "button_count": len(buttons),
        "consent_button_count": sum(
            1 for b in buttons if isinstance(b, dict) and b.get("inConsentContainer")
        ),
        "accept_labels_matched": accept_hits[:8],
        "reject_labels_matched": reject_hits[:8],
        "settings_labels_matched": settings_hits[:8],
        "sample_button_labels": labels[:12],
        "iframe_count": len(iframes),
        "cross_origin_cmp_iframes": len(cross_origin_cmp),
        "consent_state": consent_state or None,
        "rules_evaluated": rules_evaluated,
        "rules_matched": len(rules.hits) if rules else 0,
        "summary": "; ".join(reasons),
    }
