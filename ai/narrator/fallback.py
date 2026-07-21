"""Deterministic fallback executive summary."""

from __future__ import annotations

from typing import Any


def deterministic_summary(context: dict[str, Any], *, url: str = "") -> str:
    """Produce executive summary from narrator context without LLM."""
    triggered = context.get("triggered_rules") or []
    severity = context.get("severity") or {}
    level = severity.get("level", "—")
    justification = severity.get("justification", "")
    risk = context.get("risk_score", 0)
    confidence = context.get("confidence", 0)
    timeline = context.get("timeline") or []

    if not triggered:
        return (
            f"ConsentShield audited {url or 'the page'} and the deterministic rule engine "
            f"did not trigger any consent dark-pattern rules. Risk score {risk:.0f}/100, "
            f"confidence {confidence:.0%}, severity {level}. "
            f"This summary is generated without LLM inference and adds no new findings."
        )

    rule_names = ", ".join(h.get("rule_id", "?") for h in triggered[:6])
    extra = f" …and {len(triggered) - 6} more" if len(triggered) > 6 else ""
    events = len(timeline)

    parts = [
        f"ConsentShield audited {url or 'the page'} using the deterministic rule engine.",
        f"Risk score {risk:.0f}/100 with confidence {confidence:.0%} and severity tier {level}.",
        f"{len(triggered)} rule(s) triggered: {rule_names}{extra}.",
    ]
    if justification:
        parts.append(justification)
    if events:
        parts.append(f"Scan pipeline recorded {events} timeline event(s).")
    parts.append(
        "Recommendations and evidence statements below are sourced exclusively from triggered rules; "
        "this summary does not add findings beyond the rule engine output."
    )
    return " ".join(parts)
