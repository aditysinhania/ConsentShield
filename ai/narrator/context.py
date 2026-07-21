"""Strict narrator input — only rule-sourced audit facts."""

from __future__ import annotations

from typing import Any


def build_narrator_context(report: dict[str, Any]) -> dict[str, Any]:
    """
    Build the ONLY payload sent to an LLM.
    Excludes vision/text/accessibility to prevent invented findings.
    """
    rules = report.get("rules") or {}
    hits = rules.get("hits") or []
    rule_evidence = [
        {
            "rule_id": e.get("rule_id"),
            "statement": e.get("statement"),
            "severity": e.get("severity"),
            "explanation": e.get("explanation"),
            "recommendation": e.get("recommendation"),
        }
        for e in (report.get("evidence") or [])
        if e.get("source") == "rules"
    ]
    return {
        "triggered_rules": [
            {
                "rule_id": h.get("rule_id"),
                "name": h.get("name"),
                "category": h.get("category"),
                "severity": h.get("severity"),
                "score": h.get("score"),
                "evidence": h.get("evidence"),
            }
            for h in hits
        ],
        "evidence": rule_evidence,
        "risk_score": report.get("risk_score"),
        "confidence": report.get("confidence"),
        "severity": report.get("severity"),
        "timeline": report.get("timeline") or [],
    }


NARRATOR_SYSTEM_PROMPT = """You are ConsentShield's audit narrator. Summarize ONLY the JSON facts provided.
Rules:
- Do NOT invent, infer, or add findings not present in the input.
- Do NOT mention patterns, issues, or recommendations unless they appear in triggered_rules or evidence.
- Refer to risk_score, confidence, severity, and timeline only as given.
- Write 2-4 concise paragraphs suitable as an executive summary for compliance stakeholders.
- If no rules triggered, state that no dark patterns were detected by the rule engine.
"""
