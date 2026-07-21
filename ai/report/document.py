"""Structured audit report document for multi-format export."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field


class ReportDocument(BaseModel):
    scan_id: str
    url: str
    title: str | None = None
    generated_at: str
    executive_summary: str
    narrator: dict[str, Any] | None = None
    findings: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    screenshots: dict[str, str | None] = Field(default_factory=dict)
    accessibility: dict[str, Any] | None = None
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    performance: dict[str, Any] | None = None
    recommendations: list[str] = Field(default_factory=list)
    appendix: dict[str, Any] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)


def _executive_summary(ctx: dict[str, Any]) -> str:
    report = ctx.get("report") or {}
    severity = report.get("severity") or {}
    level = severity.get("level", "—")
    clusters = report.get("pattern_clusters") or []
    a11y = report.get("accessibility") or {}
    a11y_count = a11y.get("issue_count", 0)
    hits = len((report.get("rules") or {}).get("hits") or [])
    if hits == 0:
        hits = len(report.get("evidence") or [])

    return (
        f"ConsentShield audited {ctx.get('url', 'unknown URL')}. "
        f"Risk score {ctx.get('risk_score', 0):.0f}/100 with category "
        f"\"{ctx.get('category', 'Unknown')}\" and severity {level}. "
        f"Detection confidence is {ctx.get('confidence', 0):.0%}. "
        f"{hits} rule finding(s) grouped into {len(clusters)} pattern cluster(s). "
        f"{a11y_count} accessibility issue(s) recorded separately from rule findings. "
        f"All statements are derived from the deterministic rule engine; this summary does not add new findings."
    )


def _recommendations(report: dict[str, Any]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in report.get("evidence") or []:
        rec = item.get("recommendation")
        if rec and rec not in seen:
            seen.add(rec)
            out.append(rec)
    if not out and report.get("category") == "No Dark Pattern":
        out.append("No remediation required based on current rule triggers.")
    return out


def _findings(report: dict[str, Any]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    for cluster in report.get("pattern_clusters") or []:
        findings.append(
            {
                "type": "cluster",
                "cluster": cluster.get("cluster"),
                "count": cluster.get("count", 0),
                "rule_ids": cluster.get("rule_ids", []),
                "max_severity": cluster.get("max_severity"),
            }
        )
    for item in report.get("evidence") or []:
        if item.get("source") == "rules":
            findings.append(
                {
                    "type": "rule_hit",
                    "rule_id": item.get("rule_id"),
                    "statement": item.get("statement"),
                    "severity": item.get("severity"),
                    "explanation": item.get("explanation"),
                }
            )
    return findings


def build_report_document(
    *,
    scan_id: str,
    url: str,
    title: str | None,
    risk_score: float,
    category: str,
    confidence: float,
    report: dict[str, Any],
    screenshot_path: str | None = None,
    annotated_screenshot_path: str | None = None,
    narrator: dict[str, Any] | None = None,
) -> ReportDocument:
    ctx = {
        "scan_id": scan_id,
        "url": url,
        "title": title,
        "risk_score": risk_score,
        "category": category,
        "confidence": confidence,
        "report": report,
    }
    narrator_data = narrator or report.get("narrator")
    exec_summary = (
        narrator_data.get("summary")
        if narrator_data and narrator_data.get("summary")
        else _executive_summary(ctx)
    )

    return ReportDocument(
        scan_id=scan_id,
        url=url,
        title=title,
        generated_at=datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        executive_summary=exec_summary,
        narrator=narrator_data,
        findings=_findings(report),
        evidence=list(report.get("evidence") or []),
        screenshots={
            "raw": screenshot_path,
            "annotated": annotated_screenshot_path or report.get("annotated_screenshot_path"),
        },
        accessibility=report.get("accessibility"),
        timeline=list(report.get("timeline") or []),
        performance=report.get("performance"),
        recommendations=_recommendations(report),
        appendix={
            "rule_traces": report.get("rule_traces") or [],
            "confidence_breakdown": report.get("confidence_breakdown"),
            "severity": report.get("severity"),
            "rules": report.get("rules"),
            "fusion": report.get("fusion"),
            "pipeline_notes": report.get("pipeline_notes") or [],
        },
        meta={
            "risk_score": risk_score,
            "category": category,
            "confidence": confidence,
        },
    )
