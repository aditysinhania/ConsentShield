"""Export report documents to JSON, HTML, Markdown, and PDF."""

from __future__ import annotations

import html
import json
from io import BytesIO
from pathlib import Path
from typing import Any

from ai.report.document import ReportDocument
from ai.report.formatting import AUDIT_CSS, audit_header_html, evidence_detail_html

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
except ImportError:  # pragma: no cover
    SimpleDocTemplate = None  # type: ignore[misc, assignment]


def export_json(document: ReportDocument, *, indent: int = 2) -> str:
    return json.dumps(document.model_dump(), indent=indent, default=str)


def export_markdown(document: ReportDocument) -> str:
    lines: list[str] = [
        "# ConsentShield Audit Report",
        "",
        f"**URL:** {document.url}  ",
        f"**Scan ID:** `{document.scan_id}`  ",
        f"**Generated:** {document.generated_at}  ",
        "",
        "## Executive Summary",
        "",
        document.executive_summary,
        "",
    ]
    if document.narrator:
        lines.extend(
            [
                f"_Narrator source: `{document.narrator.get('source')}` · "
                f"status: `{document.narrator.get('status')}`_",
                "",
            ]
        )
    lines.extend(
        [
        "## Findings",
        "",
    ]
    )
    if document.findings:
        for f in document.findings:
            if f.get("type") == "cluster":
                lines.append(
                    f"- **{f.get('cluster')}** — {f.get('count')} item(s), "
                    f"rules: {', '.join(f.get('rule_ids') or [])}"
                )
            else:
                lines.append(f"- `{f.get('rule_id')}` — {f.get('statement')}")
    else:
        lines.append("_No findings._")
    lines.extend(["", "## Evidence", ""])
    for ev in document.evidence:
        lines.append(f"### {ev.get('rule_id') or ev.get('id')}")
        lines.append(f"- **Statement:** {ev.get('statement')}")
        if ev.get("explanation"):
            lines.append(f"- **Explanation:** {ev.get('explanation')}")
        if ev.get("recommendation"):
            lines.append(f"- **Recommendation:** {ev.get('recommendation')}")
        lines.append("")
    lines.extend(["## Screenshots", ""])
    lines.append(f"- Raw: `{document.screenshots.get('raw') or 'n/a'}`")
    lines.append(f"- Annotated: `{document.screenshots.get('annotated') or 'n/a'}`")
    lines.extend(["", "## Accessibility", ""])
    a11y = document.accessibility or {}
    lines.append(f"Issues: **{a11y.get('issue_count', 0)}**")
    for issue in a11y.get("issues") or []:
        lines.append(f"- `{issue.get('type')}` — {issue.get('description')}")
    lines.extend(["", "## Timeline", ""])
    for evt in document.timeline:
        lines.append(
            f"- `{evt.get('event')}` @ {evt.get('timestamp')} "
            f"({evt.get('duration_ms', '—')} ms)"
        )
    lines.extend(["", "## Performance", ""])
    perf = document.performance or {}
    for key in (
        "collection_ms",
        "rule_engine_ms",
        "fusion_ms",
        "screenshot_ms",
        "annotation_ms",
        "report_ms",
        "llm_ms",
        "total_ms",
    ):
        if key in perf:
            lines.append(f"- {key.replace('_', ' ').title()}: **{perf[key]} ms**")
    lines.extend(["", "## Recommendations", ""])
    for rec in document.recommendations:
        lines.append(f"- {rec}")
    lines.extend(["", "## Appendix", ""])
    lines.append(f"- Rule traces: {len(document.appendix.get('rule_traces') or [])}")
    lines.append(f"- Pipeline notes: {len(document.appendix.get('pipeline_notes') or [])}")
    return "\n".join(lines) + "\n"


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def export_html(document: ReportDocument) -> str:
    findings_html = ""
    for f in document.findings:
        if f.get("type") == "cluster":
            findings_html += (
                f"<li><strong>{_esc(f.get('cluster'))}</strong> — "
                f"{_esc(f.get('count'))} item(s)</li>"
            )
        else:
            findings_html += f"<li><code>{_esc(f.get('rule_id'))}</code> — {_esc(f.get('statement'))}</li>"

    evidence_html = "".join(evidence_detail_html(ev) for ev in document.evidence)

    a11y = document.accessibility or {}
    a11y_html = "".join(
        f"<tr><td><code>{_esc(i.get('type'))}</code></td>"
        f"<td>{_esc(i.get('description'))}</td>"
        f"<td>{_esc(i.get('element_label') or '—')}</td></tr>"
        for i in (a11y.get("issues") or [])
    )

    timeline_html = "".join(
        f"<tr><td>{_esc(e.get('event'))}</td><td>{_esc(e.get('timestamp'))}</td>"
        f"<td>{_esc(e.get('duration_ms'))}</td></tr>"
        for e in document.timeline
    )

    perf = document.performance or {}
    perf_rows = "".join(
        f"<tr><td>{_esc(k.replace('_', ' '))}</td><td>{_esc(perf.get(k))} ms</td>"
        f"<td>{_esc((perf.get('percent_of_total') or {}).get(k.replace('_ms', ''), '—'))}</td></tr>"
        for k in (
            "collection_ms",
            "rule_engine_ms",
            "fusion_ms",
            "screenshot_ms",
            "annotation_ms",
            "report_ms",
            "llm_ms",
            "total_ms",
        )
        if k in perf
    )

    rec_html = "".join(f"<li>{_esc(r)}</li>" for r in document.recommendations)
    narrator_note = ""
    if document.narrator:
        narrator_note = (
            f'<p class="meta">Narrator: {_esc(document.narrator.get("source"))} · '
            f'{_esc(document.narrator.get("status"))}</p>'
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>ConsentShield Audit — {_esc(document.url)}</title>
<style>{AUDIT_CSS}</style>
</head>
<body>
{audit_header_html(document)}
<section id="executive-summary"><h2><span class="section-num">1.</span>Executive Summary</h2>
<p>{_esc(document.executive_summary)}</p>{narrator_note}</section>
<section id="findings"><h2><span class="section-num">2.</span>Findings</h2>
<ul>{findings_html or '<li>No findings.</li>'}</ul></section>
<section id="evidence"><h2><span class="section-num">3.</span>Evidence</h2>
{evidence_html or '<p>No evidence items.</p>'}</section>
<section id="screenshots"><h2><span class="section-num">4.</span>Screenshots</h2>
<ul>
<li>Raw: <code>{_esc(document.screenshots.get('raw'))}</code></li>
<li>Annotated: <code>{_esc(document.screenshots.get('annotated'))}</code></li>
</ul></section>
<section id="accessibility"><h2><span class="section-num">5.</span>Accessibility</h2>
<p>Issues: <strong>{_esc(a11y.get('issue_count', 0))}</strong></p>
<table class="data"><thead><tr><th>Type</th><th>Description</th><th>Element</th></tr></thead>
<tbody>{a11y_html or '<tr><td colspan="3">None detected.</td></tr>'}</tbody></table></section>
<section id="timeline"><h2><span class="section-num">6.</span>Timeline</h2>
<table class="data"><thead><tr><th>Event</th><th>Timestamp</th><th>Duration (ms)</th></tr></thead>
<tbody>{timeline_html or '<tr><td colspan="3">No events.</td></tr>'}</tbody></table></section>
<section id="performance"><h2><span class="section-num">7.</span>Performance</h2>
<table class="data"><thead><tr><th>Stage</th><th>Duration</th><th>% Total</th></tr></thead>
<tbody>{perf_rows or '<tr><td colspan="3">No metrics.</td></tr>'}</tbody></table></section>
<section id="recommendations"><h2><span class="section-num">8.</span>Recommendations</h2>
<ul>{rec_html or '<li>None.</li>'}</ul></section>
<section id="appendix" class="appendix"><h2><span class="section-num">9.</span>Appendix</h2>
<p>Rule traces: {len(document.appendix.get('rule_traces') or [])} · Pipeline notes: {len(document.appendix.get('pipeline_notes') or [])}</p>
<pre>{_esc(json.dumps(document.appendix, indent=2, default=str)[:8000])}</pre></section>
<footer class="audit-footer">ConsentShield deterministic consent audit · Report format v1.1 · Rules are the source of truth.</footer>
</body>
</html>"""


def export_pdf(document: ReportDocument) -> bytes:
    if SimpleDocTemplate is None:
        raise RuntimeError("reportlab is required for PDF export")

    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, rightMargin=2 * cm, leftMargin=2 * cm, topMargin=2 * cm, bottomMargin=2 * cm)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("Title", parent=styles["Heading1"], textColor=colors.HexColor("#2d6a4f"))
    h2 = styles["Heading2"]
    body = styles["BodyText"]
    story: list[Any] = []

    story.append(Paragraph("ConsentShield Audit Report", title_style))
    story.append(Paragraph(f"URL: {document.url}", body))
    story.append(Paragraph(f"Scan ID: {document.scan_id}", body))
    story.append(Paragraph(f"Generated: {document.generated_at}", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Executive Summary", h2))
    story.append(Paragraph(document.executive_summary, body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Findings", h2))
    for f in document.findings[:20]:
        label = f.get("cluster") or f.get("rule_id") or "finding"
        story.append(Paragraph(f"• {label}: {f.get('statement') or f.get('count')}", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Recommendations", h2))
    for rec in document.recommendations[:15]:
        story.append(Paragraph(f"• {rec}", body))
    story.append(Spacer(1, 12))

    story.append(Paragraph("Performance (ms)", h2))
    perf = document.performance or {}
    perf_data = [[k, str(perf.get(k, ""))] for k in ("rule_engine_ms", "fusion_ms", "total_ms") if k in perf]
    if perf_data:
        t = Table([["Stage", "ms"]] + perf_data)
        t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey), ("GRID", (0, 0), (-1, -1), 0.5, colors.grey)]))
        story.append(t)

    doc.build(story)
    return buf.getvalue()


def write_exports(document: ReportDocument, output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    sid = document.scan_id
    paths: dict[str, str] = {}

    json_path = out / f"{sid}.json"
    json_path.write_text(export_json(document), encoding="utf-8")
    paths["json"] = str(json_path)

    html_path = out / f"{sid}.html"
    html_path.write_text(export_html(document), encoding="utf-8")
    paths["html"] = str(html_path)

    md_path = out / f"{sid}.md"
    md_path.write_text(export_markdown(document), encoding="utf-8")
    paths["markdown"] = str(md_path)

    pdf_path = out / f"{sid}.pdf"
    pdf_path.write_bytes(export_pdf(document))
    paths["pdf"] = str(pdf_path)

    return paths
