"""Professional audit report formatting helpers."""

from __future__ import annotations

import html
from typing import Any

from ai.report.document import ReportDocument


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else ""))


def _severity_badge(document: ReportDocument) -> str:
    sev = (document.appendix.get("severity") or {}) if document.appendix else {}
    level = sev.get("level", "—")
    colors = {
        "LOW": "#2d6a4f",
        "MEDIUM": "#bc6c25",
        "HIGH": "#d00000",
        "CRITICAL": "#370617",
    }
    color = colors.get(str(level), "#555")
    return f'<span class="badge" style="background:{color}">{level}</span>'


def audit_header_html(document: ReportDocument) -> str:
    meta = document.meta or {}
    return f"""
<div class="audit-header">
  <div class="audit-brand">ConsentShield · Consent Compliance Audit</div>
  <h1>Audit Report</h1>
  <table class="audit-meta">
    <tr><th>Target URL</th><td>{document.url}</td></tr>
    <tr><th>Page Title</th><td>{document.title or '—'}</td></tr>
    <tr><th>Scan ID</th><td><code>{document.scan_id}</code></td></tr>
    <tr><th>Generated</th><td>{document.generated_at}</td></tr>
    <tr><th>Category</th><td>{meta.get('category', '—')}</td></tr>
  </table>
</div>
<div class="risk-dashboard">
  <div class="metric"><div class="label">Risk Score</div><div class="value">{meta.get('risk_score', 0):.0f}</div></div>
  <div class="metric"><div class="label">Confidence</div><div class="value">{float(meta.get('confidence', 0)):.0%}</div></div>
  <div class="metric"><div class="label">Severity</div><div class="value">{_severity_badge(document)}</div></div>
  <div class="metric"><div class="label">Findings</div><div class="value">{len(document.findings)}</div></div>
</div>
"""


AUDIT_CSS = """
:root { --green:#1b4332; --accent:#2d6a4f; --border:#d8e2dc; --bg:#f8faf9; }
body { font-family: 'Segoe UI', system-ui, sans-serif; max-width: 980px; margin: 0 auto; padding: 2rem 1.5rem; color: #1a1a1a; background: #fff; line-height: 1.55; }
.audit-header { border-bottom: 3px solid var(--accent); padding-bottom: 1.25rem; margin-bottom: 1.5rem; }
.audit-brand { font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--accent); font-weight: 600; }
h1 { color: var(--green); margin: 0.4rem 0 1rem; font-size: 1.85rem; }
h2 { color: var(--green); font-size: 1.25rem; border-left: 4px solid var(--accent); padding-left: 0.65rem; margin-top: 2.25rem; }
.audit-meta { width: 100%; border-collapse: collapse; font-size: 0.92rem; }
.audit-meta th { text-align: left; width: 140px; color: #555; padding: 0.35rem 0.5rem 0.35rem 0; vertical-align: top; }
.audit-meta td { padding: 0.35rem 0; }
.risk-dashboard { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.75rem; margin-bottom: 2rem; }
.metric { background: var(--bg); border: 1px solid var(--border); border-radius: 10px; padding: 0.85rem 1rem; text-align: center; }
.metric .label { font-size: 0.75rem; text-transform: uppercase; color: #666; letter-spacing: 0.04em; }
.metric .value { font-size: 1.5rem; font-weight: 700; color: var(--green); margin-top: 0.25rem; }
.badge { display: inline-block; color: #fff; padding: 0.15rem 0.55rem; border-radius: 4px; font-size: 0.85rem; font-weight: 600; }
.evidence { border: 1px solid var(--border); border-radius: 10px; padding: 1rem 1.15rem; margin: 1rem 0; background: var(--bg); }
.evidence .quality { font-size: 0.8rem; color: #666; margin-bottom: 0.5rem; }
.evidence dl { margin: 0.5rem 0 0; display: grid; grid-template-columns: 120px 1fr; gap: 0.25rem 0.75rem; font-size: 0.9rem; }
.evidence dt { font-weight: 600; color: #444; }
table.data { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin-top: 0.5rem; }
table.data th, table.data td { border: 1px solid var(--border); padding: 0.5rem 0.65rem; text-align: left; }
table.data th { background: var(--bg); font-weight: 600; }
.section-num { color: var(--accent); font-weight: 700; margin-right: 0.35rem; }
footer.audit-footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.82rem; color: #666; }
.appendix pre { background: #f4f4f4; padding: 1rem; border-radius: 8px; overflow-x: auto; font-size: 0.78rem; max-height: 420px; }
"""


def evidence_detail_html(ev: dict[str, Any]) -> str:
    meta = ev.get("metadata") or {}
    q = meta.get("quality_score")
    qt = meta.get("quality_tier", "")
    quality_line = f'<div class="quality">Evidence quality: {q:.0%} ({qt})</div>' if q is not None else ""
    locator = ev.get("css_selector") or ev.get("xpath") or "—"
    return f"""<article class="evidence">
{quality_line}
<h3>{_esc(ev.get('rule_id') or ev.get('id'))}</h3>
<dl>
<dt>Statement</dt><dd>{_esc(ev.get('statement') or '—')}</dd>
<dt>Explanation</dt><dd>{_esc(ev.get('explanation') or '—')}</dd>
<dt>User Impact</dt><dd>{_esc(ev.get('user_impact') or '—')}</dd>
<dt>GDPR</dt><dd>{_esc(ev.get('gdpr_relevance') or '—')}</dd>
<dt>Recommendation</dt><dd>{_esc(ev.get('recommendation') or '—')}</dd>
<dt>Locator</dt><dd><code>{_esc(locator)}</code></dd>
</dl>
</article>"""
