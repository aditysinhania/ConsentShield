"""Generate and persist multi-format audit reports."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ai.report.document import ReportDocument, build_report_document
from ai.report.exporters import export_html, export_json, export_markdown, export_pdf, write_exports


def generate_report_document(
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
    return build_report_document(
        scan_id=scan_id,
        url=url,
        title=title,
        risk_score=risk_score,
        category=category,
        confidence=confidence,
        report=report,
        screenshot_path=screenshot_path,
        annotated_screenshot_path=annotated_screenshot_path,
        narrator=narrator,
    )


def generate_all_formats(
    document: ReportDocument,
    output_dir: str | Path,
) -> dict[str, str]:
    return write_exports(document, output_dir)
