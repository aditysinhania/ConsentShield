"""Generate annotated PNG screenshots with consent UI highlights."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from ai.common.types import ExplainableReport, ScanPayload

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - tested with pillow installed
    Image = None  # type: ignore[misc, assignment]
    ImageDraw = None  # type: ignore[misc, assignment]
    ImageFont = None  # type: ignore[misc, assignment]

ACCEPT_PATTERNS = re.compile(
    r"\b(accept\s*all|allow\s*all|agree\s*(to)?\s*all|i\s*agree|accept)\b",
    re.I,
)
REJECT_PATTERNS = re.compile(
    r"\b(reject\s*all|decline\s*all|refuse|deny\s*all|necessary\s*only|"
    r"essential\s*only|reject\s*non[- ]essential|reject)\b",
    re.I,
)
SETTINGS_PATTERNS = re.compile(
    r"\b(manage\s*(preferences|cookies|settings)|cookie\s*settings|"
    r"customise|customize|more\s*options|settings)\b",
    re.I,
)

_COLORS = {
    "banner": (52, 120, 246, 180),
    "accept": (34, 160, 80, 200),
    "reject": (220, 60, 60, 200),
    "settings": (230, 140, 30, 200),
    "dark_pattern": (160, 60, 200, 200),
}

_LEGEND = [
    ("Banner", _COLORS["banner"]),
    ("Accept", _COLORS["accept"]),
    ("Reject", _COLORS["reject"]),
    ("Settings", _COLORS["settings"]),
    ("Dark Pattern", _COLORS["dark_pattern"]),
]


def _rect_from_element(el: dict[str, Any]) -> tuple[int, int, int, int] | None:
    rect = el.get("boundingRect") or el.get("bounding_rect") or el
    keys = ("left", "top", "width", "height")
    if not all(k in rect and rect[k] is not None for k in keys):
        if all(k in el and el[k] is not None for k in ("x", "y", "width", "height")):
            x, y, w, h = int(el["x"]), int(el["y"]), int(el["width"]), int(el["height"])
            return x, y, x + w, y + h
        return None
    left = int(rect["left"])
    top = int(rect["top"])
    width = int(rect["width"])
    height = int(rect["height"])
    if width <= 0 or height <= 0:
        return None
    return left, top, left + width, top + height


def _classify_button(btn: dict[str, Any]) -> str:
    label = str(btn.get("text", "") or btn.get("ariaLabel", ""))
    if ACCEPT_PATTERNS.search(label):
        return "accept"
    if REJECT_PATTERNS.search(label):
        return "reject"
    if SETTINGS_PATTERNS.search(label):
        return "settings"
    return "dark_pattern"


def _default_font(size: int = 14):
    if ImageFont is None:
        return None
    for name in ("arial.ttf", "Arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_box(
    draw: Any,
    box: tuple[int, int, int, int],
    color: tuple[int, int, int, int],
    label: str,
    *,
    font: Any,
    confidence: float | None = None,
    severity: str | None = None,
    callout: int | None = None,
) -> None:
    x1, y1, x2, y2 = box
    outline = color[:3]
    width = 4 if severity in ("HIGH", "CRITICAL") else 3 if severity == "MEDIUM" else 2

    # Corner brackets
    arm = min(14, int((x2 - x1) * 0.15), int((y2 - y1) * 0.15))
    for cx, cy, dx, dy in (
        (x1, y1, 1, 1),
        (x2, y1, -1, 1),
        (x1, y2, 1, -1),
        (x2, y2, -1, -1),
    ):
        draw.line([(cx, cy), (cx + dx * arm, cy)], fill=outline, width=width)
        draw.line([(cx, cy), (cx, cy + dy * arm)], fill=outline, width=width)

    draw.rectangle([x1, y1, x2, y2], outline=outline, width=width)
    overlay = color[:3] + (35,)
    draw.rectangle([x1, y1, x2, y2], fill=overlay)

    parts = [label]
    if callout is not None:
        parts.insert(0, f"#{callout}")
    if confidence is not None:
        parts.append(f"conf {confidence:.0%}")
    if severity:
        parts.append(severity)
    text = " · ".join(parts)
    text_y = max(y1 - 20, 4)
    badge_w = min(len(text) * 7 + 16, 440)
    draw.rectangle([x1, text_y - 2, x1 + badge_w, text_y + 17], fill=outline)
    draw.text((x1 + 4, text_y), text, fill=(255, 255, 255), font=font)


def _collect_regions(
    payload: ScanPayload,
    report: ExplainableReport,
) -> list[tuple[str, tuple[int, int, int, int], str, float | None, str | None]]:
    css = payload.css_snapshot or {}
    regions: list[tuple[str, tuple[int, int, int, int], str, float | None, str | None]] = []

    banner = css.get("banner")
    if isinstance(banner, dict):
        box = _rect_from_element(banner)
        if box:
            regions.append(("banner", box, "Banner", report.confidence, None))

    for btn in list(css.get("buttons", []) or []):
        box = _rect_from_element(btn)
        if not box:
            continue
        kind = _classify_button(btn)
        label = str(btn.get("text", "") or btn.get("ariaLabel", "") or kind.title())[:40]
        regions.append((kind, box, label, report.confidence, None))

    severity_label = None
    if report.severity and isinstance(report.severity, dict):
        severity_label = str(report.severity.get("level", ""))

    for item in report.evidence:
        if item.bounding_rect is None:
            continue
        br = item.bounding_rect
        box = (
            int(br.left),
            int(br.top),
            int(br.right),
            int(br.bottom),
        )
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        label = (item.rule_id or "finding")[:36]
        regions.append(
            (
                "dark_pattern",
                box,
                label,
                report.confidence,
                severity_label,
            )
        )

    return regions


def annotate_screenshot(
    *,
    source_path: str | Path | None,
    output_path: str | Path,
    payload: ScanPayload,
    report: ExplainableReport,
    viewport: dict[str, int] | None = None,
) -> Path:
    """
    Draw highlight boxes, legend, and footer; write PNG to output_path.
    Creates a blank canvas if source_path is missing (tests).
    """
    if Image is None or ImageDraw is None:
        raise RuntimeError("Pillow is required for annotated screenshots")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    vp = viewport or payload.viewport or {"width": 1280, "height": 800}
    width = int(vp.get("width", 1280))
    height = int(vp.get("height", 800))

    if source_path and Path(source_path).is_file():
        base = Image.open(source_path).convert("RGBA")
        width, height = base.size
    else:
        base = Image.new("RGBA", (width, height), (245, 245, 245, 255))

    font = _default_font(13)
    small = _default_font(11)

    regions = _collect_regions(payload, report)

    # Header strip with audit summary
    header_h = 28
    header = Image.new("RGBA", (base.width, header_h), (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(header)
    hdraw.rectangle([0, 0, base.width, header_h], fill=(27, 67, 50, 230))
    header_text = f"ConsentShield Annotated Audit · Risk {report.risk_score:.0f} · {len(regions)} region(s)"
    hdraw.text((10, 6), header_text, fill=(255, 255, 255), font=small)
    base_with_header = Image.new("RGBA", (base.width, base.height + header_h), (0, 0, 0, 0))
    base_with_header.paste(header, (0, 0))
    base_with_header.paste(base, (0, header_h))
    base = base_with_header

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    shifted_regions = [
        (kind, (box[0], box[1] + header_h, box[2], box[3] + header_h), label, conf, sev)
        for kind, box, label, conf, sev in regions
    ]
    for idx, (kind, box, label, conf, sev) in enumerate(shifted_regions, start=1):
        color = _COLORS.get(kind, _COLORS["dark_pattern"])
        _draw_box(draw, box, color, label, font=font, confidence=conf, severity=sev, callout=idx)

    # Legend panel (top-right)
    lx, ly = base.width - 210, 12
    draw.rectangle([lx, ly, base.width - 8, ly + 8 + len(_LEGEND) * 22], fill=(20, 20, 20, 210))
    for i, (name, rgba) in enumerate(_LEGEND):
        y = ly + 10 + i * 22
        draw.rectangle([lx + 8, y, lx + 28, y + 14], fill=rgba[:3])
        draw.text((lx + 36, y - 1), name, fill=(255, 255, 255), font=small)

    composed = Image.alpha_composite(base.convert("RGBA"), overlay)

    # Footer bar
    footer_h = 36
    footer = Image.new("RGBA", (composed.width, composed.height + footer_h), (0, 0, 0, 0))
    footer.paste(composed, (0, 0))
    fdraw = ImageDraw.Draw(footer)
    fdraw.rectangle([0, composed.height, composed.width, composed.height + footer_h], fill=(25, 25, 25, 240))
    sev = "—"
    if report.severity and isinstance(report.severity, dict):
        sev = str(report.severity.get("level", "—"))
    footer_text = (
        f"ConsentShield · Risk {report.risk_score:.0f} · "
        f"Confidence {report.confidence:.0%} · Severity {sev} · {payload.url[:80]}"
    )
    fdraw.text((12, composed.height + 10), footer_text, fill=(230, 230, 230), font=small)

    footer.convert("RGB").save(out, format="PNG")
    return out
