"""Dedicated accessibility analysis — stored separately from rule findings."""

from __future__ import annotations

import re
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from ai.common.types import ScanPayload


class AccessibilityIssueType(str, Enum):
    LOW_CONTRAST = "low_contrast"
    TINY_BUTTON = "tiny_button"
    KEYBOARD_ISSUE = "keyboard_issue"
    FOCUS_TRAP = "focus_trap"
    MISSING_ARIA = "missing_aria"
    MISSING_ROLE = "missing_role"
    TABINDEX_MISSING = "tabindex_missing"
    OVERLAY_BLOCKING = "overlay_blocking"


class AccessibilityIssue(BaseModel):
    type: AccessibilityIssueType
    severity: float = Field(ge=0.0, le=1.0)
    description: str
    element_label: str | None = None
    xpath: str | None = None
    css_selector: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class AccessibilityReport(BaseModel):
    status: str = "ready"
    issue_count: int = 0
    aggregate_score: float = Field(ge=0.0, le=1.0, default=0.0)
    issues: list[AccessibilityIssue] = Field(default_factory=list)


_RGB_RE = re.compile(
    r"rgba?\(\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)"
)


def _parse_rgb(color: str) -> tuple[float, float, float] | None:
    m = _RGB_RE.match(str(color).strip())
    if not m:
        return None
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


def _relative_luminance(r: float, g: float, b: float) -> float:
    def channel(c: float) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * channel(r) + 0.7152 * channel(g) + 0.0722 * channel(b)


def _contrast_ratio(fg: str, bg: str) -> float | None:
    fg_rgb = _parse_rgb(fg)
    bg_rgb = _parse_rgb(bg)
    if fg_rgb is None or bg_rgb is None:
        return None
    l1 = _relative_luminance(*fg_rgb)
    l2 = _relative_luminance(*bg_rgb)
    lighter = max(l1, l2)
    darker = min(l1, l2)
    return (lighter + 0.05) / (darker + 0.05)


def _buttons(css: dict[str, Any]) -> list[dict[str, Any]]:
    return list(css.get("buttons", []) or [])


def analyze_accessibility(payload: ScanPayload) -> AccessibilityReport:
    css = payload.css_snapshot or {}
    viewport = payload.viewport or {}
    issues: list[AccessibilityIssue] = []

    vw = float(viewport.get("width", 0) or 0)
    vh = float(viewport.get("height", 0) or 0)
    banner = css.get("banner") or {}

    # Overlay blocking / focus trap heuristics from banner geometry
    if isinstance(banner, dict):
        bw = float(banner.get("width", 0) or 0)
        bh = float(banner.get("height", 0) or 0)
        if vw > 0 and vh > 0 and bw > 0 and bh > 0:
            coverage = (bw * bh) / (vw * vh)
            styles = banner.get("computedStyles") or {}
            position = str(styles.get("position", "")).lower()
            z_index = str(styles.get("zIndex", "0"))
            high_z = z_index.isdigit() and int(z_index) >= 1000
            if coverage >= 0.35:
                issues.append(
                    AccessibilityIssue(
                        type=AccessibilityIssueType.OVERLAY_BLOCKING,
                        severity=min(1.0, 0.5 + coverage),
                        description=f"Cookie/consent overlay covers ~{coverage:.0%} of the viewport.",
                        xpath=banner.get("xpath"),
                        css_selector=banner.get("cssSelector") or banner.get("css_selector"),
                        metadata={"coverage": round(coverage, 3)},
                    )
                )
            if coverage >= 0.25 and (position == "fixed" or high_z):
                issues.append(
                    AccessibilityIssue(
                        type=AccessibilityIssueType.FOCUS_TRAP,
                        severity=min(1.0, 0.45 + coverage * 0.5),
                        description=(
                            "Fixed/high z-index overlay may trap keyboard focus until dismissed."
                        ),
                        xpath=banner.get("xpath"),
                        css_selector=banner.get("cssSelector") or banner.get("css_selector"),
                        metadata={"position": position, "zIndex": z_index, "coverage": round(coverage, 3)},
                    )
                )

    for btn in _buttons(css):
        label = str(btn.get("text", "") or btn.get("ariaLabel", "")).strip()
        xpath = btn.get("xpath")
        selector = btn.get("cssSelector") or btn.get("css_selector")
        width = float(btn.get("width", 0) or 0)
        height = float(btn.get("height", 0) or 0)
        font_size = float(btn.get("fontSizePx", 0) or 0)
        display = str(btn.get("display", "block")).lower()
        visibility = str(btn.get("visibility", "visible")).lower()
        opacity = float(btn.get("opacity", 1) or 1)

        # Tiny buttons / touch targets
        if width > 0 and height > 0 and (width < 44 or height < 44 or font_size < 12):
            sev = 0.55
            if width < 32 or height < 32 or font_size < 10:
                sev = 0.75
            issues.append(
                AccessibilityIssue(
                    type=AccessibilityIssueType.TINY_BUTTON,
                    severity=sev,
                    description=(
                        f"Interactive control '{label or 'unnamed'}' is below recommended "
                        f"44×44px or 12px text ({int(width)}×{int(height)}px, {font_size}px)."
                    ),
                    element_label=label or None,
                    xpath=xpath,
                    css_selector=selector,
                    metadata={"width": width, "height": height, "fontSizePx": font_size},
                )
            )

        # Missing accessible name
        if not label and display != "none" and visibility != "hidden" and opacity >= 0.15:
            issues.append(
                AccessibilityIssue(
                    type=AccessibilityIssueType.MISSING_ARIA,
                    severity=0.7,
                    description="Button-like control has no visible text or aria-label.",
                    xpath=xpath,
                    css_selector=selector,
                    metadata={"display": display, "visibility": visibility},
                )
            )

        # Custom interactive element missing role
        role = str(btn.get("role", "") or "").lower()
        tag_hint = str(btn.get("tagName", "") or btn.get("tag", "")).lower()
        if (
            not role
            and tag_hint not in ("button", "input", "a")
            and width > 0
            and height > 0
            and display not in ("none", "")
            and visibility != "hidden"
        ):
            issues.append(
                AccessibilityIssue(
                    type=AccessibilityIssueType.MISSING_ROLE,
                    severity=0.6,
                    description=(
                        f"Interactive control '{label or 'unnamed'}' lacks an explicit ARIA role."
                    ),
                    element_label=label or None,
                    xpath=xpath,
                    css_selector=selector,
                    metadata={"tag": tag_hint or "unknown"},
                )
            )

        # tabindex not keyboard reachable when hidden off-screen pattern
        tabindex = btn.get("tabindex")
        if (
            tabindex is not None
            and str(tabindex) == "-1"
            and label
            and display != "none"
            and opacity >= 0.15
        ):
            issues.append(
                AccessibilityIssue(
                    type=AccessibilityIssueType.TABINDEX_MISSING,
                    severity=0.55,
                    description=(
                        f"Control '{label}' uses tabindex=-1 and may be excluded from tab order."
                    ),
                    element_label=label,
                    xpath=xpath,
                    css_selector=selector,
                    metadata={"tabindex": tabindex},
                )
            )

        # Low contrast
        fg = str(btn.get("color", ""))
        bg = str(btn.get("backgroundColor", ""))
        ratio = _contrast_ratio(fg, bg)
        if ratio is not None and ratio < 4.5:
            issues.append(
                AccessibilityIssue(
                    type=AccessibilityIssueType.LOW_CONTRAST,
                    severity=0.65 if ratio >= 3.0 else 0.85,
                    description=(
                        f"Control '{label or 'unnamed'}' contrast ratio {ratio:.1f}:1 "
                        f"is below WCAG AA (4.5:1)."
                    ),
                    element_label=label or None,
                    xpath=xpath,
                    css_selector=selector,
                    metadata={"contrast_ratio": round(ratio, 2), "foreground": fg, "background": bg},
                )
            )

        # Keyboard / visibility issues
        if display == "none" or visibility == "hidden" or opacity < 0.15:
            if label or btn.get("ariaLabel"):
                issues.append(
                    AccessibilityIssue(
                        type=AccessibilityIssueType.KEYBOARD_ISSUE,
                        severity=0.8,
                        description=(
                            f"Control '{label or btn.get('ariaLabel')}' is not visible/focusable "
                            f"via standard keyboard navigation (hidden CSS)."
                        ),
                        element_label=label or str(btn.get("ariaLabel", "")),
                        xpath=xpath,
                        css_selector=selector,
                        metadata={"display": display, "visibility": visibility, "opacity": opacity},
                    )
                )

    aggregate = 0.0
    if issues:
        aggregate = round(min(1.0, sum(i.severity for i in issues) / len(issues)), 4)

    return AccessibilityReport(
        status="ready",
        issue_count=len(issues),
        aggregate_score=aggregate,
        issues=issues,
    )
