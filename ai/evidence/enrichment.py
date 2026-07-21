"""Attach real DOM/CSS locators and page context to evidence items."""

from __future__ import annotations

from typing import Any

from ai.common.types import BoundingRect, EvidenceItem, ScanPayload


_STYLE_KEYS = (
    "fontSizePx",
    "fontWeight",
    "backgroundColor",
    "color",
    "display",
    "visibility",
    "opacity",
    "textDecoration",
)


def _bounding_rect_from(btn: dict[str, Any]) -> BoundingRect | None:
    rect = btn.get("boundingRect") or btn.get("bounding_rect")
    if isinstance(rect, dict):
        required = ("x", "y", "width", "height", "top", "left", "right", "bottom")
        if all(k in rect and rect[k] is not None for k in required):
            return BoundingRect(
                x=float(rect["x"]),
                y=float(rect["y"]),
                width=float(rect["width"]),
                height=float(rect["height"]),
                top=float(rect["top"]),
                left=float(rect["left"]),
                right=float(rect["right"]),
                bottom=float(rect["bottom"]),
            )
    # Fallback: only when full geometry was collected on the button itself
    keys = ("x", "y", "width", "height", "top", "left", "right", "bottom")
    if all(k in btn and btn[k] is not None for k in keys):
        return BoundingRect(
            x=float(btn["x"]),
            y=float(btn["y"]),
            width=float(btn["width"]),
            height=float(btn["height"]),
            top=float(btn["top"]),
            left=float(btn["left"]),
            right=float(btn["right"]),
            bottom=float(btn["bottom"]),
        )
    return None


def _computed_styles_from(btn: dict[str, Any]) -> dict[str, Any] | None:
    styles = btn.get("computedStyles") or btn.get("computed_styles")
    if isinstance(styles, dict) and styles:
        return {k: v for k, v in styles.items() if v is not None}
    out: dict[str, Any] = {}
    for key in _STYLE_KEYS:
        if key in btn and btn[key] is not None:
            out[key] = btn[key]
    return out or None


def _match_button(
    css: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any] | None:
    buttons = list(css.get("buttons", []) or [])
    if not buttons:
        return None

    # Prefer explicit element reference from rule metadata
    for key in ("accept_text", "reject_text", "settings_text", "cta", "label", "match"):
        needle = metadata.get(key)
        if not needle:
            continue
        needle_s = str(needle).strip().lower()
        for btn in buttons:
            label = str(btn.get("text", "") or btn.get("ariaLabel", "")).strip().lower()
            if label and (needle_s in label or label in needle_s):
                return btn

    # Banner geometry when obstruction rules fire
    banner = css.get("banner")
    if isinstance(banner, dict) and banner.get("xpath"):
        return banner

    return None


def enrich_evidence_item(
    item: EvidenceItem,
    payload: ScanPayload,
    *,
    element: dict[str, Any] | None = None,
) -> EvidenceItem:
    """
    Fill enrichment fields only when real collected values exist.
    Never invent xpath/selector/rect placeholders.
    """
    data = item.model_dump()
    css = payload.css_snapshot or {}
    meta = item.metadata or {}

    el = element or _match_button(css, meta)

    if payload.url:
        data["url"] = payload.url
    if payload.title:
        data["page_title"] = payload.title
    if payload.collected_at:
        data["timestamp"] = payload.collected_at
    if payload.viewport and payload.viewport.get("width") and payload.viewport.get("height"):
        data["viewport"] = {
            "width": int(payload.viewport["width"]),
            "height": int(payload.viewport["height"]),
        }
    if payload.scroll_position:
        sp = payload.scroll_position
        if "x" in sp and "y" in sp and sp["x"] is not None and sp["y"] is not None:
            data["scroll_position"] = {"x": int(sp["x"]), "y": int(sp["y"])}

    if el:
        if el.get("xpath"):
            data["xpath"] = str(el["xpath"])
        if el.get("cssSelector") or el.get("css_selector"):
            data["css_selector"] = str(el.get("cssSelector") or el.get("css_selector"))
        if el.get("domPath") or el.get("dom_path"):
            data["dom_path"] = str(el.get("domPath") or el.get("dom_path"))
        if el.get("htmlSnippet") or el.get("html_snippet"):
            data["html_snippet"] = str(el.get("htmlSnippet") or el.get("html_snippet"))

        rect = _bounding_rect_from(el)
        if rect is not None:
            data["bounding_rect"] = rect.model_dump()

        styles = _computed_styles_from(el)
        if styles:
            data["computed_styles"] = styles

    return EvidenceItem(**data)
