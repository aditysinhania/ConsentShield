"""Rule evaluators — each returns triggered/score/evidence or None."""

from __future__ import annotations

import re
from typing import Any, Callable

EvaluatorFn = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any] | None]

# Normalized consent vocabulary — real-world CMP labels (not site-specific).
REJECT_PATTERNS = re.compile(
    r"\b("
    r"reject(\s+all|\s+cookies|\s+non[- ]essential)?|"
    r"decline(\s+all|\s+cookies)?|"
    r"refuse(\s+all)?|"
    r"deny(\s+all)?|"
    r"disagree|"
    r"no[,\s]+thank\s*you|"
    r"no\s+thanks|"
    r"only\s+necessary|"
    r"necessary\s+only|"
    r"essential\s+only|"
    r"required\s+only|"
    r"reject\s+non[- ]essential"
    r")\b",
    re.I,
)
ACCEPT_PATTERNS = re.compile(
    r"\b("
    r"accept(\s+all|\s+cookies|\s+selected)?|"
    r"allow(\s+all|\s+cookies)?|"
    r"agree(\s+to\s+all|\s+and\s+continue)?|"
    r"i\s+agree|"
    r"yes[,\s]+i\s+accept|"
    r"continue\s+with\s+cookies|"
    r"confirm\s+choices|"
    r"got\s+it|"
    r"ok[,\s]+i\s+agree"
    r")\b",
    re.I,
)
SETTINGS_PATTERNS = re.compile(
    r"\b("
    r"manage\s*(preferences|cookies|settings|options)?|"
    r"cookie\s*settings|"
    r"privacy\s*settings|"
    r"customise|customize|"
    r"preferences|"
    r"more\s*options|"
    r"save\s+preferences|"
    r"confirm\s+my\s+choices|"
    r"cookie\s+preferences|"
    r"consent\s+settings"
    r")\b",
    re.I,
)
AUTO_RENEW = re.compile(
    r"\b(auto[- ]?renew(al|s)?|automatically\s+renew|recurring\s+(billing|charge|payment)|"
    r"billed\s+(monthly|annually|yearly)|subscription\s+will\s+(continue|renew))\b",
    re.I,
)
FREE_TRIAL = re.compile(
    r"\b(free\s+trial|try\s+(it\s+)?free|\$0(\.00)?\s*(today|now)|"
    r"cancel\s+anytime|then\s+\$?\d+)\b",
    re.I,
)
CONFIRMSHAME = re.compile(
    r"\b(no\s+thanks[,.]?\s+i\s+(don'?t\s+want|hate)|"
    r"i\s+don'?t\s+want\s+(to\s+)?(save|protect|improve)|"
    r"continue\s+without|miss\s+out)\b",
    re.I,
)
MARKETING_PRESELECT = re.compile(
    r"\b(marketing|advertising|targeting|personalisation|personalization)\b",
    re.I,
)


def _css_buttons(css: dict[str, Any]) -> list[dict[str, Any]]:
    return list(css.get("buttons", []) or [])


def _find_button(buttons: list[dict[str, Any]], pattern: re.Pattern[str]) -> dict[str, Any] | None:
    for btn in buttons:
        label = str(btn.get("text", "") or btn.get("ariaLabel", ""))
        if pattern.search(label):
            return btn
    return None


def hidden_reject(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    text = ctx.get("text", "")
    css = ctx.get("css", {})
    buttons = _css_buttons(css)
    reject_btn = _find_button(buttons, REJECT_PATTERNS)
    accept_btn = _find_button(buttons, ACCEPT_PATTERNS)

    # Accept present, reject absent in visible text & button list
    if accept_btn and not reject_btn and not REJECT_PATTERNS.search(text):
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.9)),
            "evidence": "Accept control found but no visible Reject / Reject All control.",
            "metadata": {"accept_text": accept_btn.get("text")},
            "features_used": [
                "buttons.text",
                "buttons.ariaLabel",
                "visible_text",
            ],
            "text_score": float(rule.get("severity", 0.9)),
            "visual_score": 0.0,
            "layout_score": 0.0,
        }

    if reject_btn:
        display = str(reject_btn.get("display", "block")).lower()
        visibility = str(reject_btn.get("visibility", "visible")).lower()
        opacity = float(reject_btn.get("opacity", 1) or 1)
        font_size = float(reject_btn.get("fontSizePx", 14) or 14)
        if display == "none" or visibility == "hidden" or opacity < 0.15 or font_size < 8:
            return {
                "triggered": True,
                "score": float(rule.get("severity", 0.9)),
                "evidence": "Reject button exists in DOM but is hidden via CSS.",
                "metadata": {
                    "display": display,
                    "visibility": visibility,
                    "opacity": opacity,
                    "fontSizePx": font_size,
                },
                "features_used": [
                    "buttons.display",
                    "buttons.visibility",
                    "buttons.opacity",
                    "buttons.fontSizePx",
                ],
                "visual_score": float(rule.get("severity", 0.9)),
                "layout_score": 0.6,
                "text_score": 0.3,
            }
    return None


def multi_click_reject(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    text = ctx.get("text", "")
    css = ctx.get("css", {})
    buttons = _css_buttons(css)
    reject_btn = _find_button(buttons, REJECT_PATTERNS)
    settings_btn = _find_button(buttons, SETTINGS_PATTERNS)
    accept_btn = _find_button(buttons, ACCEPT_PATTERNS)

    if accept_btn and not reject_btn and (settings_btn or SETTINGS_PATTERNS.search(text)):
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.75)),
            "evidence": "Reject requires opening Manage / Cookie Settings instead of one-click Reject All.",
            "metadata": {"settings_path": True},
        }
    return None


def accept_visually_dominant(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    buttons = _css_buttons(ctx.get("css", {}))
    accept_btn = _find_button(buttons, ACCEPT_PATTERNS)
    reject_btn = _find_button(buttons, REJECT_PATTERNS)
    if not accept_btn or not reject_btn:
        return None

    a_area = float(accept_btn.get("width", 0) or 0) * float(accept_btn.get("height", 0) or 0)
    r_area = float(reject_btn.get("width", 0) or 0) * float(reject_btn.get("height", 0) or 0)
    a_weight = float(accept_btn.get("fontWeight", 400) or 400)
    r_weight = float(reject_btn.get("fontWeight", 400) or 400)

    area_ratio = (a_area / r_area) if r_area > 0 else 999.0
    if area_ratio >= 1.5 or (a_weight - r_weight) >= 200:
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.8)),
            "evidence": "Accept button is visually dominant over Reject (size/weight).",
            "metadata": {"area_ratio": round(area_ratio, 2), "accept_weight": a_weight, "reject_weight": r_weight},
        }
    return None


def unequal_emphasis(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    buttons = _css_buttons(ctx.get("css", {}))
    accept_btn = _find_button(buttons, ACCEPT_PATTERNS)
    reject_btn = _find_button(buttons, REJECT_PATTERNS)
    if not accept_btn or not reject_btn:
        return None

    a_bg = str(accept_btn.get("backgroundColor", "")).lower()
    r_bg = str(reject_btn.get("backgroundColor", "")).lower()
    a_filled = a_bg not in ("", "transparent", "rgba(0, 0, 0, 0)")
    r_link_like = r_bg in ("", "transparent", "rgba(0, 0, 0, 0)") or reject_btn.get("textDecoration") == "underline"

    if a_filled and r_link_like:
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.7)),
            "evidence": "Accept is a filled button while Reject is styled as a low-emphasis link.",
            "metadata": {"accept_bg": a_bg, "reject_bg": r_bg},
        }
    return None


def preselected_marketing(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    css = ctx.get("css", {})
    checkboxes = list(css.get("checkboxes", []) or [])
    toggles = list(css.get("toggles", []) or [])
    for item in checkboxes + toggles:
        label = str(item.get("label", "") or item.get("name", ""))
        checked = bool(item.get("checked") or item.get("ariaChecked") is True)
        required = bool(item.get("required") or item.get("essential"))
        if checked and not required and MARKETING_PRESELECT.search(label):
            return {
                "triggered": True,
                "score": float(rule.get("severity", 0.85)),
                "evidence": f"Non-essential category pre-selected: '{label}'.",
                "metadata": {"label": label},
            }
    return None


def banner_obstruction(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    css = ctx.get("css", {})
    viewport = ctx.get("viewport") or {}
    banner = css.get("banner") or {}
    vw = float(viewport.get("width", 0) or 0)
    vh = float(viewport.get("height", 0) or 0)
    bw = float(banner.get("width", 0) or 0)
    bh = float(banner.get("height", 0) or 0)
    if vw <= 0 or vh <= 0 or bw <= 0 or bh <= 0:
        return None
    coverage = (bw * bh) / (vw * vh)
    if coverage >= 0.35:
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.55)),
            "evidence": f"Cookie banner covers ~{coverage:.0%} of the viewport.",
            "metadata": {"coverage": round(coverage, 3)},
        }
    return None


def auto_renew(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    text = ctx.get("text", "")
    m = AUTO_RENEW.search(text)
    if not m:
        return None
    css = ctx.get("css", {})
    billing = css.get("billingSection") or {}
    font_size = float(billing.get("fontSizePx", 14) or 14)
    near_cta = bool(billing.get("nearPrimaryCta", True))
    score = float(rule.get("severity", 0.8))
    evidence = f"Auto-renewal language detected: '{m.group(0)}'."
    if font_size < 11 or not near_cta:
        evidence += " Billing notice appears de-emphasized relative to the CTA."
        score = min(1.0, score + 0.05)
    return {"triggered": True, "score": score, "evidence": evidence, "metadata": {"match": m.group(0)}}


def hidden_billing(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    css = ctx.get("css", {})
    billing = css.get("billingSection") or {}
    if not billing:
        text = ctx.get("text", "")
        if AUTO_RENEW.search(text) and "below_fold" in str(css.get("layoutHints", [])):
            return {
                "triggered": True,
                "score": float(rule.get("severity", 0.85)),
                "evidence": "Recurring billing text appears below the fold / away from primary CTA.",
                "metadata": {},
            }
        return None
    if billing.get("hidden") or float(billing.get("opacity", 1) or 1) < 0.2:
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.85)),
            "evidence": "Billing section is hidden or near-invisible via CSS.",
            "metadata": billing,
        }
    return None


def small_font_disclaimer(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    css = ctx.get("css", {})
    billing = css.get("billingSection") or {}
    body = float((css.get("body") or {}).get("fontSizePx", 16) or 16)
    disclaimer = float(billing.get("fontSizePx", body) or body)
    if disclaimer > 0 and body > 0 and disclaimer <= body * 0.7 and disclaimer < 12:
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.7)),
            "evidence": f"Billing disclaimer font ({disclaimer}px) is much smaller than body ({body}px).",
            "metadata": {"disclaimerPx": disclaimer, "bodyPx": body},
        }
    return None


def misleading_free_trial(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    text = ctx.get("text", "")
    if FREE_TRIAL.search(text) and AUTO_RENEW.search(text):
        # Trial + auto-renew together — flag when trial is prominent in title/CTA area
        css = ctx.get("css", {})
        cta = str((css.get("primaryCta") or {}).get("text", ""))
        if re.search(r"free|trial|\$0", cta, re.I) or re.search(r"\bfree\s+trial\b", text[:500], re.I):
            return {
                "triggered": True,
                "score": float(rule.get("severity", 0.8)),
                "evidence": "Free trial is highlighted while auto-renew / paid conversion terms are also present.",
                "metadata": {"cta": cta},
            }
    return None


def preselected_plan(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    css = ctx.get("css", {})
    plans = list(css.get("subscriptionOptions", []) or [])
    for plan in plans:
        if plan.get("checked") or plan.get("selected"):
            label = str(plan.get("label", "plan"))
            if re.search(r"annual|yearly|premium|pro|plus", label, re.I):
                return {
                    "triggered": True,
                    "score": float(rule.get("severity", 0.65)),
                    "evidence": f"Higher-cost plan pre-selected: '{label}'.",
                    "metadata": {"label": label},
                }
    return None


def confirmshaming(ctx: dict[str, Any], rule: dict[str, Any]) -> dict[str, Any] | None:
    text = ctx.get("text", "")
    m = CONFIRMSHAME.search(text)
    if m:
        return {
            "triggered": True,
            "score": float(rule.get("severity", 0.7)),
            "evidence": f"Confirmshaming language detected: '{m.group(0)}'.",
            "metadata": {"match": m.group(0)},
        }
    return None


EVALUATORS: dict[str, EvaluatorFn] = {
    "hidden_reject": hidden_reject,
    "multi_click_reject": multi_click_reject,
    "accept_visually_dominant": accept_visually_dominant,
    "unequal_emphasis": unequal_emphasis,
    "preselected_marketing": preselected_marketing,
    "banner_obstruction": banner_obstruction,
    "auto_renew": auto_renew,
    "hidden_billing": hidden_billing,
    "small_font_disclaimer": small_font_disclaimer,
    "misleading_free_trial": misleading_free_trial,
    "preselected_plan": preselected_plan,
    "confirmshaming": confirmshaming,
}
