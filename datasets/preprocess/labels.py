"""Unified fine-grained + binary label taxonomy for multimodal dark-pattern data."""

from __future__ import annotations

from dataclasses import dataclass


# Fine labels used across sources (snake_case).
FINE_LABELS = (
    "no_dark_pattern",
    "default_choice",
    "nagging",
    "disguised_ads",
    "gamification",
    "attention_distraction",
    "scarcity_urgency",
    "forced_action",
    "sneaking",
    "obstruction",
    "confirmshaming",
    "interface_interference",
    "hidden_subscription",
    "hidden_costs",
    "ui_component_context",  # B4E2: DP-context screenshots labeled by UI widgets
    "unknown",
)


@dataclass(frozen=True)
class NormalizedLabel:
    binary: str  # dark_pattern | no_dark_pattern
    fine: str
    consentshield: str  # maps toward ai.common.types.Category values
    source_labels: tuple[str, ...]


def _cs(name: str) -> str:
    return name


# ConsentShield Category string values
CS_NO = "No Dark Pattern"
CS_COOKIE = "Cookie Consent Manipulation"
CS_HIDDEN_SUB = "Hidden Subscription"
CS_HIDDEN_BILL = "Hidden Billing"
CS_CONFIRM = "Confirmshaming"
CS_FREE = "Misleading Free Trial"
CS_MIXED = "Mixed Consent Manipulation"
CS_UNKNOWN = "Unknown"


CONTEXTDP_MAP: dict[str, tuple[str, str]] = {
    # fine, consentshield
    "NO DP": ("no_dark_pattern", CS_NO),
    "DEFAULT CHOICE": ("default_choice", CS_COOKIE),
    "NAGGING": ("nagging", CS_UNKNOWN),
    "DISGUISED ADS": ("disguised_ads", CS_UNKNOWN),
    "GAMIFICATION": ("gamification", CS_UNKNOWN),
    "ATTENTION DISTRACTION": ("attention_distraction", CS_UNKNOWN),
    "ACTIVITY MESSAGE": ("scarcity_urgency", CS_UNKNOWN),
    "COUNTDOWN TIMER": ("scarcity_urgency", CS_UNKNOWN),
    "LIMITED TIME MESSAGE": ("scarcity_urgency", CS_UNKNOWN),
    "LOW STOCK MESSAGE": ("scarcity_urgency", CS_UNKNOWN),
    "HIGH DEMAND MESSAGE": ("scarcity_urgency", CS_UNKNOWN),
}


HF_LABEL_MAP: dict[str, tuple[str, str]] = {
    "non-deceptive": ("no_dark_pattern", CS_NO),
    "sneaking": ("sneaking", CS_HIDDEN_BILL),
    "interface-interference": ("interface_interference", CS_COOKIE),
    "forced-action": ("forced_action", CS_COOKIE),
    "obstruction": ("obstruction", CS_UNKNOWN),
}

HF_PATTERN_MAP: dict[str, tuple[str, str]] = {
    "not-applicable": ("no_dark_pattern", CS_NO),
    "disguised-ads": ("disguised_ads", CS_UNKNOWN),
    "trick-wording": ("interface_interference", CS_CONFIRM),
    "fake-scarcity-fake-urgency": ("scarcity_urgency", CS_UNKNOWN),
    "hidden-subscription": ("hidden_subscription", CS_HIDDEN_SUB),
    "forced-action": ("forced_action", CS_COOKIE),
    "hidden-costs": ("hidden_costs", CS_HIDDEN_BILL),
    "nudge": ("interface_interference", CS_COOKIE),
    "visual-interference": ("attention_distraction", CS_UNKNOWN),
    "confirmshaming": ("confirmshaming", CS_CONFIRM),
    "pre-selection": ("default_choice", CS_COOKIE),
}


B4E2_COMPONENT_MAP = {
    "0": "button",
    "1": "checkbox",
    "2": "input_field",
    "3": "popup",
    "4": "qr_code",
    "button": "button",
    "checkbox": "checkbox",
    "input_field": "input_field",
    "popup": "popup",
    "qr_code": "qr_code",
}


def normalize_contextdp(category_names: list[str]) -> NormalizedLabel:
    names = [n.strip().upper() for n in category_names if n and n.strip()]
    if not names:
        return NormalizedLabel("no_dark_pattern", "no_dark_pattern", CS_NO, ())
    # Prefer any non-NO-DP label
    dp = [n for n in names if n != "NO DP"]
    chosen = dp[0] if dp else "NO DP"
    fine, cs = CONTEXTDP_MAP.get(chosen, ("unknown", CS_UNKNOWN))
    binary = "no_dark_pattern" if fine == "no_dark_pattern" else "dark_pattern"
    if len(set(dp)) > 1:
        cs = CS_MIXED
        if fine == "no_dark_pattern":
            fine = "unknown"
            binary = "dark_pattern"
    return NormalizedLabel(binary, fine, cs, tuple(sorted(set(names))))


def normalize_hf(label: str, pattern_type: str | None = None) -> NormalizedLabel:
    lab = (label or "").strip().lower()
    pat = (pattern_type or "").strip().lower()
    if pat and pat in HF_PATTERN_MAP and lab != "non-deceptive":
        fine, cs = HF_PATTERN_MAP[pat]
    elif lab in HF_LABEL_MAP:
        fine, cs = HF_LABEL_MAP[lab]
        if pat and pat in HF_PATTERN_MAP and lab != "non-deceptive":
            fine, cs = HF_PATTERN_MAP[pat]
    else:
        fine, cs = ("unknown", CS_UNKNOWN)
    binary = "no_dark_pattern" if fine == "no_dark_pattern" else "dark_pattern"
    src = (lab,) if not pat else (lab, pat)
    return NormalizedLabel(binary, fine, cs, src)


def normalize_hf_page(element_outputs: list[tuple[str, str]]) -> NormalizedLabel:
    """Aggregate multi-element page annotations into one page label."""
    if not element_outputs:
        return NormalizedLabel("no_dark_pattern", "no_dark_pattern", CS_NO, ())
    norms = [normalize_hf(lab, pat) for lab, pat in element_outputs]
    dp = [n for n in norms if n.binary == "dark_pattern"]
    if not dp:
        src = tuple(sorted({s for n in norms for s in n.source_labels}))
        return NormalizedLabel("no_dark_pattern", "no_dark_pattern", CS_NO, src)
    # Majority fine label among deceptive elements
    counts: dict[str, int] = {}
    for n in dp:
        counts[n.fine] = counts.get(n.fine, 0) + 1
    fine = sorted(counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
    chosen = next(n for n in dp if n.fine == fine)
    src = tuple(sorted({s for n in dp for s in n.source_labels}))
    cs = CS_MIXED if len({n.fine for n in dp}) > 1 else chosen.consentshield
    return NormalizedLabel("dark_pattern", fine, cs, src)


def normalize_b4e2(components: list[str]) -> NormalizedLabel:
    comps = tuple(sorted({B4E2_COMPONENT_MAP.get(c, c) for c in components if c}))
    # Entire corpus is DP-context UI screenshots
    return NormalizedLabel(
        "dark_pattern",
        "ui_component_context",
        CS_UNKNOWN,
        comps or ("ui_component_context",),
    )
