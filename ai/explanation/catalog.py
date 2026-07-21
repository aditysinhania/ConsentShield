"""Deterministic per-rule explanations — no LLM."""

from __future__ import annotations

from ai.common.types import RuleExplanation

# GDPR Art. 4(11), 7, 12, 25 — consent must be freely given, specific, informed,
# unambiguous; withdraw as easy as give; transparent information; data protection by design.

_RULE_EXPLANATIONS: dict[str, RuleExplanation] = {
    "cookie.hidden_reject": RuleExplanation(
        explanation=(
            "An Accept control is present while Reject / Reject All is missing from visible "
            "controls or is hidden with CSS (display, visibility, opacity, or tiny font)."
        ),
        user_impact=(
            "Users cannot refuse non-essential cookies as easily as they can accept them, "
            "pushing them toward broader tracking consent."
        ),
        gdpr_relevance=(
            "GDPR Art. 4(11) and Art. 7 require freely given, unambiguous consent; "
            "withdrawal of consent must be as easy as giving it (Art. 7(3))."
        ),
        recommendation=(
            "Expose a Reject / Reject All control at the same interaction depth and "
            "visibility as Accept All."
        ),
    ),
    "cookie.multi_click_reject": RuleExplanation(
        explanation=(
            "Rejecting non-essential cookies requires opening Manage Preferences / Cookie "
            "Settings instead of a one-click Reject All on the first layer."
        ),
        user_impact=(
            "Extra clicks create friction that steers users toward Accept All rather than "
            "a genuine choice."
        ),
        gdpr_relevance=(
            "GDPR Art. 7(3) and EDPB guidelines require that refusing consent must not be "
            "more difficult than accepting it."
        ),
        recommendation=(
            "Provide Reject All on the first banner layer with the same number of clicks "
            "as Accept All."
        ),
    ),
    "cookie.accept_visually_dominant": RuleExplanation(
        explanation=(
            "The Accept control is significantly larger, heavier, or more prominent than "
            "the Reject control based on measured size and font weight."
        ),
        user_impact=(
            "Visual hierarchy biases attention toward acceptance and away from refusal."
        ),
        gdpr_relevance=(
            "GDPR Art. 4(11) and Art. 7 require consent without undue influence; unequal "
            "visual prominence undermines free choice."
        ),
        recommendation=(
            "Match Accept and Reject button size, weight, and contrast so neither choice "
            "is visually privileged."
        ),
    ),
    "cookie.unequal_emphasis": RuleExplanation(
        explanation=(
            "Accept is rendered as a filled primary button while Reject is styled as a "
            "low-emphasis link or transparent control."
        ),
        user_impact=(
            "Users perceive Accept as the expected action and Reject as secondary or optional."
        ),
        gdpr_relevance=(
            "GDPR Art. 7 and transparency duties (Art. 12) require consent interfaces that "
            "do not nudge users toward one outcome."
        ),
        recommendation=(
            "Style Accept and Reject with equal emphasis (both buttons or both links of "
            "comparable weight)."
        ),
    ),
    "cookie.preselected_marketing": RuleExplanation(
        explanation=(
            "A non-essential marketing/advertising/personalisation category is enabled by "
            "default via checkbox or toggle."
        ),
        user_impact=(
            "Users may leave the page with marketing tracking enabled without affirmative action."
        ),
        gdpr_relevance=(
            "GDPR Art. 4(11) requires affirmative action; pre-ticked boxes do not constitute "
            "valid consent (Art. 7; CJEU Planet49)."
        ),
        recommendation=(
            "Leave all non-essential categories unchecked by default; require an explicit opt-in."
        ),
    ),
    "cookie.banner_obstruction": RuleExplanation(
        explanation=(
            "The cookie banner covers a large fraction of the viewport, obstructing page content "
            "until the user interacts with it."
        ),
        user_impact=(
            "Users feel pressured to dismiss the banner quickly, often by accepting, to regain "
            "access to the site."
        ),
        gdpr_relevance=(
            "Forced interaction and obstruction can undermine freely given consent under "
            "GDPR Art. 4(11) and Art. 7."
        ),
        recommendation=(
            "Reduce banner coverage so primary content remains usable, or allow dismiss without "
            "granting non-essential consent."
        ),
    ),
    "sub.auto_renew": RuleExplanation(
        explanation=(
            "Auto-renewal or recurring billing language is present, often de-emphasized "
            "relative to the primary call to action."
        ),
        user_impact=(
            "Users may enroll in a subscription without noticing ongoing charges will renew."
        ),
        gdpr_relevance=(
            "While primarily a consumer-protection issue, opaque renewal terms also conflict "
            "with transparency expectations under GDPR Art. 12 when personal data processing "
            "is tied to the subscription."
        ),
        recommendation=(
            "Disclose auto-renewal next to the CTA in body-size text; require an explicit opt-in."
        ),
    ),
    "sub.hidden_billing": RuleExplanation(
        explanation=(
            "Billing or recurring-charge details are hidden, below the fold, or visually "
            "suppressed relative to the primary CTA."
        ),
        user_impact=(
            "Users can complete checkout without seeing material cost and billing terms."
        ),
        gdpr_relevance=(
            "Lack of clear information conflicts with transparent processing communication "
            "(GDPR Art. 12) when payment data and account data are processed."
        ),
        recommendation=(
            "Place full billing terms adjacent to the CTA at readable font size and contrast."
        ),
    ),
    "sub.small_font_disclaimer": RuleExplanation(
        explanation=(
            "Material billing disclaimer text uses a font size much smaller than body text."
        ),
        user_impact=(
            "Important cost terms are easy to miss, increasing unintended subscriptions."
        ),
        gdpr_relevance=(
            "GDPR Art. 12 requires information to be provided in a concise, intelligible, "
            "and easily accessible form."
        ),
        recommendation=(
            "Use at least body-equivalent font size and contrast for billing disclaimers."
        ),
    ),
    "sub.misleading_free_trial": RuleExplanation(
        explanation=(
            "Free-trial messaging is highlighted while auto-renew or paid-conversion terms "
            "are also present on the same flow."
        ),
        user_impact=(
            "Users expect a no-cost trial and may overlook conversion to paid billing."
        ),
        gdpr_relevance=(
            "Misleading prominence of benefit language versus cost terms undermines informed "
            "consent and transparency (GDPR Art. 4(11), Art. 12)."
        ),
        recommendation=(
            "State trial length, price after trial, and renewal terms with equal prominence "
            "to the free-trial claim."
        ),
    ),
    "sub.preselected_plan": RuleExplanation(
        explanation=(
            "A higher-cost or longer-term subscription option is pre-selected without clear "
            "affirmative user choice."
        ),
        user_impact=(
            "Users may purchase a more expensive plan than they intended by default."
        ),
        gdpr_relevance=(
            "Pre-selected commercial options conflict with the affirmative-action standard "
            "for consent-related choices (GDPR Art. 4(11), Art. 7)."
        ),
        recommendation=(
            "Default to no plan selected, or to the lowest-cost option, and require an "
            "explicit selection."
        ),
    ),
    "sub.confirmshaming": RuleExplanation(
        explanation=(
            "The decline path uses guilt- or shame-inducing language rather than a neutral refusal."
        ),
        user_impact=(
            "Emotional pressure increases the likelihood of accepting unwanted offers or tracking."
        ),
        gdpr_relevance=(
            "Consent must be freely given without detriment or undue influence "
            "(GDPR Art. 4(11), Art. 7)."
        ),
        recommendation=(
            "Replace confirmshaming copy with a neutral decline label (e.g. 'No thanks')."
        ),
    ),
}


def explanation_for_rule(rule_id: str, *, fallback_evidence: str = "") -> RuleExplanation:
    """Return deterministic explanation for a rule_id; synthesize from evidence if unknown."""
    if rule_id in _RULE_EXPLANATIONS:
        return _RULE_EXPLANATIONS[rule_id]
    statement = fallback_evidence or f"Rule '{rule_id}' was triggered by the deterministic rule engine."
    return RuleExplanation(
        explanation=statement,
        user_impact=(
            "This pattern may impair the user's ability to make an informed, free choice "
            "about consent or commercial terms."
        ),
        gdpr_relevance=(
            "GDPR Art. 4(11), Art. 7, and Art. 12 require freely given, informed, and "
            "transparent consent and information."
        ),
        recommendation=(
            "Review the highlighted control and copy; ensure accept and refuse paths are "
            "equally available and clearly disclosed."
        ),
    )


def all_rule_explanations() -> dict[str, RuleExplanation]:
    return dict(_RULE_EXPLANATIONS)
