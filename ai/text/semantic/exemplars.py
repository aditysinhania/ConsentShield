"""Semantic exemplars for consent dark-pattern NLP matching."""

from __future__ import annotations

EXEMPLARS: dict[str, list[str]] = {
    "accept_button": [
        "Accept all cookies",
        "Yes I accept",
        "Allow all",
        "I agree to cookies",
        "Accept and continue",
        "Agree to all",
    ],
    "reject_button": [
        "Reject all cookies",
        "No thank you",
        "Decline all",
        "Only necessary cookies",
        "Essential only",
        "Reject non-essential",
    ],
    "cookie_settings": [
        "Manage cookies",
        "Cookie settings",
        "Customize cookie preferences",
        "More options",
    ],
    "privacy_settings": [
        "Privacy settings",
        "Manage privacy preferences",
        "Consent preferences",
    ],
    "preselected_consent": [
        "Marketing cookies are pre-selected",
        "Advertising category enabled by default",
        "Personalization already checked",
    ],
    "forced_action": [
        "You must accept cookies to continue",
        "Accept to use this website",
        "Consent is required to proceed",
    ],
    "confirmshaming": [
        "No thanks I do not want to save money",
        "I hate discounts",
        "Continue without protecting my privacy",
    ],
    "obstruction": [
        "Large cookie banner covering the page",
        "Consent dialog blocks the content",
        "Overlay prevents reading the article",
    ],
    "interface_interference": [
        "Accept button is much larger than reject",
        "Reject is a small grey link",
        "Unequal visual emphasis on consent choices",
    ],
}

PATTERN_TO_CATEGORY = {
    "accept_button": "Cookie Consent Manipulation",
    "reject_button": "Cookie Consent Manipulation",
    "cookie_settings": "Cookie Consent Manipulation",
    "privacy_settings": "Cookie Consent Manipulation",
    "preselected_consent": "Cookie Consent Manipulation",
    "forced_action": "Cookie Consent Manipulation",
    "confirmshaming": "Confirmshaming",
    "obstruction": "Cookie Consent Manipulation",
    "interface_interference": "Cookie Consent Manipulation",
}
