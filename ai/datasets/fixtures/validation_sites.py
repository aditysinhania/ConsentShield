"""Validation fixtures approximating Phase 2B site scans."""

from __future__ import annotations

from ai.common.types import ScanPayload


def _btn(
    text: str,
    *,
    width: int = 160,
    height: int = 40,
    font_weight: int = 700,
    bg: str = "rgb(0, 100, 0)",
    fg: str = "rgb(255, 255, 255)",
    font_size: int = 16,
    display: str = "block",
    visibility: str = "visible",
    opacity: float = 1.0,
    text_decoration: str = "none",
    aria: str | None = None,
) -> dict:
    return {
        "text": text,
        "ariaLabel": aria,
        "width": width,
        "height": height,
        "fontSizePx": font_size,
        "fontWeight": font_weight,
        "backgroundColor": bg,
        "color": fg,
        "display": display,
        "visibility": visibility,
        "opacity": opacity,
        "textDecoration": text_decoration,
        "xpath": f"/html/body/button[{text[:8]}]",
        "cssSelector": f"button.{text.lower().replace(' ', '-')[:20]}",
    }


def _banner(width: int, height: int, *, fixed: bool = True) -> dict:
    return {
        "width": width,
        "height": height,
        "xpath": "/html/body/div#cookie-banner",
        "cssSelector": "#cookie-banner",
        "computedStyles": {
            "position": "fixed" if fixed else "absolute",
            "zIndex": "9999",
            "opacity": 1,
        },
    }


def guardian_payload() -> ScanPayload:
    """Guardian-like: accept dominant, hidden reject path, moderate risk ~43."""
    return ScanPayload(
        url="https://www.theguardian.com/",
        title="The Guardian",
        visible_text="We use cookies. Accept all. Manage preferences.",
        viewport={"width": 1280, "height": 800},
        css_snapshot={
            "buttons": [
                _btn("Accept all", width=200, height=48, font_weight=700),
                _btn(
                    "Reject all",
                    width=80,
                    height=28,
                    font_weight=400,
                    bg="transparent",
                    fg="rgb(80, 80, 80)",
                    font_size=11,
                    text_decoration="underline",
                ),
                _btn("Manage preferences", width=140, height=32, font_weight=400, bg="transparent", fg="rgb(80,80,80)"),
            ],
            "banner": _banner(1200, 420),
            "cmp": {"name": "Sourcepoint", "detected": True},
        },
    )


def adobe_payload() -> ScanPayload:
    """Adobe-like: unequal emphasis + multi-click reject, risk ~41."""
    return ScanPayload(
        url="https://www.adobe.com/",
        title="Adobe",
        visible_text="We use cookies to improve your experience. Accept all cookies. Cookie settings.",
        viewport={"width": 1440, "height": 900},
        css_snapshot={
            "buttons": [
                _btn("Accept all cookies", width=190, height=44),
                _btn(
                    "Cookie settings",
                    width=130,
                    height=30,
                    font_weight=400,
                    bg="transparent",
                    fg="rgb(100,100,100)",
                    font_size=12,
                ),
            ],
            "banner": _banner(1300, 380),
            "cmp": {"name": "OneTrust", "detected": True, "version": "6.x"},
        },
    )


def spotify_payload() -> ScanPayload:
    """Spotify-like: minimal issues, risk ~6."""
    return ScanPayload(
        url="https://www.spotify.com/",
        title="Spotify",
        visible_text="We and our partners use cookies. Accept. Reject non-essential.",
        viewport={"width": 1280, "height": 800},
        css_snapshot={
            "buttons": [
                _btn("Accept", width=120, height=44),
                _btn("Reject non-essential", width=160, height=44, font_weight=600, bg="rgb(40,40,40)"),
            ],
            "banner": _banner(900, 180, fixed=False),
            "cmp": {"name": "Spotify CMP", "detected": True},
        },
    )


def mozilla_payload() -> ScanPayload:
    """Mozilla-like: settings-path friction only, risk ~26."""
    return ScanPayload(
        url="https://www.mozilla.org/",
        title="Mozilla",
        visible_text="Cookies help us improve. Accept All. Reject All. Cookie settings.",
        viewport={"width": 1280, "height": 800},
        css_snapshot={
            "buttons": [
                _btn("Accept All", width=150, height=42),
                _btn("Reject All", width=120, height=38, font_weight=500, bg="rgb(220,220,220)", fg="rgb(30,30,30)"),
                _btn(
                    "Cookie settings",
                    width=120,
                    height=36,
                    font_weight=500,
                    bg="transparent",
                    fg="rgb(0,96,223)",
                    font_size=14,
                ),
            ],
            "banner": _banner(1100, 260),
            "cmp": {"name": "Mozilla", "detected": True},
        },
    )


def reuters_payload() -> ScanPayload:
    """Reuters-like: hidden reject + visual dominance, moderate-high risk."""
    return ScanPayload(
        url="https://www.reuters.com/",
        title="Reuters",
        visible_text="We use cookies. Accept All.",
        viewport={"width": 1366, "height": 768},
        css_snapshot={
            "buttons": [
                _btn("Accept All", width=210, height=50, font_weight=800),
            ],
            "banner": _banner(1250, 400),
            "cmp": {"name": "TrustArc", "detected": True},
        },
    )


def cookielawinfo_payload() -> ScanPayload:
    """CookieLawInfo demo: preselected marketing + obstruction, risk ~36."""
    return ScanPayload(
        url="https://www.cookielawinfo.com/",
        title="CookieLawInfo",
        visible_text="This website uses cookies. Accept All. Reject All.",
        viewport={"width": 1280, "height": 800},
        css_snapshot={
            "buttons": [
                _btn("Accept All", width=140, height=40),
                _btn("Reject All", width=120, height=36, font_weight=500, bg="rgb(220,220,220)", fg="rgb(30,30,30)"),
            ],
            "banner": _banner(1200, 450),
            "checkboxes": [
                {"label": "Marketing cookies", "checked": True, "required": False, "name": "marketing"},
            ],
            "cmp": {"name": "CookieYes", "detected": True},
        },
    )
