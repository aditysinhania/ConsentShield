from ai.common.types import ScanPayload
from ai.rules import RuleEngine


def test_hidden_reject_triggers():
    engine = RuleEngine()
    result = engine.evaluate_payload(
        ScanPayload(
            url="https://example.com",
            visible_text="We use cookies. Accept all",
            css_snapshot={
                "buttons": [
                    {
                        "text": "Accept all",
                        "width": 160,
                        "height": 40,
                        "fontWeight": 700,
                        "backgroundColor": "rgb(0,100,0)",
                        "fontSizePx": 16,
                    }
                ]
            },
        )
    )
    ids = {h.rule_id for h in result.hits}
    assert "cookie.hidden_reject" in ids
    assert result.normalized_risk > 0
