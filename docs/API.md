# API

Base URL: `http://localhost:8000`

Prefix: `/api/v1`

Interactive docs: `/docs`

## Auth

| Method | Path | Description |
|--------|------|-------------|
| POST | `/auth/register` | Create user |
| POST | `/auth/login` | JWT access + refresh |

## Scan

| Method | Path | Description |
|--------|------|-------------|
| POST | `/scan` | Accept extension/dashboard payload, run pipeline |
| GET | `/scan` | List scans |
| GET | `/scan/{id}` | Scan metadata |

### Scan payload (JSON)

```json
{
  "url": "https://example.com",
  "title": "Example",
  "html": "<html>…</html>",
  "css_snapshot": { "buttons": [], "checkboxes": [] },
  "visible_text": "…",
  "screenshot_base64": "data:image/png;base64,…",
  "viewport": { "width": 1280, "height": 720 }
}
```

## Module results

| Method | Path |
|--------|------|
| GET | `/rules/catalog` |
| GET | `/rules/{scan_id}` |
| GET | `/vision/{scan_id}` |
| GET | `/text/{scan_id}` |
| GET | `/fusion/{scan_id}` |
| GET | `/report/{scan_id}` |
| GET | `/models/registry` |
| POST | `/feedback` |

## Report shape

```json
{
  "scan_id": "…",
  "risk_score": 72,
  "category": "Cookie Consent Manipulation",
  "confidence": 0.55,
  "evidence": [
    {
      "id": "rule:cookie.hidden_reject",
      "statement": "Accept control found but no visible Reject / Reject All control.",
      "severity": 0.9,
      "source": "rules"
    }
  ],
  "pipeline_notes": [
    "Vision model not loaded…",
    "Text model not loaded…"
  ]
}
```
