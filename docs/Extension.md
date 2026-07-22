# Chrome Extension (v1.0)

Manifest V3 extension under `apps/extension`.

Professional MVP popup + details + settings for ConsentShield scans. Compatible with the existing FastAPI backend — no inference runs inside the extension.

## Installation

```bash
pnpm install
pnpm --filter @consentshield/extension build
```

## Loading the extension

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. **Load unpacked** → select `apps/extension/dist`
4. Pin ConsentShield from the toolbar

## Running the backend

The extension expects the API at `http://localhost:8000` (changeable in Settings).

```powershell
# Windows
.\scripts\dev-api.ps1
```

```bash
# macOS / Linux
./scripts/dev-api.sh
```

Health check: `GET /health`

## AI models

AI (Sentence Transformer + CLIP) runs on the **backend**, not in the extension.

| Mode | Env | Behavior |
|------|-----|----------|
| Stub (default) | `AI_STUB_MODE=true` | Fast, no weights required |
| Pretrained | `AI_STUB_MODE=false`, `PHASE4_BACKEND=auto\|pretrained` | Real NLP/Vision after `pip install -e ".[ai]"` |

Rules remain the source of truth; fusion is rule-dominant.

## Architecture

```text
Popup / Details / Settings (React)
        │
        ▼
Background service worker  ── progress stages ──► Popup
        │
        ├── content script: DOM / CSS / CMP / iframe metadata
        ├── chrome.tabs.captureVisibleTab (optional screenshot)
        └── RemoteApiProvider → POST /api/v1/scan → GET /api/v1/report/:id
```

| Piece | Role |
|-------|------|
| `src/content/collect.ts` | DOM/CSS/text/CMP collection + iframe awareness |
| `src/background/service-worker.ts` | Orchestrates collect → screenshot → API + scan progress |
| `src/shared/*` | Settings, history, export, errors, explanations, annotations |
| `src/popup/` | Main results UI |
| `src/details/` | Collapsible full analysis page |
| `src/settings/` | Persisted preferences |

## Features (v1.0)

- Redesigned popup with circular risk score (green → yellow → orange → red)
- Light / dark / system themes
- Staged scan progress (DOM → rules → NLP → vision → fusion → report)
- Details page with annotated screenshot zoom
- Explainable findings labeled by detector (Rule Engine / ST / CLIP / Fusion)
- Cross-origin iframe limitation messaging (not “No Dark Pattern”)
- Scan history (last 25) with open / delete / clear
- Export PDF / HTML / Markdown / JSON (+ iframe limitation note)
- Settings: AI, screenshots, history, theme, export format, confidence threshold, auto-scan, advanced, developer mode
- Performance metrics from report payload
- CMP vendor display (Sourcepoint, OneTrust, Cookiebot, Didomi, TrustArc, Usercentrics, Unknown)
- Graceful error handling (backend / AI / screenshot / permission / iframe / timeout / network)
- Accessibility: keyboard focus, ARIA labels, `sr-only`, high-contrast media query

## Screenshots

After a scan, open **View details** → **Screenshot preview** for labeled bounding boxes (banner, accept, reject, settings, overlay, checkboxes) with confidence and zoom controls.

## Known limitations

- Cross-origin CMP iframes cannot be DOM-inspected by Chrome extensions
- Progress stages after DOM collect are optimistic while the API runs (backend does not stream stages)
- Annotation boxes use collected viewport coordinates; DPR / scroll edge cases may misalign slightly
- `chrome.storage.session` holds the last details payload until overwritten
- Pretrained AI requires backend packages and first-load model download time

## Supported browsers

- Google Chrome (primary, MV3)
- Chromium-based browsers that support MV3 unpacked extensions (Edge, Brave) — smoke-test recommended

## Performance

Typical local stub scan: a few seconds (DOM + network). Pretrained first call is slower (model load). Popup stays responsive via progress UI; API timeout is 120s.

## Roadmap

- Streamed backend progress events
- On-device inference provider stub → real
- Smarter annotation alignment (devicePixelRatio)
- Chrome Web Store packaging + icons set
