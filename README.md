# ConsentShield

AI-assisted detection of **verifiable consent dark patterns** in cookie banners and related interfaces.

ConsentShield combines **DOM/CSS analysis**, a **configurable rule engine**, and optional **Sentence Transformer + CLIP** models into an explainable multimodal pipeline. Rules are the source of truth; AI is assistive. It intentionally does **not** claim to detect unverifiable patterns (e.g. fake scarcity).

**Current focus:** Chrome Extension **v1.0** MVP (professional UX). The web dashboard exists in-repo but is not required for extension demos.

## Monorepo layout

```text
ConsentShield/
├── apps/
│   ├── api/          # FastAPI backend
│   ├── extension/    # Chrome MV3 extension (v1.0)
│   └── web/          # React dashboard (optional)
├── ai/               # Rule engine + NLP/Vision/Fusion
├── packages/shared/  # Shared TypeScript contracts
├── models/           # Model weights (gitignored)
├── storage/          # Screenshots / reports (gitignored)
├── research/
├── database/
├── docker/
├── docs/
└── scripts/
```

## Quick start

### Prerequisites

- Node 20+, pnpm 9+
- Python 3.12+
- Docker (recommended for Postgres + Redis)

### 1. Environment

```bash
cp .env.example .env
```

### 2. Infrastructure

```bash
cd docker
docker compose up -d postgres redis
```

### 3. API (required for extension scans)

```powershell
# Windows
.\scripts\dev-api.ps1
```

```bash
# macOS / Linux
pip install -r apps/api/requirements.txt
./scripts/dev-api.sh
```

API docs: http://localhost:8000/docs

### 4. Chrome extension (v1.0)

```bash
pnpm install
pnpm --filter @consentshield/extension build
```

1. Open `chrome://extensions`
2. Enable **Developer mode**
3. **Load unpacked** → `apps/extension/dist`
4. Open a site with a cookie banner → click ConsentShield → **Analyze this page**

Full extension docs: [docs/Extension.md](docs/Extension.md)

### 5. AI models (optional)

Default: `AI_STUB_MODE=true` (no heavy deps).

For pretrained NLP/Vision:

```bash
pip install -e ".[ai]"
# set AI_STUB_MODE=false and PHASE4_BACKEND=auto|pretrained in .env
```

### 6. Web dashboard (optional — not part of extension v1.0)

```bash
pnpm --filter @consentshield/shared build
pnpm --filter @consentshield/web dev
```

## Pipeline

```text
Extension → DOM + CSS + text + screenshot
         → FastAPI /scan
         → RuleEngine
         → Sentence Transformer (NLP) / CLIP (Vision)  [or stubs]
         → FusionEngine (rule-dominant)
         → Explainable report → Popup / Details / Export
```

## Features (extension v1.0)

- Professional popup with circular risk score and light/dark theme
- Details page with collapsible sections and annotated screenshots
- Scan history, settings, multi-format export (PDF/HTML/MD/JSON)
- Clear messaging when analysis is limited by cross-origin iframes
- Staged loading UX and performance metrics display

## Known limitations

- Protected cross-origin consent iframes cannot be fully inspected
- AI quality depends on backend mode (stub vs pretrained)
- Annotation overlays are best-effort from DOM bounding boxes

## Supported browsers

Chrome (primary). Other Chromium MV3 browsers may work with unpacked load.

## Performance

Stub backend scans are typically a few seconds locally. First pretrained inference may take longer while models load.

## Documentation

| Doc | Topic |
|-----|--------|
| [docs/Architecture.md](docs/Architecture.md) | System design |
| [docs/API.md](docs/API.md) | REST endpoints |
| [docs/Extension.md](docs/Extension.md) | Chrome extension v1.0 |
| [docs/Dataset.md](docs/Dataset.md) | Datasets |
| [docs/Training.md](docs/Training.md) | Model training |
| [docs/Deployment.md](docs/Deployment.md) | Docker / deploy |
| [docs/Research.md](docs/Research.md) | Research notes |

## Roadmap

- Streamed scan progress from API
- Chrome Web Store packaging
- Improved annotation alignment
- On-device inference path

## License

Proprietary — final-year / research project. Update before public release.
