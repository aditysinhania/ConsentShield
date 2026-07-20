# ConsentShield

AI-powered detection of **verifiable consent dark patterns** in cookie banners and subscription interfaces.

ConsentShield combines **DOM/CSS analysis**, a **configurable rule engine**, and future **vision + NLP** models into an explainable multimodal pipeline. It intentionally does **not** claim to detect unverifiable patterns (e.g. fake scarcity).

## Monorepo layout

```text
ConsentShield/
├── apps/
│   ├── api/          # FastAPI backend
│   ├── extension/    # Chrome MV3 extension
│   └── web/          # React dashboard
├── ai/               # Independent ML modules (rules live; vision/text stubs)
├── packages/shared/  # Shared TypeScript contracts
├── models/           # Model weights (gitignored)
├── storage/          # Screenshots / reports (gitignored)
├── research/         # Papers, notebooks, benchmarks
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

### 3. API

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

### 4. Web dashboard

```bash
pnpm install
pnpm --filter @consentshield/shared build
pnpm --filter @consentshield/web dev
```

Dashboard: http://localhost:5173

### 5. Chrome extension

```bash
pnpm --filter @consentshield/extension build
```

Load `apps/extension/dist` as an unpacked extension in `chrome://extensions`.

## Pipeline

```text
Extension → DOM + CSS + text + screenshot
         → FastAPI /scan
         → RuleEngine (real)
         → VisionDetector (stub until weights)
         → TextClassifier (stub until checkpoint)
         → FusionEngine (rule-dominant)
         → ExplanationGenerator (evidence list)
         → Dashboard / popup
```

Vision and text modules return `status: not_loaded` — they **never invent** dark-pattern labels.

## Documentation

| Doc | Topic |
|-----|--------|
| [docs/Architecture.md](docs/Architecture.md) | System design |
| [docs/API.md](docs/API.md) | REST endpoints |
| [docs/Extension.md](docs/Extension.md) | Chrome extension |
| [docs/Dataset.md](docs/Dataset.md) | Datasets |
| [docs/Training.md](docs/Training.md) | Model training |
| [docs/Deployment.md](docs/Deployment.md) | Docker / deploy |
| [docs/Research.md](docs/Research.md) | Research notes |

## License

Proprietary — final-year / research project. Update before public release.
