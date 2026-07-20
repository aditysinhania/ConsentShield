# Architecture

## Goals

- Detect **objectively verifiable** consent manipulation (cookie banners, subscriptions).
- Keep AI modules **independent and replaceable**.
- Always return **evidence**, never a bare "dark pattern found".
- Prefer modularity and research reproducibility over quick hacks.

## High-level diagram

```text
┌──────────────┐     ScanPayload      ┌─────────────────────┐
│  Extension   │ ───────────────────► │  FastAPI (apps/api) │
│  Web UI      │                      │  routers → services │
└──────────────┘ ◄── ExplainableReport└──────────┬──────────┘
                                                 │
                                      ┌──────────▼──────────┐
                                      │  ai/InferencePipeline│
                                      │  Rules · Vision ·    │
                                      │  Text · Fusion · XAI │
                                      └─────────────────────┘
```

## Module contracts

Every detector implements:

- `train()`
- `predict()`
- `evaluate()`
- `load_model()`
- `save_model()`

### RuleEngine

YAML-configured evaluators under `ai/rules/`. Fully operational.

### VisionDetector / TextClassifier

Stubs. Until models are loaded they report `not_loaded` and contribute pipeline notes only.

### FusionEngine

Rule-dominant aggregator. Learned fusion (XGBoost) reserved under `ai/fusion/classifier/`.

### ExplanationGenerator

Builds `ExplainableReport` with evidence items sourced from rules (and later vision/text).

## Backend layers

`routers → services → repositories/models → ai interfaces`

Routers stay thin. Scan orchestration lives in `app/services/scan_service.py`.

## Extension inference providers

`RemoteApiProvider` (current) and `LocalOnDeviceProvider` (future) share one interface so local inference can be added without redesign.

## Storage

- Postgres: users, scans, results, predictions, feedback
- Filesystem: `storage/screenshots`, `storage/reports`
- Models: `models/vision`, `models/text`, `models/checkpoints`
