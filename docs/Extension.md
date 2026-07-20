# Chrome Extension

Manifest V3 extension under `apps/extension`.

## Capabilities

- Capture visible-tab screenshot
- Extract HTML (truncated), visible text, button/checkbox CSS features
- POST payload to FastAPI
- Render risk score, category, evidence in popup

## Build

```bash
pnpm --filter @consentshield/extension build
```

Load unpacked: `apps/extension/dist`

## Architecture

| Piece | Role |
|-------|------|
| `content/collect.ts` | DOM/CSS/text collection |
| `background/service-worker.ts` | Orchestrates capture + API call |
| `shared/inference.ts` | `RemoteApiProvider` / future `LocalOnDeviceProvider` |
| `popup/` | React UI |

AI does **not** run in the extension yet. All inference goes through the backend.
