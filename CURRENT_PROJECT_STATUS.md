# ConsentShield — Current Project Status

**Generated:** 2026-08-26 (inspection-only audit)  
**Source of truth:** On-disk repository at `ConsentShield/`  
**Git HEAD:** `8eeac7c` on `main` (1 commit ahead of `origin/main`)

---

## SECTION 1 — Executive Summary

| Item | Status |
|------|--------|
| **Overall completion** | **~72%** of product MVP scope (extension + backend pipeline shipped; website / auth UX / full model training incomplete) |
| **Project version** | Python package `0.1.0` (`pyproject.toml`); Chrome Extension **`1.0.0`**; AI pipeline version string **`0.5.0`** (Phase 3.5 tests) |
| **Current phase** | **Phase 2 MiniLM fine-tuning — interrupted** (framework done; **2/5 epochs** completed). Next product phases after that: CLIP training, fusion learning, website polish |
| **Last major completed milestone** | **Chrome Extension v1.0.0** + Phase 3.0–3.5 backend (explainability → severity/a11y → timeline/annotation → reports → narrator → production hardening). Datasets unified. Training **framework** (Phase 1) implemented on disk (mostly **uncommitted**) |
| **Chrome Extension production-ready?** | **Yes for local MVP demos** (MV3, polished UX). Not Chrome Web Store–packaged; requires local FastAPI |
| **Backend production-ready?** | **Near-ready for local/dev**. Full stack works with stubs. Postgres/Redis/auth exist but not required for extension scan path. Default `AI_STUB_MODE=true` |
| **Website** | **Exists as scaffold** under `apps/web/` (`0.1.0`) — optional, **not** part of extension v1.0. No real login/signup pages |

**Bottom line:** The shippable demo path is **Extension ↔ FastAPI ↔ Rule Engine (+ optional pretrained ST/CLIP)**. Learned MiniLM checkpoint exists but is **not** wired into scan inference by default. CLIP/fusion **training** has not started. Large dataset/training artifacts are mostly uncommitted / gitignored.

---

## SECTION 2 — Architecture Audit

```text
Chrome Extension (MV3)
  collect DOM/CSS/text/CMP/iframe + optional screenshot
        ↓
FastAPI  POST /api/v1/scan
        ↓
InferencePipeline (ai/inference/pipeline.py)
        ↓
RuleEngine  ←── source of truth
        ↓
TextClassifier (MiniLM embeddings + exemplars OR stub)
        ↓
VisionDetector (CLIP zero-shot OR DOM lexical fallback OR stub)
        ↓
FusionEngine (rule-dominant aggregator; learned FusionClassifier stub)
        ↓
ExplanationGenerator + enrich_report + severity + a11y + timeline
        ↓
Report document / exporters (PDF, HTML, MD, JSON)
        ↓
Extension popup / details / history / export
```

| Module | Path | Implemented? |
|--------|------|--------------|
| Extension collect + API client | `apps/extension/` | **Yes** |
| FastAPI scan/report/export | `apps/api/` | **Yes** |
| Rule Engine | `ai/rules/` | **Yes** |
| Text / NLP | `ai/text/` | **Yes** (pretrained ST or stub) |
| Vision / CLIP | `ai/vision/` | **Yes** (CLIP or lexical or stub) |
| Fusion aggregator | `ai/fusion/aggregator/` | **Yes** (rules-dominant) |
| Learned fusion classifier | `ai/fusion/classifier/model.py` | **Stub only** (`not_loaded`) |
| Explanation | `ai/explanation/` | **Yes** |
| Evidence / quality | `ai/evidence/` | **Yes** |
| Severity | `ai/severity/` | **Yes** |
| Accessibility | `ai/accessibility/` | **Yes** |
| Timeline | `ai/timeline/` | **Yes** |
| Screenshot annotator | `ai/screenshots/` | **Yes** |
| Report / exporters | `ai/report/` | **Yes** |
| Narrator (LLM optional) | `ai/narrator/` | **Yes** (offline fallback default) |
| Performance metrics | `ai/performance/` | **Yes** |
| Model Registry | `ai/registry/` | **Yes** (+ optional fine-tuned MiniLM slot) |
| Training framework | `ai/training/` | **Yes** (uncommitted) |
| Web dashboard | `apps/web/` | **Partial** |

---

## SECTION 3 — Chrome Extension Audit

**Location:** `apps/extension/`  
**Version:** `1.0.0`  
**Manifest:** MV3 (`apps/extension/public/manifest.json`)

| Feature | Status | Notes |
|---------|--------|-------|
| Popup UI | **Completed** | `popup/PopupApp.tsx`, circular `ScoreRing`, staged `ProgressSteps` |
| Details page | **Completed** | `details/DetailsApp.tsx` — collapsible sections, findings, exports |
| Settings page | **Completed** | `settings/SettingsApp.tsx` — API URL, AI, theme, history, export, thresholds |
| History page | **Completed** | In-popup / shared history (`shared/history.ts`), last 25 scans |
| Theme support | **Completed** | Light / dark / system (`shared/theme.ts`) |
| Export | **Completed** | PDF / HTML / MD / JSON (`shared/export.ts`) |
| Annotated screenshots | **Completed** | `AnnotatedScreenshot.tsx` + `shared/annotations.ts` |
| Performance metrics | **Completed** | Shown from report `performance` payload |
| Error handling | **Completed** | `shared/errors.ts` |
| CMP detection | **Completed** | Rich signatures in `content/collect.ts` (Sourcepoint, OneTrust, Cookiebot, …) |
| Iframe limitation handling | **Completed** | Explicit messaging via `iframeLimitationCopy` — not “No Dark Pattern” |
| Accessibility support | **Completed** | Roles/labels in UI; backend a11y analysis consumed in details |
| Dark mode | **Completed** | Via theme |
| Version number | **Completed** | `1.0.0` |
| Manifest version | **Completed** | Manifest V3 |
| Background service worker | **Completed** | `background/service-worker.ts` |
| Content script | **Completed** | `content/collect.ts` |
| Popup ↔ backend | **Completed** | Scan + report fetch through background / shared inference |

**Verdict:** Extension v1.0 feature set is **complete for MVP**. Remaining gaps are packaging (Web Store), E2E automated extension tests, and hosting a non-localhost API.

---

## SECTION 4 — Backend Audit

**Location:** `apps/api/`  
**Router mount:** `app/api/v1/router.py`

| Area | Status | Notes |
|------|--------|-------|
| API routes | **Completed** | `/auth`, `/scan`, `/vision`, `/text`, `/rules`, `/fusion`, `/report`, `/models`, `/feedback` |
| Scan endpoint | **Completed** | `routers/scan.py` + `scan_service.py` → `InferencePipeline` |
| Report endpoint | **Completed** | `routers/report.py` |
| Export endpoint | **Completed** | Report exporters (PDF/HTML/MD/JSON) |
| Screenshot annotation | **Completed** | `ai/screenshots/annotator.py` |
| Timeline | **Completed** | `ai/timeline/` wired in pipeline |
| Evidence enrichment | **Completed** | `ai/evidence/`, `ai/report/enricher.py` |
| Severity model | **Completed** | `ai/severity/` |
| Accessibility analyzer | **Completed** | `ai/accessibility/` |
| Performance metrics | **Completed** | `ai/performance/` |
| Storage system | **Completed** | `./storage` screenshots/reports (gitignored) |
| Database models | **Completed** | `User`, scan/detection/feedback models + Alembic `0001_initial` |
| Authentication support | **Partial** | Register/login JWT implemented; **extension scan path does not require auth** |
| Website API readiness | **Partial** | CORS includes `localhost:5173`; web UI mostly demo shells |

**Default AI mode (`.env.example`):** `AI_STUB_MODE=true` → stubs for text/vision/embedding when `PHASE4_BACKEND=auto`.

---

## SECTION 5 — AI Pipeline Audit

| Component | Status |
|-----------|--------|
| Rule Engine | **Implemented** — primary detector |
| Vision detector | **Implemented** — CLIP zero-shot + DOM lexical fallback |
| Text classifier | **Implemented** — SentenceTransformer exemplars + lexical fallback |
| Fusion engine | **Implemented** — rule-dominant aggregation |
| Confidence scoring | **Implemented** |
| Explanation generator | **Implemented** |
| Model Registry | **Implemented** — DI adapters + optional `finetuned_minilm` |
| CLIP integration | **Implemented for inference** (not trained/fine-tuned) |
| MiniLM integration | **Inference:** pretrained ST embeddings. **Training:** fine-tuned classifier checkpoint exists but **does not replace** scan text path by default |
| Stub mode | **Default on** |
| Pretrained inference | **Available** when `AI_STUB_MODE=false` + `pip install -e ".[ai]"` |
| Checkpoint loading | **Available** via `ModelRegistry.try_load_finetuned_minilm()` / `MINILM_CHECKPOINT` |

### What inference uses today (default config)

**Rule Engine + stubs** (when `AI_STUB_MODE=true` and `PHASE4_BACKEND=auto`).

When stubs are off:

**Rule Engine + pretrained MiniLM (exemplar matching) + CLIP (or lexical vision fallback) + rule-dominant fusion.**

Not used in the live scan path by default:

- Fine-tuned MiniLM classification head (`runs/minilm/models/best.pt`)
- Learned fusion / XGBoost head (`FusionClassifier` stays `not_loaded`)

---

## SECTION 6 — Dataset Audit

**Root:** `datasets/`

| Corpus | Status | Notes |
|--------|--------|-------|
| ContextDP | **Present** | ~501 samples; OCR completed in unified build |
| B4E2 | **Present** | ~1962; labels/bboxes; **OCR empty** in unified build |
| HF Deceptive Patterns | **Present** | `hf_synthetic` + `hf_manual` |
| CookieScanner | **Not downloaded** | `DOWNLOAD_REQUIRED.txt` only (~6.4 GB archive); **ignored** by unified builder |
| Unified dataset | **Present** | `train/val/test.jsonl`, `label_map.json`, `ocr_cache.json`, `preprocessing_report.md`, `sample_report.html` |
| Training text views | **Present** | `datasets/training/text_{train,val,test}.jsonl` + stats |

### Unified corpus (from `preprocessing_report.md`)

| Metric | Value |
|--------|-------|
| Raw before dedupe | 14,337 |
| After dedupe | **14,130** (207 text dupes dropped) |
| Train / Val / Test | 11,325 / 1,399 / 1,406 |
| With images | 2,463 |
| With OCR text | **501** (ContextDP only in this build) |
| Binary DP / no-DP | 8,047 / 6,083 |
| ConsentShield `Unknown` | 3,671 |
| Sources in map | `b4e2`, `contextdp`, `hf_manual`, `hf_synthetic` |
| Ignored | `cookiescanner` |

### MiniLM text views (`datasets/training/`)

| Split | Samples | Notes |
|-------|---------|-------|
| train | 9,731 | Balanced binary; HF + ContextDP usable text |
| val | 1,214 | Balanced |
| test | 1,223 | ~Balanced |

B4E2 excluded from text views (no usable OCR/text).

**Git note:** `datasets/raw/**` and most of `datasets/unified/**` are gitignored; reports/`label_map` allowed.

---

## SECTION 7 — Model Training Audit

### Phase 1 — Training framework

| Item | Status |
|------|--------|
| Phase 1 completed? | **Yes (on disk)** |
| Framework under `ai/training/` | **Yes** |
| Dataset classes | **Yes** — `UnifiedDataset`, `TextDataset`, `ImageDataset` |
| DataLoaders / collate | **Yes** |
| Callbacks | **Yes** — early stopping, checkpoint, logging |
| Metrics | **Yes** — classification, confusion, reports |
| Logging | **Yes** — JSONL under `runs/<run>/logs/` |
| Checkpointing | **Yes** — atomic `.tmp` replace |
| Configs | **Yes** — `common.yaml`, `text.yaml`, `vision.yaml` |
| Evaluation scripts | **Yes** — `evaluate.py`, `train.py` dry-run |
| Docs | **Yes** — `ai/training/README.md`, `docs/Training.md` |

**Note:** Most of `ai/training/` is **untracked** in git (not in the last committed snapshot).

### Phase 2 — MiniLM fine-tuning

| Question | Answer |
|----------|--------|
| Did training start? | **Yes** |
| Did training finish (5 epochs)? | **No — stopped mid epoch 3** |
| Best checkpoint exists? | **Yes** — `runs/minilm/models/best.pt` (~259 MB), epoch **2**, best val F1 **0.7928** |
| Latest checkpoint? | **Yes** — `runs/minilm/models/latest.pt` (also epoch 2) |
| Training logs? | **Yes** — several runs under `runs/minilm/logs/` |
| Metrics / confusion matrix? | **Yes** — `runs/minilm/reports/` (val/test evaluation from **epoch-1 weights**; curves CSV has epochs 1–2) |
| `best_model.pt` alias? | **Missing now** (summary still references it; only `best.pt` / `latest.pt` present) |

### Where training stopped

1. Epoch 1 completed (CPU) — val acc ~0.736, val F1 ~0.708; first crash risk: low disk / ACCESS_VIOLATION on OneDrive.  
2. Resumed → **Epoch 2 completed** — val acc **0.783**, val F1 **0.793**, val loss 0.485 (~3.4 h/epoch on CPU).  
3. Epoch 3 **started** (`minilm_20260814T091038Z.jsonl`: `epoch_start` only) — **no `epoch_end`**. Process interrupted (session/shell drop).  
4. Planned epochs: **5**. Remaining: **3, 4, 5**.

### Reported test metrics (from epoch-1 evaluation artifacts)

| Metric | Val (epoch 1 eval) | Test (epoch 1 eval) |
|--------|--------------------|---------------------|
| Accuracy | 0.736 | **0.738** |
| Macro F1 | 0.733 | **0.736** |
| F1 (binary pos) | 0.708 | 0.716 |

**Caveat:** Epoch-2 checkpoint is **better on validation** than the stored test evaluation files (those were not re-run after epoch 2). Resume command:  
`python -m ai.training.train_minilm --run-name minilm --epochs 5 --resume auto`

### CLIP / fusion training

**Not started.** `VisionTrainer` remains Phase-3 gated scaffold.

---

## SECTION 8 — Models Folder Audit

**Path:** `models/` (gitignored)

| Item | Status |
|------|--------|
| Structure | `cache/`, `checkpoints/`, `text/`, `vision/`, `README.md` |
| Downloaded pretrained weights in-repo | **Empty / cache-driven** — HF downloads go to user HF cache when models load |
| CLIP cache in-repo | **Not populated as committed files** |
| MiniLM cache in-repo | Same |
| Fine-tuned checkpoint | Under **`runs/minilm/models/`** (~271 MB `.pt` files), not `models/text/` |
| HF / torch caches | Gitignored (`.cache/`, `hf_cache/`, `torch_cache/`) |
| `*.pt` | Globally gitignored |

---

## SECTION 9 — Testing Audit

**Location:** `apps/api/tests/` (pytest `testpaths`)

| Suite | File | Focus |
|-------|------|-------|
| Rule engine | `test_rule_engine.py` | Rules |
| Phase 2B / collection | `test_collection_robustness.py` | Robust collection |
| Phase 3 explainability | `test_phase3_explanation.py` | Explanations |
| Phase 3.1 | `test_phase31_validation.py` | Severity / a11y validation |
| Phase 3.2 | `test_phase32_timeline_metrics.py` | Timeline / metrics |
| Phase 3.3 | `test_phase33_reports.py` | Reports / exports |
| Phase 3.4 | `test_phase34_narrator.py` | Narrator |
| Phase 3.5 | `test_phase35_regression.py` | Site risk bands + DI registry |
| Phase 4 | `test_phase4_models.py` | Stub vs assisted AI bands |

| Area | Status |
|------|--------|
| Training unit tests | **Missing** |
| Extension automated tests | **Missing** |
| Latest CI result in-repo | **No committed pytest log found** in this audit |

**Regression risk bands (fixture payloads, Phase 3.5):**

| Site | Expected risk band |
|------|--------------------|
| Guardian | 30–55 |
| Adobe | 28–55 |
| Spotify | 0–25 |
| Mozilla | 12–40 |
| Reuters | 20–55 |
| BBC | 20–55 |

Phase 4 adds CookieLawInfo **20–55** and slightly wider upper bounds when AI assist enabled.

**Note:** `test_model_registry_di_interfaces` asserts `len(desc) == 4`, but registry `describe()` now also emits **FineTunedMiniLM** → may be **5** rows if that change is loaded (uncommitted). Worth fixing before CI.

---

## SECTION 10 — Live Website Validation Audit

**Important:** This audit found **fixture-based** validation (`ai/datasets/fixtures/validation_sites.py` + pytest), **not** live browser captures stored in-repo from Guardian/BBC/etc.

| Site | Fixture risk band | Confidence | Rules | Known issues |
|------|-------------------|------------|-------|--------------|
| Guardian | 30–55 | From report confidence fields in pipeline | Sourcepoint-like Yes/No dominance | Cross-origin SP iframes limit live DOM |
| BBC | 20–55 | Same | Accept-dominant + settings path | Iframe / CMP nesting |
| Reuters | 20–55 | Same | Hidden reject + dominance | Same |
| Adobe | 28–55 | Same | Unequal emphasis + multi-click reject | Same |
| Mozilla | 12–40 | Same | Settings friction primarily | Lower risk expected |
| Spotify | 0–25 | Same | Minimal issues | Should stay low |
| CookieLawInfo | 20–55 (Phase 4) | Same | Preselected marketing + obstruction | Demo CMP patterns |

| Topic | Finding |
|-------|---------|
| Iframe limitations | **Documented and handled in extension** — analysis limited message, not false “clean” |
| Do pretrained models improve results? | **Designed to assist slightly** within rule-dominant fusion (± small boost); stubs vs pretrained covered in Phase 4 tests. Fine-tuned MiniLM **not** in default scan path |
| Live production crawls | **No dated live score dump** found in-repo for this audit |

---

## SECTION 11 — Git & Version Audit

| Item | Value |
|------|-------|
| Branch | `main` |
| HEAD | `8eeac7c` — *Clean up storage paths and ignore runtime artifacts* |
| vs origin | **Ahead by 1** commit |
| Extension v1.0 | `455a57f` — *Release ConsentShield Chrome Extension v1.0.0* |
| Phase 3.5 | `bfa4926` |
| Phase 3.4 | `a6eec63` |
| Phase 3.3 | `fe87b1f` |
| Phase 3.2 | `736e4f3` |
| Phase 3.1 | `1dc4193` |
| Phase 3.0 | `d9e6107` |
| Phase 2B | `372ed80` |

### Uncommitted / untracked (significant)

**Modified:** `.gitignore`, `ai/datasets/*`, `ai/registry/model_registry.py`, `ai/training/__init__.py`, `ai/training/__main__.py`, `docs/Training.md`

**Untracked:** full `ai/training/{callbacks,configs,datasets,metrics,trainers,...}`, `train_minilm.py`, `evaluate.py`, `checkpoint_loader.py`, `datasets/` tree, `runs/`

**Gitignore pitfall:** root rule `models/` may also hide `ai/training/models/` (where `minilm_classifier.py` lives). Confirm with `git check-ignore -v ai/training/models/minilm_classifier.py` before committing — you may need `!ai/training/models/` exceptions.

### Should not be committed

- `runs/minilm/models/*.pt` (~270 MB each) — already covered by `*.pt` gitignore  
- `datasets/unified/*.jsonl`, `ocr_cache.json` — gitignored  
- `datasets/raw/**` large blobs — gitignored  
- HF/torch caches, `.env`, `storage/`

---

## SECTION 12 — Website Audit

**Location:** `apps/web/` version **0.1.0**

| Feature | Status |
|---------|--------|
| React + Vite app | **Completed** (scaffold) |
| Landing / Home | **Partial** — `HomePage.tsx` |
| Login page | **Missing** |
| Signup page | **Missing** |
| Dashboard | **Partial** — shell via `AppShell` + home/analytics |
| Analyze URL page | **Partial** — `ScanPage.tsx` |
| History page | **Partial** — `HistoryPage.tsx` |
| Reports / explanation | **Partial** — `ExplanationPage.tsx` |
| Settings | **Partial** — mentions API auth token |
| Authentication UX | **Missing** (API auth exists; web not integrated) |
| Backend integration | **Partial** — `api/client.ts` |

**Status:** Optional companion UI; **not production**. Extension is the primary client.

---

## SECTION 13 — Remaining Roadmap

| Feature | Status | Priority | Est. effort | Dependencies |
|---------|--------|----------|-------------|--------------|
| Commit training framework (no weights) | Pending | High | 0.5 d | Clean gitignore for `runs/` / JSONL |
| Finish MiniLM epochs 3–5 + re-eval test | Interrupted | High | 1–2 d CPU / hours GPU | Disk space; prefer non-OneDrive path |
| Wire fine-tuned MiniLM into assistive text path (optional flag) | Not done | Medium | 1–2 d | Finished checkpoint; keep rules dominant |
| CLIP fine-tuning (Phase 3 vision) | Not started | Medium | 3–5 d | Images + GPU; VisionTrainer |
| B4E2 OCR backfill | Incomplete | Medium | 1–2 d | RapidOCR; disk I/O |
| CookieScanner download | Skipped | Low | Manual | ≥15 GB free |
| Learned fusion classifier | Stub | Medium | 3–5 d | Text+vision features; labels |
| Website auth + real dashboard | Scaffold | Medium | 1–2 wk | API auth, design |
| Extension Web Store package | Not done | Medium | 2–3 d | Privacy policy, review assets |
| Docker full deploy | Partial (`docker/`) | Medium | 2–3 d | Compose already present |
| CI/CD (pytest + build) | Minimal / local | High | 1–2 d | GitHub Actions |
| Automated extension tests | Missing | Low | 2–3 d | Playwright/Puppeteer |
| Model evaluation harness vs live sites | Fixtures only | Medium | 2–3 d | Capture protocol |

---

## SECTION 14 — Immediate Next Steps

### HIGH PRIORITY

1. **Resume MiniLM to complete epochs 3–5**  
   - **Why:** Best checkpoint is epoch 2; training plan was 5 epochs; epoch 3 already started once and died.  
   - **Files:** `ai/training/train_minilm.py`, `runs/minilm/models/best.pt`, configs under `ai/training/configs/`  
   - **Cursor can automate?** Yes (long-running; needs free disk, prefer local SSD over OneDrive).

2. **Re-run test evaluation on current `best.pt` and refresh summary reports**  
   - **Why:** Stored test metrics are from epoch 1; epoch 2 val improved materially.  
   - **Files:** `ai/training/evaluate.py`, `runs/minilm/reports/`  
   - **Cursor can automate?** Yes.

3. **Commit training framework source (exclude `.pt` / large JSONL)**  
   - **Why:** Phase 1–2 code is only on disk as untracked files — risk of loss.  
   - **Files:** `ai/training/**`, `docs/Training.md`, `.gitignore`  
   - **Cursor can automate?** Yes (on explicit commit request).

4. **Fix registry `describe()` vs Phase 3.5 test expecting 4 interfaces**  
   - **Why:** Uncommitted FineTunedMiniLM row may break `test_model_registry_di_interfaces`.  
   - **Files:** `ai/registry/model_registry.py`, `apps/api/tests/test_phase35_regression.py`  
   - **Cursor can automate?** Yes.

### MEDIUM PRIORITY

5. **Optional: assistive use of fine-tuned MiniLM behind env flag** (do not replace rules).  
6. **B4E2 OCR backfill** for vision/text multimodal quality.  
7. **Add GitHub Actions:** pytest + extension/web typecheck.  
8. **Website:** login/signup pages + token wiring to existing `/auth`.  
9. **Plan CLIP Phase 3 training** using `ImageDataset` + `VisionTrainer`.

### LOW PRIORITY

10. CookieScanner download when disk allows.  
11. Chrome Web Store listing assets.  
12. Learned fusion (XGBoost) training.  
13. Extension E2E tests.

---

## Appendix — Key paths quick reference

```text
apps/extension/     Chrome MV3 v1.0.0
apps/api/           FastAPI + tests
apps/web/           React dashboard 0.1.0 (scaffold)
ai/inference/       End-to-end pipeline
ai/rules/           Rule engine (source of truth)
ai/training/        Training framework + MiniLM trainer (mostly uncommitted)
datasets/unified/   14,130-sample corpus
datasets/training/  MiniLM text views
runs/minilm/        Checkpoints + reports (epoch 2 best)
models/             Empty cache dirs (gitignored)
docker/             compose + Dockerfiles
docs/               Architecture, Extension, Training, API, …
```

---

*End of audit. No source code was modified for this report except creation of this file.*
