# Unified Dataset — Preprocessing Report

- Generated (UTC): `2026-07-24T11:56:35.924859+00:00`
- Sources used: **ContextDP**, **B4E2**, **Hugging Face Deceptive Patterns**
- Sources ignored: **CookieScanner**

## Pipeline steps
1. Load raw samples from each source
2. Normalize labels → binary + fine + ConsentShield category
3. OCR screenshots (ContextDP + B4E2) via RapidOCR; HF uses ElementMap text
4. Deduplicate (image perceptual/file hash; text sha1)
5. Create train / val / test splits (B4E2 keeps original split; others stratified 80/10/10)

## Load counts (before dedupe)
- `contextdp`: 501
- `b4e2`: 1962
- `hf`: 11874
- `total`: 14337

## Deduplication
- Before: **14337**
- After: **14130**
- Dropped: `{"text_duplicate": 207}`

## OCR
- Cache hits: 333
- Cache misses / new OCR: 168
- OCR errors: 0
- Characters extracted: 95955

## Final corpus
- Total samples: **14130**
- With images: 2463
- With OCR text: 501

### By split
- **train**: 11325 (binary={'dark_pattern': 6461, 'no_dark_pattern': 4864})
- **val**: 1399 (binary={'dark_pattern': 792, 'no_dark_pattern': 607})
- **test**: 1406 (binary={'no_dark_pattern': 612, 'dark_pattern': 794})

### By source
- `b4e2`: 1962
- `contextdp`: 501
- `hf_manual`: 1153
- `hf_synthetic`: 10514

### Binary labels
- `dark_pattern`: 8047
- `no_dark_pattern`: 6083

### Fine labels
- `no_dark_pattern`: 6083
- `ui_component_context`: 1962
- `disguised_ads`: 1588
- `forced_action`: 1398
- `hidden_subscription`: 1027
- `interface_interference`: 866
- `scarcity_urgency`: 423
- `attention_distraction`: 292
- `default_choice`: 196
- `confirmshaming`: 159
- `hidden_costs`: 77
- `nagging`: 51
- `gamification`: 7
- `obstruction`: 1

### ConsentShield category mapping
- `No Dark Pattern`: 6083
- `Unknown`: 3671
- `Mixed Consent Manipulation`: 1878
- `Cookie Consent Manipulation`: 1567
- `Hidden Subscription`: 558
- `Confirmshaming`: 331
- `Hidden Billing`: 42

## Label normalization notes
- ContextDP categories mapped into unified fine taxonomy (scarcity_* → `scarcity_urgency`).
- B4E2 screenshots are treated as **dark_pattern** context with fine label `ui_component_context` (YOLO classes retained under `components` / bboxes).
- HF rows use MIT synthetic labels; page-level rows with mixed elements aggregate to dark_pattern if any element is deceptive.
- Scarcity/urgency patterns map to ConsentShield `Unknown` (project does not claim unverifiable scarcity as a primary product category).

## Outputs
```text
datasets/unified/
  train.jsonl
  val.jsonl
  test.jsonl
  label_map.json
  ocr_cache.json
  preprocessing_report.md
```

## Notes / warnings
- Image dedupe uses file signature (path+size+mtime). Pass `--phash` for perceptual hashing.
- OCR sources for this build: **contextdp** (all 501 screenshots OCR’d via RapidOCR).
- **B4E2** screenshots (1962) are included with labels/bboxes but OCR text is empty in this build (OneDrive I/O made full OCR very slow). Backfill with:

```bash
python datasets/preprocess/build_unified.py --workers 2 --ocr-sources contextdp,b4e2
```

  Cached ContextDP OCR is reused automatically.
