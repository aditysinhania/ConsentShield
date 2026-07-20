# Datasets

## External sources (planned)

| Dataset | Use |
|---------|-----|
| B4E2 DarkPattern YOLO | Vision UI objects |
| Edinburgh Cookie Dialog | Cookie banners |
| Princeton Dark Patterns | Text language |
| EC-DarkPattern | Transformer baselines |
| CookieScanner | Cookie banner text |

Raw datasets belong under `research/datasets/` or `storage/datasets/` and are **gitignored**.

## Custom multimodal corpus

Target: 2,000–3,000 sites with:

- screenshot
- HTML / CSS
- OCR + visible text
- rule-engine outputs
- human-verified label

Schema: `ai/datasets/schemas/sample.py` (`MultimodalSample`).

Loader stub: `ai/datasets/loaders/jsonl.py`.
