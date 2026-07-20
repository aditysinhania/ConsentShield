# Training

Training entrypoint stub:

```bash
PYTHONPATH=. python -m ai.training
```

## Planned jobs

1. **Vision** — fine-tune YOLOv11 / Grounding DINO on UI screenshots → `models/vision/`
2. **Text** — fine-tune RoBERTa / DeBERTa on consent language → `models/text/`
3. **Fusion** — train XGBoost/sklearn on multimodal feature vectors → `models/checkpoints/`

## Rules

Do not invent labels in stubs. Training code should write metrics to `research/results/` and checkpoints only under `models/`.

See also `docs/Research.md`.
