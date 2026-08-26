# ConsentShield training

Phase 1 training **framework** (no model fine-tuning yet):

```bash
# Validate datasets + dataloaders + config (default)
PYTHONPATH=. python -m ai.training.train --task text --dry-run
PYTHONPATH=. python -m ai.training.train --task vision --dry-run

# Majority-baseline evaluation reports (not a neural model)
PYTHONPATH=. python -m ai.training.evaluate --task text --baseline majority --split val
```

Full documentation: [`ai/training/README.md`](../ai/training/README.md)

## Planned phases

1. **Phase 1 (done)** — datasets, dataloaders, metrics, checkpoints, logging, YAML configs, trainer scaffolds
2. **Phase 2 (MiniLM)** — `python -m datasets.preprocess.make_text_views` then `python -m ai.training.train_minilm`
3. **Phase 3** — CLIP fine-tuning via `VisionTrainer` + `configs/vision.yaml`
4. **Later** — multimodal / fusion models on `UnifiedDataset`

Fine-tuned MiniLM checkpoints land in `runs/minilm/models/`. `ModelRegistry.try_load_finetuned_minilm()` loads them when present without changing the default scan inference path.

## Artifact layout

Training runs write under `ai/training/runs/<run_name>/{logs,models,reports}/`.

## Rules

- Do not invent labels.
- Hyperparameters must come from YAML configs, not hardcoded trainer values.
- Do not start MiniLM/CLIP training until Phase 2/3 is explicitly approved.

See also `docs/Research.md`.
