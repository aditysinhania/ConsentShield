# ConsentShield Training Framework (Phase 1)

Production-ready training infrastructure for ConsentShield. **No model training runs in Phase 1.**

This package prepares reusable datasets, dataloaders, metrics, checkpoints, logging, YAML configs, and trainer scaffolds for:

- **Phase 2** — MiniLM fine-tuning (`TextTrainer` + `TextDataset`)
- **Phase 3** — CLIP fine-tuning (`VisionTrainer` + `ImageDataset`)
- **Later** — multimodal / fusion models (same `BaseTrainer` + `UnifiedDataset`)

---

## Directory structure

```
ai/training/
├── datasets/
│   ├── unified_dataset.py   # All modalities from train/val/test.jsonl
│   ├── text_dataset.py      # Usable text only (MiniLM)
│   ├── image_dataset.py     # Image paths only (CLIP)
│   └── collate.py           # Collate fns + DataLoader factory
├── trainers/
│   ├── base_trainer.py      # Shared fit / eval / checkpoint loop
│   ├── text_trainer.py      # MiniLM scaffold (Phase 2)
│   ├── vision_trainer.py    # CLIP scaffold (Phase 3)
│   └── utils.py             # Config merge, seed, optimizer, device
├── metrics/
│   ├── classification.py    # Accuracy, P/R/F1, macro/weighted
│   ├── confusion.py         # Confusion matrix
│   └── report.py            # JSON / Markdown / CSV writers
├── callbacks/
│   ├── early_stopping.py
│   ├── checkpoint.py        # best + latest + optimizer/scheduler/RNG
│   └── logging.py           # JSONL logs + GPU stats
├── configs/
│   ├── common.yaml
│   ├── text.yaml
│   └── vision.yaml
├── train.py                 # CLI (dry-run by default)
├── evaluate.py              # Evaluation pipeline
└── README.md
```

Run artifacts are written under:

```
ai/training/runs/<run_name>/
├── logs/
├── models/
└── reports/
```

---

## Training workflow (Phase 1)

1. Ensure unified JSONL exists: `datasets/unified/{train,val,test}.jsonl`
2. Validate the framework (no weights updated):

```bash
# From repo root, with AI extras installed (torch)
python -m ai.training.train --task text --dry-run
python -m ai.training.train --task vision --dry-run
```

3. Optional baseline evaluation (majority class — not a neural model):

```bash
python -m ai.training.evaluate --task text --baseline majority --split val
```

4. **Stop.** Do not pass `--allow-train` until Phase 2/3 is approved.

---

## Configuration

Hyperparameters live in YAML only. `load_config(task)` merges:

1. `configs/common.yaml`
2. `configs/text.yaml` or `configs/vision.yaml`
3. Optional `--config path/to/override.yaml`

Configurable fields include: batch size, epochs, learning rate, weight decay, optimizer, scheduler, device, seed, checkpoint names, logging directories, dataset filters (`sources`, label fields, etc.).

Example override:

```yaml
train:
  batch_size: 8
  learning_rate: 1.0e-5
data:
  sources: [huggingface_wipi, contextdp]
  label_field: label_binary
```

---

## Dataset classes

| Class | Purpose | Filters |
|-------|---------|---------|
| `UnifiedDataset` | Full JSONL rows | source, modality, binary/fine/ConsentShield labels |
| `TextDataset` | Samples with usable OCR or ElementMap text | same + `min_text_chars` |
| `ImageDataset` | Samples with image paths | same + `require_image_exists` |

Each `__getitem__` returns: `id`, text/OCR fields, `image_path`, encoded `label_id`, label strings, metadata.

Shared label vocab: build on **train**, pass `label_vocab=` to val/test.

---

## DataLoaders

`create_dataloader(...)` supports batch size, shuffle, `num_workers`, `pin_memory`, `persistent_workers`, and custom collate (`collate_text_batch` / `collate_image_batch`).

---

## Metrics & evaluation

- Accuracy, precision, recall, F1, macro F1, weighted F1
- Confusion matrix, per-class accuracy
- Binary and multiclass

`evaluate.py` writes `reports/{stem}.json`, `.md`, and `_predictions.csv`.

Checkpoint-based neural eval is stubbed until Phase 2.

---

## Checkpointing & logging

`CheckpointCallback` saves **best** and **latest** `.pt` files containing:

- model / optimizer / scheduler state
- epoch, metrics, config, label vocab, RNG state

`TrainingLogger` records train/val loss, metrics, epoch time, learning rate, and GPU memory (if CUDA).

---

## Phase 2 — MiniLM fine-tuning

```bash
# 1) Build filtered text views (HF + ContextDP, usable text only)
python -m datasets.preprocess.make_text_views

# 2) Fine-tune MiniLM (writes runs/minilm/)
python -m ai.training.train_minilm --run-name minilm
```

Artifacts: `runs/minilm/models/{best_model,last_model,training_state}.pt`, `training_config.yaml`, and reports under `runs/minilm/reports/`.

Load checkpoint via registry (does not replace scan inference by default):

```python
from ai.registry import ModelRegistry
reg = ModelRegistry.default()
reg.try_load_finetuned_minilm()  # or MINILM_CHECKPOINT=/path/to/best_model.pt
```

---

## How Phase 3 (CLIP) will use this

1. Use `configs/vision.yaml` + `ImageDataset` + `collate_image_batch`.
2. Implement `VisionTrainer.build_model` for CLIP (and image transform in collate or dataset).
3. Same `BaseTrainer.fit` loop, checkpoints, and metrics — no rewrite of the framework.

---

## Design rules

- Trainers never hardcode hyperparameters — read from config.
- Phase gates: `PhaseGateError` blocks accidental MiniLM/CLIP training in Phase 1.
- Fusion / multimodal models can subclass `BaseTrainer` and consume `UnifiedDataset` later.
