"""Unified multimodal dataset over datasets/unified/*.jsonl."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional until `pip install -e ".[ai]"`

    class Dataset:  # type: ignore[no-redef]
        """Minimal stand-in when PyTorch is not installed."""

        def __getitem__(self, index: int):
            raise NotImplementedError

        def __len__(self) -> int:
            raise NotImplementedError


from ai.training.datasets._helpers import (
    DEFAULT_UNIFIED_DIR,
    LABEL_FIELDS,
    REPO_ROOT,
    build_label_vocab,
    encode_label,
    filter_samples,
    resolve_repo_path,
    usable_text,
)


class UnifiedDataset(Dataset):
    """
    Reads train/val/test JSONL from the unified corpus.

    Each item is a dict with text, OCR, image path, labels, and metadata,
    plus encoded ``label_id`` for the configured label field.
    """

    def __init__(
        self,
        split: str = "train",
        *,
        data_dir: str | Path | None = None,
        path: str | Path | None = None,
        label_field: str = "label_binary",
        label_vocab: dict[str, int] | None = None,
        sources: Sequence[str] | None = None,
        modalities: Sequence[str] | None = None,
        label_binary: Sequence[str] | None = None,
        label_fine: Sequence[str] | None = None,
        label_consentshield: Sequence[str] | None = None,
        repo_root: str | Path | None = None,
        max_samples: int | None = None,
    ) -> None:
        if label_field not in LABEL_FIELDS:
            raise ValueError(f"label_field must be one of {LABEL_FIELDS}, got {label_field!r}")

        self.repo_root = Path(repo_root) if repo_root else REPO_ROOT
        self.label_field = label_field
        self.split = split

        if path is not None:
            self.path = Path(path)
        else:
            root = Path(data_dir) if data_dir else DEFAULT_UNIFIED_DIR
            self.path = root / f"{split}.jsonl"

        if not self.path.is_file():
            raise FileNotFoundError(f"Unified split not found: {self.path}")

        raw = self._load_jsonl(self.path)
        raw = filter_samples(
            raw,
            sources=sources,
            modalities=modalities,
            label_binary=label_binary,
            label_fine=label_fine,
            label_consentshield=label_consentshield,
        )
        if max_samples is not None:
            raw = raw[: max(0, int(max_samples))]

        self.samples = raw
        self.label_vocab = label_vocab or build_label_vocab(s[label_field] for s in self.samples)
        self.id2label = {i: lab for lab, i in self.label_vocab.items()}

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        text = usable_text(sample)
        image_path = resolve_repo_path(sample.get("image_path"), repo_root=self.repo_root)
        label_str = str(sample[self.label_field])
        return {
            "id": sample["id"],
            "source": sample.get("source"),
            "modality": sample.get("modality"),
            "text": sample.get("text") or "",
            "ocr_text": sample.get("ocr_text") or "",
            "usable_text": text,
            "image_path": str(image_path) if image_path else None,
            "label": label_str,
            "label_id": encode_label(label_str, self.label_vocab),
            "label_binary": sample.get("label_binary"),
            "label_fine": sample.get("label_fine"),
            "label_consentshield": sample.get("label_consentshield"),
            "source_labels": sample.get("source_labels") or [],
            "bboxes": sample.get("bboxes") or [],
            "components": sample.get("components") or [],
            "metadata": sample.get("metadata") or {},
        }

    def class_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for s in self.samples:
            key = str(s[self.label_field])
            counts[key] = counts.get(key, 0) + 1
        return dict(sorted(counts.items()))

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "path": str(self.path),
            "num_samples": len(self),
            "label_field": self.label_field,
            "num_classes": len(self.label_vocab),
            "label_vocab": dict(self.label_vocab),
            "class_counts": self.class_counts(),
            "sources": sorted({str(s.get("source")) for s in self.samples}),
            "modalities": sorted({str(s.get("modality")) for s in self.samples}),
        }
