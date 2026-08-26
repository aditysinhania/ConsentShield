"""Text-only dataset for MiniLM / transformer fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ai.training.datasets._helpers import has_usable_text, usable_text
from ai.training.datasets.unified_dataset import UnifiedDataset


class TextDataset(UnifiedDataset):
    """
    Subset of UnifiedDataset with usable text (OCR or ElementMap text).

    Intended for Phase 2 MiniLM fine-tuning.
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
        min_text_chars: int = 1,
    ) -> None:
        self.min_text_chars = int(min_text_chars)
        super().__init__(
            split,
            data_dir=data_dir,
            path=path,
            label_field=label_field,
            label_vocab=label_vocab,
            sources=sources,
            modalities=modalities,
            label_binary=label_binary,
            label_fine=label_fine,
            label_consentshield=label_consentshield,
            repo_root=repo_root,
            max_samples=None,
        )
        self.samples = [
            s
            for s in self.samples
            if has_usable_text(s) and len(usable_text(s)) >= self.min_text_chars
        ]
        if max_samples is not None:
            self.samples = self.samples[: max(0, int(max_samples))]
        if label_vocab is None:
            from ai.training.datasets._helpers import build_label_vocab

            self.label_vocab = build_label_vocab(s[self.label_field] for s in self.samples)
            self.id2label = {i: lab for lab, i in self.label_vocab.items()}

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        item["input_text"] = item["usable_text"]
        return item
