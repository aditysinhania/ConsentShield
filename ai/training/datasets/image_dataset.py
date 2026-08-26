"""Image-only dataset for CLIP / vision fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from ai.training.datasets._helpers import has_image
from ai.training.datasets.unified_dataset import UnifiedDataset


class ImageDataset(UnifiedDataset):
    """
    Subset of UnifiedDataset with resolvable image paths.

    Intended for Phase 3 CLIP fine-tuning.
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
        require_image_exists: bool = True,
    ) -> None:
        self.require_image_exists = bool(require_image_exists)
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
            if has_image(s, repo_root=self.repo_root, require_exists=self.require_image_exists)
        ]
        if max_samples is not None:
            self.samples = self.samples[: max(0, int(max_samples))]
        if label_vocab is None:
            from ai.training.datasets._helpers import build_label_vocab

            self.label_vocab = build_label_vocab(s[self.label_field] for s in self.samples)
            self.id2label = {i: lab for lab, i in self.label_vocab.items()}

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = super().__getitem__(index)
        # Vision trainers may load PIL/tensor later; path is enough for Phase 1.
        item["image"] = item["image_path"]
        return item
