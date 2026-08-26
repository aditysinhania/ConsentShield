"""Collate functions and DataLoader factory."""

from __future__ import annotations

from typing import Any, Callable, Sequence


def collate_identity(batch: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep samples as a list of dicts (useful for dry-runs / custom models)."""
    return list(batch)


def collate_text_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate text samples into lists + label tensor-ready ids."""
    return {
        "ids": [b["id"] for b in batch],
        "texts": [b.get("input_text") or b.get("usable_text") or "" for b in batch],
        "label_ids": [int(b["label_id"]) for b in batch],
        "labels": [b["label"] for b in batch],
        "sources": [b.get("source") for b in batch],
        "metadata": [b.get("metadata") or {} for b in batch],
    }


def collate_image_batch(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Collate image samples (paths); pixel tensors added in Phase 3."""
    return {
        "ids": [b["id"] for b in batch],
        "image_paths": [b.get("image_path") for b in batch],
        "texts": [b.get("usable_text") or "" for b in batch],
        "label_ids": [int(b["label_id"]) for b in batch],
        "labels": [b["label"] for b in batch],
        "sources": [b.get("source") for b in batch],
        "metadata": [b.get("metadata") or {} for b in batch],
    }


def create_dataloader(
    dataset: Any,
    *,
    batch_size: int = 16,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    drop_last: bool = False,
    collate_fn: Callable | None = None,
):
    """Build a reusable PyTorch DataLoader with production-friendly defaults."""
    try:
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            'PyTorch is required for DataLoaders. Install with: pip install -e ".[ai]"'
        ) from exc

    if num_workers <= 0:
        persistent_workers = False
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        drop_last=drop_last,
        collate_fn=collate_fn or collate_identity,
    )
