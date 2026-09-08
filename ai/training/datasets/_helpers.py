"""Shared helpers for training datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UNIFIED_DIR = REPO_ROOT / "datasets" / "unified"

LABEL_FIELDS = ("label_binary", "label_fine", "label_consentshield", "label_vision")


def resolve_repo_path(path: str | Path | None, *, repo_root: Path | None = None) -> Path | None:
    if path is None or path == "":
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    root = repo_root or REPO_ROOT
    return (root / p).resolve()


def usable_text(sample: dict[str, Any]) -> str:
    """Prefer OCR, then ElementMap/visible text."""
    for key in ("ocr_text", "text"):
        val = sample.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def has_usable_text(sample: dict[str, Any]) -> bool:
    return bool(usable_text(sample))


def has_image(
    sample: dict[str, Any],
    *,
    repo_root: Path | None = None,
    require_exists: bool = True,
) -> bool:
    path = resolve_repo_path(sample.get("image_path"), repo_root=repo_root)
    if path is None:
        return False
    if not require_exists:
        return True
    return path.is_file()


def build_label_vocab(values: Iterable[str]) -> dict[str, int]:
    uniq = sorted({str(v) for v in values})
    return {label: idx for idx, label in enumerate(uniq)}


def encode_label(label: str, vocab: dict[str, int]) -> int:
    if label not in vocab:
        raise KeyError(f"Unknown label {label!r}; vocab has {len(vocab)} classes")
    return vocab[label]


def filter_samples(
    samples: Sequence[dict[str, Any]],
    *,
    sources: Sequence[str] | None = None,
    modalities: Sequence[str] | None = None,
    label_binary: Sequence[str] | None = None,
    label_fine: Sequence[str] | None = None,
    label_consentshield: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    src_set = set(sources) if sources else None
    mod_set = set(modalities) if modalities else None
    bin_set = set(label_binary) if label_binary else None
    fine_set = set(label_fine) if label_fine else None
    cs_set = set(label_consentshield) if label_consentshield else None

    for s in samples:
        if src_set is not None and s.get("source") not in src_set:
            continue
        if mod_set is not None and s.get("modality") not in mod_set:
            continue
        if bin_set is not None and s.get("label_binary") not in bin_set:
            continue
        if fine_set is not None and s.get("label_fine") not in fine_set:
            continue
        if cs_set is not None and s.get("label_consentshield") not in cs_set:
            continue
        out.append(s)
    return out
