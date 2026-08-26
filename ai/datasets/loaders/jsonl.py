"""Dataset loaders — unified JSONL + legacy helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ai.datasets.schemas.sample import MultimodalSample, UnifiedSample


def load_jsonl(path: Path) -> Iterator[MultimodalSample]:
    import json

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if "label_binary" in raw:
                yield UnifiedSample.model_validate(raw).to_multimodal()
            else:
                yield MultimodalSample.model_validate(raw)


def load_unified_jsonl(path: Path) -> Iterator[UnifiedSample]:
    import json

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield UnifiedSample.model_validate(json.loads(line))
