"""Dataset loaders — placeholders for future training corpora."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from ai.datasets.schemas.sample import MultimodalSample


def load_jsonl(path: Path) -> Iterator[MultimodalSample]:
    import json

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield MultimodalSample.model_validate(json.loads(line))
