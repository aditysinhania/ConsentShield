"""Build filtered MiniLM training views under datasets/training/."""

from __future__ import annotations

import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from ai.training.datasets._helpers import REPO_ROOT, has_usable_text, usable_text

ALLOWED_SOURCES = ("hf_synthetic", "hf_manual", "contextdp")
UNIFIED_DIR = REPO_ROOT / "datasets" / "unified"
OUT_DIR = REPO_ROOT / "datasets" / "training"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def filter_text_samples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep HF + ContextDP samples with usable text; drop empty B4E2/etc."""
    out: list[dict[str, Any]] = []
    for row in rows:
        if row.get("source") not in ALLOWED_SOURCES:
            continue
        if not has_usable_text(row):
            continue
        # Normalize a dedicated training field without mutating schema unexpectedly
        enriched = dict(row)
        enriched["usable_text"] = usable_text(row)
        out.append(enriched)
    return out


def split_stats(name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [len(usable_text(r)) for r in rows]
    return {
        "split": name,
        "num_samples": len(rows),
        "class_distribution": dict(Counter(str(r.get("label_binary")) for r in rows)),
        "source_distribution": dict(Counter(str(r.get("source")) for r in rows)),
        "avg_text_length": round(statistics.mean(lengths), 2) if lengths else 0.0,
        "median_text_length": float(statistics.median(lengths)) if lengths else 0.0,
        "min_text_length": min(lengths) if lengths else 0,
        "max_text_length": max(lengths) if lengths else 0,
    }


def write_stats_report(stats: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# MiniLM text training views — statistics",
        "",
        "Sources included: `hf_synthetic`, `hf_manual`, `contextdp` (usable text only).",
        "Excluded: B4E2 and any rows without usable OCR/text.",
        "",
    ]
    for s in stats:
        lines.extend(
            [
                f"## {s['split']}",
                "",
                f"- Samples: **{s['num_samples']}**",
                f"- Avg text length: **{s['avg_text_length']}** (median {s['median_text_length']})",
                f"- Min/max length: {s['min_text_length']} / {s['max_text_length']}",
                "",
                "### Class distribution",
                "",
            ]
        )
        for k, v in sorted((s.get("class_distribution") or {}).items()):
            lines.append(f"- `{k}`: {v}")
        lines.extend(["", "### Source distribution", ""])
        for k, v in sorted((s.get("source_distribution") or {}).items()):
            lines.append(f"- `{k}`: {v}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stats: list[dict[str, Any]] = []
    for split in ("train", "val", "test"):
        src = UNIFIED_DIR / f"{split}.jsonl"
        if not src.is_file():
            raise FileNotFoundError(src)
        filtered = filter_text_samples(_load_jsonl(src))
        out = OUT_DIR / f"text_{split}.jsonl"
        _write_jsonl(out, filtered)
        stats.append(split_stats(split, filtered))
        print(f"Wrote {out} ({len(filtered)} samples)")

    report = OUT_DIR / "text_stats_report.md"
    write_stats_report(stats, report)
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
