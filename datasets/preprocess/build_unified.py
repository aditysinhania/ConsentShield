"""
Build a unified multimodal dark-pattern dataset from:
  - ContextDP (AidUI)
  - B4E2 YOLO export
  - Hugging Face WIPI deceptive patterns (synthetic + manual)

Ignores CookieScanner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from PIL import Image  # noqa: F401 — kept for optional phash path

ROOT = Path(__file__).resolve().parents[1]  # datasets/
REPO = ROOT.parent
RAW = ROOT / "raw"
OUT = ROOT / "unified"

sys.path.insert(0, str(ROOT))
from preprocess.labels import (  # noqa: E402
    B4E2_COMPONENT_MAP,
    normalize_b4e2,
    normalize_contextdp,
    normalize_hf,
    normalize_hf_page,
)
from preprocess import ocr as ocr_mod  # noqa: E402

try:
    import imagehash
except ImportError:  # pragma: no cover
    imagehash = None


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def sha1_text(text: str) -> str:
    norm = re.sub(r"\s+", " ", (text or "").strip().lower())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def phash_image(path: Path, enabled: bool = False) -> str | None:
    if not enabled or imagehash is None:
        return None
    try:
        with Image.open(path) as im:
            return str(imagehash.phash(im.convert("RGB")))
    except Exception:  # noqa: BLE001
        return None


def file_sig(path: Path) -> str:
    st = path.stat()
    return sha1_text(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}")


def make_id(source: str, *parts: str) -> str:
    raw = "|".join([source, *parts])
    return f"{source}_{hashlib.sha1(raw.encode()).hexdigest()[:16]}"


# ---------------------------------------------------------------------------
# ContextDP
# ---------------------------------------------------------------------------

def load_contextdp(use_phash: bool = False) -> list[dict[str, Any]]:
    base = RAW / "contextdp" / "evaluation_dataset"
    samples: list[dict[str, Any]] = []
    for modality in ("mobile", "web"):
        ann_path = base / modality / "bounding-boxes.json"
        if not ann_path.exists():
            continue
        print(f"  ContextDP {modality}...", flush=True)
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        cat_by_id = {c["id"]: c["name"] for c in data.get("categories", [])}
        anns_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for a in data.get("annotations", []):
            anns_by_image[int(a["image_id"])].append(a)

        for img in data.get("images", []):
            file_name = img["file_name"]  # e.g. images/1944.jpg
            img_path = (base / modality / Path(file_name)).resolve()
            if not img_path.exists() or img_path.name.startswith("._"):
                continue

            local_anns = anns_by_image.get(int(img["id"]), [])
            cat_names = [cat_by_id.get(int(a["category_id"]), "UNKNOWN") for a in local_anns]
            if not cat_names:
                cat_names = ["NO DP"]
            norm = normalize_contextdp(cat_names)
            bboxes = []
            for a in local_anns:
                bb = a.get("bbox") or [0, 0, 0, 0]
                bboxes.append(
                    {
                        "x": bb[0],
                        "y": bb[1],
                        "width": bb[2],
                        "height": bb[3],
                        "category": cat_by_id.get(int(a["category_id"]), "UNKNOWN"),
                    }
                )
            text = ""
            dedupe_img = phash_image(img_path, enabled=use_phash) or file_sig(img_path)
            samples.append(
                {
                    "id": make_id("contextdp", modality, str(img["id"]), img_path.name),
                    "source": "contextdp",
                    "modality": "screenshot+ocr",
                    "image_path": rel(img_path),
                    "text": text,
                    "ocr_text": text,
                    "label_binary": norm.binary,
                    "label_fine": norm.fine,
                    "label_consentshield": norm.consentshield,
                    "source_labels": list(norm.source_labels),
                    "bboxes": bboxes,
                    "components": [],
                    "metadata": {
                        "modality": modality,
                        "width": img.get("width"),
                        "height": img.get("height"),
                        "source_file": file_name,
                    },
                    "_dedupe_image": dedupe_img,
                    "_dedupe_text": sha1_text(text) if text else None,
                }
            )
    return samples


# ---------------------------------------------------------------------------
# B4E2
# ---------------------------------------------------------------------------

def _parse_yolo_label(path: Path) -> list[str]:
    comps: list[str] = []
    if not path.exists():
        return comps
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if not parts:
            continue
        comps.append(B4E2_COMPONENT_MAP.get(parts[0], parts[0]))
    return comps


def _yolo_boxes_norm(path: Path) -> list[dict[str, Any]]:
    """Keep YOLO-normalized boxes (xc, yc, w, h in 0-1) when pixel size unknown."""
    boxes: list[dict[str, Any]] = []
    if not path.exists():
        return boxes
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls, xc, yc, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        boxes.append(
            {
                "xc": xc,
                "yc": yc,
                "w": w,
                "h": h,
                "format": "yolo_norm",
                "category": B4E2_COMPONENT_MAP.get(cls, cls),
            }
        )
    return boxes


def load_b4e2(use_phash: bool = False) -> list[dict[str, Any]]:
    base = RAW / "b4e2" / "extracted"
    samples: list[dict[str, Any]] = []
    for split_name, folder in (("train", "train"), ("val", "valid"), ("test", "test")):
        img_dir = base / folder / "images"
        lab_dir = base / folder / "labels"
        if not img_dir.exists():
            continue
        print(f"  B4E2 {folder}...", flush=True)
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
                continue
            if img_path.name.startswith("._"):
                continue
            lab_path = lab_dir / f"{img_path.stem}.txt"
            comps = _parse_yolo_label(lab_path)
            norm = normalize_b4e2(comps)
            text = ""
            samples.append(
                {
                    "id": make_id("b4e2", folder, img_path.name),
                    "source": "b4e2",
                    "modality": "screenshot+ocr",
                    "image_path": rel(img_path),
                    "text": text,
                    "ocr_text": text,
                    "label_binary": norm.binary,
                    "label_fine": norm.fine,
                    "label_consentshield": norm.consentshield,
                    "source_labels": list(norm.source_labels),
                    "bboxes": _yolo_boxes_norm(lab_path),
                    "components": list(norm.source_labels),
                    "metadata": {
                        "original_split": split_name,
                    },
                    "_preset_split": split_name,
                    "_dedupe_image": phash_image(img_path, enabled=use_phash) or file_sig(img_path),
                    "_dedupe_text": None,
                }
            )
    return samples


def apply_ocr(
    samples: list[dict[str, Any]],
    *,
    workers: int,
    limit: int = 0,
    sources: set[str] | None = None,
) -> None:
    # Always hydrate from cache for every screenshot sample.
    for s in samples:
        if not s.get("image_path"):
            continue
        p = REPO / s["image_path"]
        key = ocr_mod.file_key(p)
        cached = ocr_mod._CACHE.get(key)  # noqa: SLF001
        if cached is not None:
            s["text"] = cached
            s["ocr_text"] = cached
            s["_dedupe_text"] = sha1_text(cached) if cached else None

    paths: list[Path] = []
    owners: list[dict[str, Any]] = []
    for s in samples:
        if not s.get("image_path"):
            continue
        if sources and s.get("source") not in sources:
            continue
        if s.get("ocr_text"):
            continue  # already have cached OCR
        paths.append(REPO / s["image_path"])
        owners.append(s)
        if limit and len(paths) >= limit:
            break
    if not paths:
        print("OCR: nothing new to process (all requested images cached or filtered).", flush=True)
        return
    print(f"Running OCR on {len(paths)} screenshots (workers={workers})...", flush=True)
    mapping = ocr_mod.ocr_many(paths, workers=workers)
    for s, p in zip(owners, paths):
        text = mapping.get(str(p), "")
        s["text"] = text
        s["ocr_text"] = text
        s["_dedupe_text"] = sha1_text(text) if text else None


# ---------------------------------------------------------------------------
# Hugging Face
# ---------------------------------------------------------------------------

_LINE_RE = re.compile(
    r'^"?([^"|]+)"?\s*\|\s*"?([^"|]+)"?\s*\|\s*"?([^"]*)"?\s*$'
)


def _parse_hf_output_block(block: str) -> list[tuple[str, str, str]]:
    """Parse multi-line or single-line HF output into (label, pattern, reasoning)."""
    rows: list[tuple[str, str, str]] = []
    text = (block or "").strip()
    if not text:
        return rows
    # Prefer line-wise when multiple annotations present
    chunks = re.split(r"\n+", text) if "\n" in text else [text]
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # strip surrounding quotes per field via split on "|"
        parts = [p.strip().strip('"') for p in chunk.split("|")]
        if len(parts) >= 2:
            label = parts[0]
            pattern = parts[1]
            reasoning = parts[2] if len(parts) > 2 else ""
            rows.append((label, pattern, reasoning))
    return rows


def _extract_visible_text(inp: str) -> str:
    """Pull UI text tokens from ElementMap-style input strings."""
    texts: list[str] = []
    for line in (inp or "").splitlines():
        line = line.strip()
        if not line:
            continue
        # "Some label"|"button"|...
        m = re.match(r'^"?([^"|]+)"?\s*\|', line)
        if m:
            tok = m.group(1).strip()
            if tok and tok.lower() not in {"text", "button"}:
                texts.append(tok)
        else:
            parts = [p.strip().strip('"') for p in line.split("|")]
            if parts and parts[0]:
                texts.append(parts[0])
    # de-dup preserving order
    seen: set[str] = set()
    out: list[str] = []
    for t in texts:
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return "\n".join(out)


def load_hf() -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    files = [
        (RAW / "deceptive_patterns" / "synthetic" / "train-00000-of-00001.parquet", "synthetic", "train"),
        (RAW / "deceptive_patterns" / "synthetic" / "test-00000-of-00001.parquet", "synthetic", "test"),
        (RAW / "deceptive_patterns" / "manual_annotations" / "data-00000-of-00001.parquet", "manual", None),
    ]
    for path, subset, preset in files:
        if not path.exists():
            continue
        table = pq.read_table(path)
        cols = table.to_pydict()
        n = len(next(iter(cols.values())))
        for i in range(n):
            inp = cols.get("input", [""] * n)[i] or ""
            out = cols.get("output", [""] * n)[i] or ""
            site = cols.get("site_name", [None] * n)[i]
            file_url = cols.get("file_url", [None] * n)[i]
            parsed = _parse_hf_output_block(out)
            if not parsed:
                # fallback single field
                parts = [p.strip().strip('"') for p in out.split("|")]
                parsed = [(parts[0] if parts else "unknown", parts[1] if len(parts) > 1 else "", parts[2] if len(parts) > 2 else "")]
            element_pairs = [(lab, pat) for lab, pat, _ in parsed]
            # If this looks like a single-element row, use single normalize; else page aggregate
            if len(parsed) == 1:
                norm = normalize_hf(parsed[0][0], parsed[0][1])
            else:
                norm = normalize_hf_page(element_pairs)
            visible = _extract_visible_text(inp)
            reason_blob = " | ".join(r for _, _, r in parsed if r)
            text = visible if visible else inp[:4000]
            sample: dict[str, Any] = {
                "id": make_id("hf", subset, str(i), sha1_text(inp)[:12]),
                "source": f"hf_{subset}",
                "modality": "text",
                "image_path": None,
                "text": text,
                "ocr_text": None,
                "label_binary": norm.binary,
                "label_fine": norm.fine,
                "label_consentshield": norm.consentshield,
                "source_labels": list(norm.source_labels),
                "bboxes": [],
                "components": [],
                "metadata": {
                    "subset": subset,
                    "site_name": site,
                    "file_url": file_url,
                    "n_elements": len(parsed),
                    "reasoning": reason_blob[:2000],
                    "raw_output_preview": out[:500],
                },
                "_dedupe_image": None,
                "_dedupe_text": sha1_text(text or inp),
            }
            if preset:
                sample["_preset_split"] = "val" if preset == "test" and subset == "synthetic" and i == 0 else (
                    "test" if preset == "test" else None
                )
                # Keep synthetic official test rows as test; train unmarked for global split
                if preset == "test":
                    sample["_preset_split"] = "test"
            samples.append(sample)
    return samples


# ---------------------------------------------------------------------------
# Dedup + split
# ---------------------------------------------------------------------------

def deduplicate(samples: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    seen_img: set[str] = set()
    seen_txt: set[str] = set()
    kept: list[dict[str, Any]] = []
    dropped = Counter()
    for s in samples:
        img_k = s.get("_dedupe_image")
        txt_k = s.get("_dedupe_text")
        # Screenshot samples: dedupe by perceptual/file hash
        if s.get("image_path"):
            if img_k and img_k in seen_img:
                dropped["image_duplicate"] += 1
                continue
            if img_k:
                seen_img.add(img_k)
            # also text collision among screenshots with identical OCR
            if txt_k and txt_k in seen_txt and s["source"] != "contextdp":
                # allow contextdp OCR collisions lightly; still keep unique images
                pass
            if txt_k:
                seen_txt.add(txt_k)
            kept.append(s)
            continue
        # Text-only: dedupe by normalized text hash
        if txt_k and txt_k in seen_txt:
            dropped["text_duplicate"] += 1
            continue
        if txt_k:
            seen_txt.add(txt_k)
        kept.append(s)
    return kept, {"dropped": dict(dropped), "before": len(samples), "after": len(kept)}


def stratified_split(
    samples: list[dict[str, Any]],
    seed: int = 42,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[dict[str, Any]]]:
    """
    Assign splits. Prefer B4E2 original splits when present.
    Remaining samples stratified by (source, label_binary).
    """
    train, val, test = [], [], []
    remaining: list[dict[str, Any]] = []

    for s in samples:
        preset = s.get("_preset_split")
        if s["source"] == "b4e2" and preset in {"train", "val", "test"}:
            {"train": train, "val": val, "test": test}[preset].append(s)
        elif preset == "test" and str(s["source"]).startswith("hf_"):
            test.append(s)
        else:
            remaining.append(s)

    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for s in remaining:
        buckets[(s["source"], s["label_binary"])].append(s)

    for key, items in buckets.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # ensure non-empty leftovers go to test
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return {"train": train, "val": val, "test": test}


def strip_private(sample: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in sample.items() if not k.startswith("_")}


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(s["source"] for s in samples)
    by_binary = Counter(s["label_binary"] for s in samples)
    by_fine = Counter(s["label_fine"] for s in samples)
    by_cs = Counter(s["label_consentshield"] for s in samples)
    with_ocr = sum(1 for s in samples if s.get("ocr_text"))
    with_image = sum(1 for s in samples if s.get("image_path"))
    return {
        "n": len(samples),
        "by_source": dict(by_source),
        "by_binary": dict(by_binary),
        "by_fine": dict(by_fine),
        "by_consentshield": dict(by_cs),
        "with_ocr_text": with_ocr,
        "with_image": with_image,
    }


def write_report(
    path: Path,
    *,
    loaded: dict[str, int],
    dedupe_info: dict[str, Any],
    splits: dict[str, list[dict[str, Any]]],
    ocr_stats: dict[str, Any],
    notes: list[str],
) -> None:
    all_samples = [s for v in splits.values() for s in v]
    overall = summarize(all_samples)
    lines: list[str] = []
    lines.append("# Unified Dataset — Preprocessing Report")
    lines.append("")
    lines.append(f"- Generated (UTC): `{datetime.now(timezone.utc).isoformat()}`")
    lines.append("- Sources used: **ContextDP**, **B4E2**, **Hugging Face Deceptive Patterns**")
    lines.append("- Sources ignored: **CookieScanner**")
    lines.append("")
    lines.append("## Pipeline steps")
    lines.append("1. Load raw samples from each source")
    lines.append("2. Normalize labels → binary + fine + ConsentShield category")
    lines.append("3. OCR screenshots (ContextDP + B4E2) via RapidOCR; HF uses ElementMap text")
    lines.append("4. Deduplicate (image perceptual/file hash; text sha1)")
    lines.append("5. Create train / val / test splits (B4E2 keeps original split; others stratified 80/10/10)")
    lines.append("")
    lines.append("## Load counts (before dedupe)")
    for k, v in loaded.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Deduplication")
    lines.append(f"- Before: **{dedupe_info['before']}**")
    lines.append(f"- After: **{dedupe_info['after']}**")
    lines.append(f"- Dropped: `{json.dumps(dedupe_info.get('dropped', {}))}`")
    lines.append("")
    lines.append("## OCR")
    lines.append(f"- Cache hits: {ocr_stats.get('hits', 0)}")
    lines.append(f"- Cache misses / new OCR: {ocr_stats.get('misses', 0)}")
    lines.append(f"- OCR errors: {ocr_stats.get('errors', 0)}")
    lines.append(f"- Characters extracted: {ocr_stats.get('chars', 0)}")
    lines.append("")
    lines.append("## Final corpus")
    lines.append(f"- Total samples: **{overall['n']}**")
    lines.append(f"- With images: {overall['with_image']}")
    lines.append(f"- With OCR text: {overall['with_ocr_text']}")
    lines.append("")
    lines.append("### By split")
    for name, rows in splits.items():
        s = summarize(rows)
        lines.append(f"- **{name}**: {s['n']} (binary={s['by_binary']})")
    lines.append("")
    lines.append("### By source")
    for k, v in sorted(overall["by_source"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### Binary labels")
    for k, v in sorted(overall["by_binary"].items()):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### Fine labels")
    for k, v in sorted(overall["by_fine"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("### ConsentShield category mapping")
    for k, v in sorted(overall["by_consentshield"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{k}`: {v}")
    lines.append("")
    lines.append("## Label normalization notes")
    lines.append("- ContextDP categories mapped into unified fine taxonomy (scarcity_* → `scarcity_urgency`).")
    lines.append("- B4E2 screenshots are treated as **dark_pattern** context with fine label `ui_component_context` (YOLO classes retained under `components` / bboxes).")
    lines.append("- HF rows use MIT synthetic labels; page-level rows with mixed elements aggregate to dark_pattern if any element is deceptive.")
    lines.append("- Scarcity/urgency patterns map to ConsentShield `Unknown` (project does not claim unverifiable scarcity as a primary product category).")
    lines.append("")
    lines.append("## Outputs")
    lines.append("```text")
    lines.append("datasets/unified/")
    lines.append("  train.jsonl")
    lines.append("  val.jsonl")
    lines.append("  test.jsonl")
    lines.append("  label_map.json")
    lines.append("  ocr_cache.json")
    lines.append("  preprocessing_report.md")
    lines.append("```")
    lines.append("")
    if notes:
        lines.append("## Notes / warnings")
        for n in notes:
            lines.append(f"- {n}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build unified dark-pattern dataset")
    parser.add_argument("--no-ocr", action="store_true", help="Skip screenshot OCR (not recommended)")
    parser.add_argument("--phash", action="store_true", help="Use perceptual image hashing (slower on OneDrive)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel OCR workers")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-ocr", type=int, default=0, help="OCR at most N images (0=all), for smoke tests")
    parser.add_argument(
        "--ocr-sources",
        type=str,
        default="contextdp,b4e2",
        help="Comma list of screenshot sources to OCR (e.g. contextdp or contextdp,b4e2)",
    )
    args = parser.parse_args()

    print("ConsentShield unified dataset builder", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ocr_mod.load_cache(OUT / "ocr_cache.json")
    notes: list[str] = []
    do_ocr = not args.no_ocr
    use_phash = bool(args.phash)
    ocr_sources = {s.strip() for s in args.ocr_sources.split(",") if s.strip()}
    if not use_phash:
        notes.append("Image dedupe uses file signature (path+size+mtime). Pass --phash for perceptual hashing.")

    print("Loading ContextDP...", flush=True)
    contextdp = load_contextdp(use_phash=use_phash)
    print("Loading B4E2...", flush=True)
    b4e2 = load_b4e2(use_phash=use_phash)

    if do_ocr:
        if args.limit_ocr:
            notes.append(f"OCR limited to first {args.limit_ocr} screenshot samples (smoke mode).")
        notes.append(f"OCR sources: {sorted(ocr_sources)}")
        apply_ocr(contextdp + b4e2, workers=args.workers, limit=args.limit_ocr, sources=ocr_sources)
        missing = sum(1 for s in contextdp + b4e2 if s.get("image_path") and not s.get("ocr_text"))
        if missing:
            notes.append(
                f"{missing} screenshot samples have empty OCR (not in --ocr-sources or OCR failed). "
                "Re-run with --ocr-sources contextdp,b4e2 to backfill."
            )
    else:
        notes.append("OCR skipped (--no-ocr).")


    print("Loading Hugging Face...")
    hf = load_hf()

    loaded = {
        "contextdp": len(contextdp),
        "b4e2": len(b4e2),
        "hf": len(hf),
        "total": len(contextdp) + len(b4e2) + len(hf),
    }
    print("Loaded:", loaded)

    combined = contextdp + b4e2 + hf
    print("Deduplicating...")
    kept, dedupe_info = deduplicate(combined)
    print("Dedupe:", dedupe_info)

    print("Splitting...")
    splits = stratified_split(kept, seed=args.seed)

    label_map = {
        "binary": ["dark_pattern", "no_dark_pattern"],
        "fine": sorted({s["label_fine"] for s in kept}),
        "consentshield": sorted({s["label_consentshield"] for s in kept}),
        "sources": sorted({s["source"] for s in kept}),
        "ignored": ["cookiescanner"],
    }
    (OUT / "label_map.json").write_text(json.dumps(label_map, indent=2), encoding="utf-8")

    counts = {}
    for name, rows in splits.items():
        clean = [strip_private(r) for r in rows]
        counts[name] = write_jsonl(OUT / f"{name}.jsonl", clean)
        print(f"Wrote {name}.jsonl ({counts[name]})")

    ocr_mod.save_cache()
    write_report(
        OUT / "preprocessing_report.md",
        loaded=loaded,
        dedupe_info=dedupe_info,
        splits=splits,
        ocr_stats=ocr_mod.stats(),
        notes=notes,
    )
    print("Report:", OUT / "preprocessing_report.md")
    print("Done.")


if __name__ == "__main__":
    main()
