"""OCR helpers with RapidOCR + cache + optional parallelism."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from PIL import Image


_CACHE: dict[str, str] = {}
_CACHE_PATH: Path | None = None
_STATS = {"hits": 0, "misses": 0, "errors": 0, "chars": 0}


def load_cache(path: Path) -> None:
    global _CACHE, _CACHE_PATH
    _CACHE_PATH = path
    if path.exists():
        _CACHE = json.loads(path.read_text(encoding="utf-8"))
    else:
        _CACHE = {}


def save_cache() -> None:
    if _CACHE_PATH is None:
        return
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _CACHE_PATH.write_text(json.dumps(_CACHE, ensure_ascii=False), encoding="utf-8")


def stats() -> dict[str, Any]:
    return dict(_STATS)


def file_key(path: Path) -> str:
    st = path.stat()
    return hashlib.sha1(f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}".encode()).hexdigest()


def _ocr_one(path_str: str, max_side: int = 960) -> tuple[str, str, str]:
    """Worker-safe OCR. Returns (key, text, error)."""
    path = Path(path_str)
    key = file_key(path)
    try:
        from rapidocr_onnxruntime import RapidOCR

        engine = RapidOCR()
        img = Image.open(path).convert("RGB")
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.BILINEAR)
        result, _ = engine(img)
        lines: list[str] = []
        if result:
            for item in result:
                if len(item) >= 2 and item[1]:
                    lines.append(str(item[1]).strip())
        return key, "\n".join(lines).strip(), ""
    except Exception as exc:  # noqa: BLE001
        return key, "", str(exc)


def ocr_image(path: Path, max_side: int = 960) -> str:
    """Return OCR text for an image path; cached by content signature."""
    key = file_key(path)
    if key in _CACHE and not key.endswith("__error"):
        # cached empty string is valid
        if key in _CACHE:
            _STATS["hits"] += 1
            return _CACHE[key]

    _STATS["misses"] += 1
    key2, text, err = _ocr_one(str(path), max_side=max_side)
    _CACHE[key2] = text
    _STATS["chars"] += len(text)
    if err:
        _STATS["errors"] += 1
        _CACHE[f"{key2}__error"] = err
    if _STATS["misses"] % 25 == 0:
        save_cache()
    return text


def ocr_many(paths: list[Path], workers: int = 4, max_side: int = 960) -> dict[str, str]:
    """OCR many images in parallel; updates global cache; returns path->text."""
    pending: list[Path] = []
    out: dict[str, str] = {}
    for p in paths:
        key = file_key(p)
        if key in _CACHE:
            _STATS["hits"] += 1
            out[str(p)] = _CACHE[key]
        else:
            pending.append(p)

    if not pending:
        return out

    workers = max(1, min(workers, len(pending)))
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_ocr_one, str(p), max_side): p for p in pending}
        for fut in as_completed(futs):
            p = futs[fut]
            key, text, err = fut.result()
            _CACHE[key] = text
            _STATS["misses"] += 1
            _STATS["chars"] += len(text)
            if err:
                _STATS["errors"] += 1
                _CACHE[f"{key}__error"] = err
            out[str(p)] = text
            done += 1
            if done % 20 == 0 or done == len(pending):
                print(f"  OCR parallel {done}/{len(pending)}", flush=True)
                save_cache()
    save_cache()
    return out
