"""Fine-tuned MiniLM inference helpers (Phase 2.5)."""

from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any

from ai.training.datasets._helpers import REPO_ROOT

# Report-facing class keys (trained head is binary; subtypes are distributed).
REPORT_CLASSES = (
    "cookie_consent_manipulation",
    "confirmshaming",
    "subscription_trap",
    "forced_continuity",
    "no_dark_pattern",
)

_SUBTYPE_CUES: dict[str, tuple[str, ...]] = {
    "confirmshaming": (
        "no thanks",
        "i don't want",
        "continue without",
        "reject all",
        "decline",
        "guilt",
        "miss out",
    ),
    "subscription_trap": (
        "subscription",
        "subscribe",
        "auto-renew",
        "automatically renew",
        "billing",
        "recurring",
        "membership",
    ),
    "forced_continuity": (
        "free trial",
        "after trial",
        "cancel anytime",
        "will be charged",
        "payment method",
        "credit card required",
    ),
    "cookie_consent_manipulation": (
        "cookie",
        "consent",
        "accept all",
        "agree",
        "privacy",
        "partners",
        "legitimate interest",
        "manage preferences",
    ),
}

_CACHE_LOCK = threading.Lock()
_CACHE: dict[str, dict[str, Any]] = {}


def default_checkpoint_candidates() -> list[Path]:
    return [
        Path("models/checkpoints/minilm/best_model.pt"),
        Path("ai/training/runs/minilm/models/best_model.pt"),
        Path("ai/training/runs/minilm/models/best.pt"),
        Path("runs/minilm/models/best_model.pt"),
        Path("runs/minilm/models/best.pt"),
    ]


def resolve_checkpoint(explicit: str | Path | None = None) -> Path | None:
    """
    Resolve MiniLM checkpoint path.

    If ``explicit`` is provided (including via config), only that path is tried
    (no silent fallback). Use ``none`` / ``disabled`` to skip loading.
    When ``explicit`` is None, try ``MINILM_CHECKPOINT`` then default candidates.
    """
    if explicit is not None:
        raw = str(explicit).strip()
        if raw.lower() in ("", "none", "disabled", "__disabled__"):
            return None
        p = Path(raw).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        return p if p.is_file() else None

    env = os.getenv("MINILM_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.is_file():
            return p

    for path in default_checkpoint_candidates():
        p = path.expanduser()
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.is_file():
            return p
    return None


def load_finetuned_minilm(
    checkpoint: str | Path | None = None,
    *,
    device: str = "cpu",
    force_reload: bool = False,
) -> dict[str, Any]:
    """
    Load tokenizer + MiniLMClassifier once (process singleton cache).

    Returns dict with keys: model, tokenizer, meta, device, checkpoint, loaded.
    Raises FileNotFoundError / Exception on hard failure (caller may fall back).
    """
    path = resolve_checkpoint(checkpoint)
    if path is None:
        raise FileNotFoundError("Fine-tuned MiniLM checkpoint not found.")

    cache_key = f"{path.resolve()}::{device}"
    with _CACHE_LOCK:
        if not force_reload and cache_key in _CACHE:
            return _CACHE[cache_key]

    from ai.training.callbacks.checkpoint import load_checkpoint
    from ai.training.models.minilm_classifier import MiniLMClassifier, build_tokenizer

    ckpt = load_checkpoint(path, map_location=device)
    cfg = ckpt.get("config") or {}
    model_cfg = cfg.get("model") or {}
    label_vocab: dict[str, int] = dict(ckpt.get("label_vocab") or {})
    num_labels = len(label_vocab) if label_vocab else 2
    model_name = str(model_cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2"))
    max_length = int(model_cfg.get("max_length", 256))

    model = MiniLMClassifier(
        model_name,
        num_labels=num_labels,
        dropout=float(model_cfg.get("dropout", 0.1)),
        freeze_encoder=bool(model_cfg.get("freeze_encoder", False)),
        freeze_layers=int(model_cfg.get("freeze_layers", 0)),
    )
    state = ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Checkpoint missing model_state_dict: {path}")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    tokenizer = build_tokenizer(model_name, max_length=max_length)
    meta = {
        "checkpoint": str(path),
        "label_vocab": label_vocab,
        "id2label": {i: lab for lab, i in label_vocab.items()},
        "config": cfg,
        "epoch": ckpt.get("epoch"),
        "metrics": ckpt.get("metrics") or {},
        "model_version": f"finetuned-minilm-epoch{ckpt.get('epoch', '?')}",
        "base_model": model_name,
        "max_length": max_length,
    }
    bundle = {
        "loaded": True,
        "model": model,
        "tokenizer": tokenizer,
        "meta": meta,
        "device": device,
        "checkpoint": str(path),
        "status": "loaded",
        "model_version": meta["model_version"],
        "inference_device": device,
    }
    with _CACHE_LOCK:
        _CACHE[cache_key] = bundle
    return bundle


def collect_classification_text(payload: Any) -> str:
    """Aggregate DOM / banner / button / subscription / OCR text for MiniLM."""
    parts: list[str] = []
    css = getattr(payload, "css_snapshot", None) or {}
    if not isinstance(css, dict):
        css = {}

    banner = css.get("banner")
    if isinstance(banner, dict):
        for key in ("text", "ariaLabel", "innerText"):
            val = banner.get(key)
            if isinstance(val, str) and val.strip():
                parts.append(val.strip())

    for btn in css.get("buttons") or []:
        if isinstance(btn, dict):
            label = str(btn.get("text") or btn.get("ariaLabel") or "").strip()
            if label:
                parts.append(label)

    for box in list(css.get("checkboxes") or []) + list(css.get("toggles") or []):
        if isinstance(box, dict):
            label = str(box.get("label") or box.get("name") or box.get("text") or "").strip()
            if label:
                parts.append(label)

    visible = str(getattr(payload, "visible_text", None) or "").strip()
    if visible:
        parts.append(visible[:1200])

    meta = css.get("ocr") or css.get("ocr_text")
    if isinstance(meta, str) and meta.strip():
        parts.append(meta.strip()[:800])
    ocr_list = css.get("ocr_blocks") or []
    if isinstance(ocr_list, list):
        for block in ocr_list[:20]:
            if isinstance(block, dict):
                t = str(block.get("text") or "").strip()
                if t:
                    parts.append(t)
            elif isinstance(block, str) and block.strip():
                parts.append(block.strip())

    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in parts:
        key = re.sub(r"\s+", " ", p.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(p.strip())
    return "\n".join(out)[:2000]


def distribute_binary_to_report_classes(p_dark: float, p_none: float, text: str) -> dict[str, float]:
    """
    Map binary MiniLM outputs onto the five report-facing probability keys.

    The trained head is binary; dark-pattern mass is shared across subtypes using
    lightweight keyword cues so reports can show a structured probability vector.
    """
    p_dark = max(0.0, min(1.0, float(p_dark)))
    p_none = max(0.0, min(1.0, float(p_none)))
    # Renormalize in case of numerical drift
    s = p_dark + p_none
    if s > 0:
        p_dark, p_none = p_dark / s, p_none / s

    low = (text or "").lower()
    weights = {
        "cookie_consent_manipulation": 0.15,
        "confirmshaming": 0.05,
        "subscription_trap": 0.05,
        "forced_continuity": 0.05,
    }
    for cls, cues in _SUBTYPE_CUES.items():
        hit = sum(1 for c in cues if c in low)
        if hit:
            weights[cls] = weights.get(cls, 0.05) + float(hit)

    wsum = sum(weights.values()) or 1.0
    probs = {k: p_dark * (v / wsum) for k, v in weights.items()}
    probs["no_dark_pattern"] = p_none
    # Ensure all keys present and sum ~ 1
    for k in REPORT_CLASSES:
        probs.setdefault(k, 0.0)
    total = sum(probs[k] for k in REPORT_CLASSES) or 1.0
    return {k: round(probs[k] / total, 6) for k in REPORT_CLASSES}


def top_k_probs(probs: dict[str, float], k: int = 3) -> list[dict[str, Any]]:
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    return [{"class": name, "probability": float(p)} for name, p in ranked[:k]]
