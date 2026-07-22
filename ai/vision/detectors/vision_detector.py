"""CLIP-based vision detector with DOM-grounded lexical fallback."""

from __future__ import annotations

import base64
import io
import time
from typing import Any

from ai.common.base import BaseDetector
from ai.common.types import EvalMetrics, PredictionResult, ScanPayload, TrainResult, VisionFeatures
from ai.models.cache import timed_load
from ai.models.config import Phase4ModelConfig

_CLIP_PROMPTS = [
    "a cookie consent banner on a website",
    "a privacy consent dialog overlay",
    "a large modal blocking the webpage",
    "an Accept All cookies button",
    "a Reject All cookies button",
    "a Manage cookie settings button",
    "a website with no cookie banner",
]


class VisionDetector(BaseDetector[ScanPayload, PredictionResult]):
    """
    Phase 4 CLIP zero-shot vision for consent UI.
    Falls back to DOM geometry when pretrained stack unavailable.
    """

    name = "vision_detector"
    version = "0.4.0"

    def __init__(self, config: Phase4ModelConfig | None = None) -> None:
        self.config = config or Phase4ModelConfig.from_env()
        self._backend: str | None = None
        self._load_ms = 0.0
        self._inference_ms = 0.0

    def _resolve_backend(self) -> str:
        if self._backend:
            return self._backend
        if self.config.stub_mode and self.config.backend == "auto":
            self._backend = "stub"
            return self._backend
        if self.config.backend == "lexical":
            self._backend = "lexical"
            return self._backend
        if self.config.backend in ("pretrained", "auto") and not self.config.stub_mode:
            try:
                import transformers  # noqa: F401
                import torch  # noqa: F401

                self._backend = "pretrained"
                return self._backend
            except ImportError:
                if self.config.backend == "pretrained":
                    self._backend = "unavailable"
                    return self._backend
                self._backend = "lexical"
                return self._backend
        self._backend = "lexical" if not self.config.stub_mode else "stub"
        return self._backend

    def _get_clip(self):
        cfg = self.config
        key = f"clip:{cfg.vision_model_name}:{cfg.device}"

        def factory():
            from transformers import CLIPModel, CLIPProcessor

            cache = str(cfg.cache_dir) if cfg.cache_models else None
            if cache:
                cfg.cache_dir.mkdir(parents=True, exist_ok=True)
            model = CLIPModel.from_pretrained(cfg.vision_model_name, cache_dir=cache)
            processor = CLIPProcessor.from_pretrained(cfg.vision_model_name, cache_dir=cache)
            model.to(cfg.device)
            model.eval()
            return {"model": model, "processor": processor}

        bundle, load_ms, _ = timed_load(key, factory)
        self._load_ms = load_ms
        return bundle

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(status="not_applicable", message="Phase 4 uses pretrained CLIP.")

    def extract_features(self, input_data: ScanPayload) -> VisionFeatures:
        t0 = time.perf_counter()
        backend = self._resolve_backend()
        if backend == "stub":
            return VisionFeatures(
                status="not_loaded",
                message="Vision stub mode (AI_STUB_MODE=true). Set AI_STUB_MODE=false for Phase 4 CLIP.",
            )
        if backend == "unavailable":
            return VisionFeatures(
                status="error",
                message="transformers/torch not installed for CLIP.",
            )

        css = input_data.css_snapshot or {}
        banner = css.get("banner") if isinstance(css.get("banner"), dict) else None
        buttons = list(css.get("buttons") or [])

        if backend == "lexical":
            features = self._from_dom(banner, buttons, backend="lexical")
            self._inference_ms = (time.perf_counter() - t0) * 1000.0
            features.message = (
                f"Vision lexical fallback (DOM-grounded). inference_ms={self._inference_ms:.1f}"
            )
            return features

        # pretrained CLIP
        image = _decode_screenshot(input_data.screenshot_base64)
        if image is None:
            # No screenshot — still return DOM-grounded features, never invent
            features = self._from_dom(banner, buttons, backend="pretrained-dom")
            self._inference_ms = (time.perf_counter() - t0) * 1000.0
            features.message = "CLIP ready but no screenshot; used DOM geometry only."
            return features

        try:
            import torch

            bundle = self._get_clip()
            model = bundle["model"]
            processor = bundle["processor"]
            inputs = processor(
                text=_CLIP_PROMPTS,
                images=image,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits_per_image[0]
                probs = logits.softmax(dim=0).detach().cpu().tolist()
        except Exception as exc:  # pragma: no cover - runtime env dependent
            features = self._from_dom(banner, buttons, backend="pretrained-error")
            features.status = "error"
            features.message = f"CLIP inference failed: {exc}"
            return features

        self._inference_ms = (time.perf_counter() - t0) * 1000.0
        labels = [
            "cookie_banner",
            "consent_dialog",
            "large_overlay",
            "accept_button",
            "reject_button",
            "settings_button",
            "no_banner",
        ]
        scores = {labels[i]: float(probs[i]) for i in range(len(labels))}
        banner_score = max(scores["cookie_banner"], scores["consent_dialog"], scores["large_overlay"])
        no_banner = scores["no_banner"]
        banner_detected = banner_score > no_banner and banner_score >= 0.22

        # Ground in DOM: if CLIP says banner but DOM has nothing, lower confidence
        dom_banner = bool(banner and (banner.get("width") or banner.get("height")))
        if banner_detected and not dom_banner and not buttons:
            banner_detected = False

        accept = _btn_box(buttons, ("accept", "agree", "allow", "yes"))
        reject = _btn_box(buttons, ("reject", "decline", "no thank", "necessary", "essential"))
        settings = _btn_box(buttons, ("manage", "settings", "preferences", "customize", "customise"))

        detections = [
            {"label": k, "confidence": round(v, 4)}
            for k, v in sorted(scores.items(), key=lambda x: -x[1])
            if v >= 0.12
        ]
        feature_vector = [scores[k] for k in labels]

        return VisionFeatures(
            status="ready",
            banner_detected=banner_detected or dom_banner,
            accept_button=accept,
            reject_button=reject,
            checkboxes=list(css.get("checkboxes") or []),
            toggles=list(css.get("toggles") or []),
            layout={
                "banner": banner,
                "overlay_score": round(scores["large_overlay"], 4),
                "visual_emphasis": round(max(scores["accept_button"] - scores["reject_button"], 0.0), 4),
                "detections": detections,
                "backend": "pretrained",
                "model": self.config.vision_model_name,
                "load_ms": round(self._load_ms, 2),
                "inference_ms": round(self._inference_ms, 2),
            },
            feature_vector=feature_vector,
            message=(
                f"CLIP vision ready. banner={banner_detected} "
                f"top={detections[0]['label'] if detections else 'none'} "
                f"inference_ms={self._inference_ms:.1f}"
            ),
        )

    def _from_dom(self, banner, buttons, *, backend: str) -> VisionFeatures:
        dom_banner = bool(banner and (banner.get("width") or banner.get("height")))
        accept = _btn_box(buttons, ("accept", "agree", "allow", "yes"))
        reject = _btn_box(buttons, ("reject", "decline", "no thank", "necessary", "essential"))
        coverage = 0.0
        if isinstance(banner, dict):
            # approximate if coverage provided
            coverage = float(banner.get("coverage") or 0.0)
        return VisionFeatures(
            status="ready",
            banner_detected=dom_banner,
            accept_button=accept,
            reject_button=reject,
            layout={
                "banner": banner,
                "overlay_score": coverage,
                "visual_emphasis": _emphasis(accept, reject),
                "detections": (
                    [{"label": "cookie_banner", "confidence": 0.7, "bounding_box": _banner_box(banner)}]
                    if dom_banner
                    else []
                ),
                "backend": backend,
            },
            feature_vector=[1.0 if dom_banner else 0.0],
            message=f"Vision {backend} using collected DOM/CSS.",
        )

    def predict(self, input_data: ScanPayload) -> PredictionResult:
        features = self.extract_features(input_data)
        return PredictionResult(status=features.status, data=features.model_dump(), message=features.message)

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(status="not_implemented", message="See test_phase4_models.py.")

    def load_model(self, path: str) -> None:
        return None

    def save_model(self, path: str) -> None:
        return None

    def is_ready(self) -> bool:
        return self._resolve_backend() not in ("stub", "unavailable")


def _decode_screenshot(b64: str | None):
    if not b64:
        return None
    try:
        from PIL import Image

        raw = b64.split(",", 1)[-1] if "," in b64 else b64
        data = base64.b64decode(raw)
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        return None


def _btn_box(buttons: list, tokens: tuple[str, ...]) -> dict[str, Any] | None:
    for btn in buttons:
        if not isinstance(btn, dict):
            continue
        label = str(btn.get("text") or btn.get("ariaLabel") or "").lower()
        if any(tok in label for tok in tokens):
            return {
                "text": btn.get("text"),
                "bounding_box": {
                    "x": btn.get("x") or btn.get("left"),
                    "y": btn.get("y") or btn.get("top"),
                    "width": btn.get("width"),
                    "height": btn.get("height"),
                },
                "confidence": 0.75,
            }
    return None


def _banner_box(banner: dict | None) -> dict[str, Any] | None:
    if not isinstance(banner, dict):
        return None
    return {
        "x": banner.get("x") or banner.get("left"),
        "y": banner.get("y") or banner.get("top"),
        "width": banner.get("width"),
        "height": banner.get("height"),
    }


def _emphasis(accept: dict | None, reject: dict | None) -> float:
    if not accept or not reject:
        return 0.0
    ab = accept.get("bounding_box") or {}
    rb = reject.get("bounding_box") or {}
    aa = float(ab.get("width") or 0) * float(ab.get("height") or 0)
    ra = float(rb.get("width") or 0) * float(rb.get("height") or 0)
    if ra <= 0:
        return 1.0 if aa > 0 else 0.0
    return round(min(1.0, max(0.0, (aa / ra - 1.0) / 2.0)), 4)
