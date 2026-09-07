"""Phase 4 / 2.5 text classifier — fine-tuned MiniLM with exemplar fallback."""

from __future__ import annotations

import time
from typing import Any

import torch

from ai.common.base import BaseDetector
from ai.common.types import (
    Category,
    EvalMetrics,
    PredictionResult,
    ScanPayload,
    TextPrediction,
    TrainResult,
)
from ai.models.config import Phase4ModelConfig
from ai.text.classifier.finetuned_minilm import (
    REPORT_CLASSES,
    collect_classification_text,
    distribute_binary_to_report_classes,
    load_finetuned_minilm,
    top_k_probs,
)
from ai.text.embeddings.encoder import SentenceTransformerEngine
from ai.text.semantic.exemplars import PATTERN_TO_CATEGORY
from ai.text.semantic.similarity import collect_candidate_texts
from ai.training.models.minilm_classifier import tokenize_batch

_CLASS_TO_CATEGORY: dict[str, Category] = {
    "cookie_consent_manipulation": Category.COOKIE_CONSENT_MANIPULATION,
    "confirmshaming": Category.CONFIRMSHAMING,
    "subscription_trap": Category.HIDDEN_SUBSCRIPTION,
    "forced_continuity": Category.MISLEADING_FREE_TRIAL,
    "no_dark_pattern": Category.NO_DARK_PATTERN,
}


class TextClassifier(BaseDetector[ScanPayload, PredictionResult]):
    """
    Prefer Phase 2 fine-tuned MiniLM; fall back to exemplar similarity if the
    checkpoint cannot load. Rules remain the source of truth in fusion.
    """

    name = "text_classifier"
    version = "0.5.0"

    def __init__(
        self,
        config: Phase4ModelConfig | None = None,
        embedding: SentenceTransformerEngine | None = None,
        *,
        finetuned_bundle: dict[str, Any] | None = None,
        skip_finetuned_autoload: bool = False,
    ) -> None:
        self.config = config or Phase4ModelConfig.from_env()
        self._embedding = embedding or SentenceTransformerEngine(self.config)
        self._last_inference_ms = 0.0
        self._finetuned: dict[str, Any] | None = finetuned_bundle
        self._finetuned_error: str | None = None
        if self._finetuned is None and not skip_finetuned_autoload:
            self._try_autoload_finetuned()

    def _try_autoload_finetuned(self) -> None:
        try:
            self._finetuned = load_finetuned_minilm(device=self.config.device)
            self._finetuned_error = None
        except Exception as exc:  # noqa: BLE001 — intentional soft fallback
            self._finetuned = None
            self._finetuned_error = str(exc)

    def bind_finetuned(self, bundle: dict[str, Any] | None) -> None:
        self._finetuned = bundle
        if bundle:
            self._finetuned_error = None

    def finetuned_status(self) -> dict[str, Any]:
        if self._finetuned and self._finetuned.get("loaded"):
            return {
                "loaded": True,
                "status": "loaded",
                "checkpoint": self._finetuned.get("checkpoint"),
                "model_version": self._finetuned.get("model_version"),
                "inference_device": self._finetuned.get("inference_device")
                or self._finetuned.get("device"),
                "name": "Fine-tuned MiniLM",
            }
        return {
            "loaded": False,
            "status": "not_loaded",
            "checkpoint": None,
            "model_version": None,
            "inference_device": self.config.device,
            "name": "Fine-tuned MiniLM",
            "error": self._finetuned_error,
        }

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(
            status="not_applicable",
            message="Fine-tune via `python -m ai.training.train_minilm`; inference loads the checkpoint.",
        )

    def classify(self, input_data: ScanPayload) -> TextPrediction:
        t0 = time.perf_counter()

        if self._finetuned and self._finetuned.get("loaded"):
            pred = self._classify_finetuned(input_data)
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
            if pred.message:
                pred = pred.model_copy(
                    update={
                        "message": (
                            f"{pred.message} inference_ms={self._last_inference_ms:.1f}"
                        )
                    }
                )
            return pred

        # Fallback: exemplar similarity (or stub)
        return self._classify_exemplar_fallback(input_data, t0)

    def _classify_finetuned(self, input_data: ScanPayload) -> TextPrediction:
        text = collect_classification_text(input_data)
        css = input_data.css_snapshot or {}
        has_dom = bool(css.get("banner") or css.get("cmp") or css.get("buttons") or text)

        if not text.strip():
            empty_probs = {k: (1.0 if k == "no_dark_pattern" else 0.0) for k in REPORT_CLASSES}
            return TextPrediction(
                status="ready",
                category=Category.NO_DARK_PATTERN,
                confidence=0.0,
                evidence_spans=[],
                message="NLP (Fine-tuned MiniLM) ready; no text candidates collected.",
                backend="finetuned_minilm",
                predicted_class="no_dark_pattern",
                class_probabilities=empty_probs,
                logits=[],
            )

        bundle = self._finetuned or {}
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]
        device = bundle.get("device") or self.config.device
        meta = bundle.get("meta") or {}
        max_length = int(meta.get("max_length") or 256)
        label_vocab: dict[str, int] = dict(meta.get("label_vocab") or {})
        id2label: dict[int, str] = {
            int(k): str(v) for k, v in (meta.get("id2label") or {}).items()
        }
        if not id2label and label_vocab:
            id2label = {int(i): str(lab) for lab, i in label_vocab.items()}

        encoded = tokenize_batch(
            tokenizer,
            [text],
            max_length=max_length,
            device=device,
        )
        with torch.inference_mode():
            out = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
            logits_t = out["logits"][0]
            probs_t = out["probs"][0]
            logits = [float(x) for x in logits_t.detach().cpu().tolist()]
            probs_bin = [float(x) for x in probs_t.detach().cpu().tolist()]

        # Binary head: dark_pattern / no_dark_pattern (order from vocab)
        p_dark = 0.0
        p_none = 0.0
        for idx, p in enumerate(probs_bin):
            lab = (id2label.get(idx) or "").lower()
            if lab in ("no_dark_pattern", "none", "benign", "clean"):
                p_none = p
            elif lab in ("dark_pattern", "dark", "positive"):
                p_dark = p
            elif len(probs_bin) == 2 and idx == 0:
                p_dark = p
            elif len(probs_bin) == 2 and idx == 1:
                p_none = p

        if p_dark + p_none <= 0 and len(probs_bin) >= 2:
            p_dark, p_none = probs_bin[0], probs_bin[1]

        class_probabilities = distribute_binary_to_report_classes(p_dark, p_none, text)
        top = top_k_probs(class_probabilities, k=3)
        predicted_key = max(class_probabilities.items(), key=lambda kv: kv[1])[0]
        category = _CLASS_TO_CATEGORY.get(predicted_key, Category.UNKNOWN)
        confidence = float(class_probabilities[predicted_key])

        # Do not invent dark-pattern labels without DOM/text grounding
        if not has_dom and category != Category.NO_DARK_PATTERN:
            category = Category.NO_DARK_PATTERN
            predicted_key = "no_dark_pattern"
            confidence = float(class_probabilities.get("no_dark_pattern", p_none))

        spans = [
            {
                "predicted_class": predicted_key,
                "confidence": confidence,
                "class_probabilities": class_probabilities,
                "top_3": top,
                "logits": logits,
                "binary": {"dark_pattern": round(p_dark, 6), "no_dark_pattern": round(p_none, 6)},
                "source_text": text[:400],
                "backend": "finetuned_minilm",
                "checkpoint": bundle.get("checkpoint"),
            }
        ]
        return TextPrediction(
            status="ready",
            category=category,
            confidence=round(confidence, 4),
            evidence_spans=spans,
            message=(
                f"NLP (Fine-tuned MiniLM) predicted={predicted_key} "
                f"conf={confidence:.2f} device={device}"
            ),
            backend="finetuned_minilm",
            predicted_class=predicted_key,
            class_probabilities=class_probabilities,
            logits=logits,
        )

    def _classify_exemplar_fallback(self, input_data: ScanPayload, t0: float) -> TextPrediction:
        if self.config.stub_mode and self.config.backend == "auto":
            return TextPrediction(
                status="not_loaded",
                message=(
                    "Text model stub mode (AI_STUB_MODE=true) and Fine-tuned MiniLM "
                    "checkpoint unavailable. Set AI_STUB_MODE=false or MINILM_CHECKPOINT."
                ),
                backend="stub",
            )

        candidates = collect_candidate_texts(input_data)
        if not candidates:
            self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
            return TextPrediction(
                status="ready",
                category=Category.NO_DARK_PATTERN,
                confidence=0.0,
                evidence_spans=[],
                message="Exemplar NLP ready; no consent-related text candidates collected.",
                backend="exemplar_similarity",
            )

        css = input_data.css_snapshot or {}
        has_dom_signal = bool(css.get("banner") or css.get("cmp") or css.get("buttons"))
        matches = self._embedding.match_exemplars(candidates)
        if not has_dom_signal:
            matches = [m for m in matches if m["pattern"] in ("confirmshaming",)]

        self._last_inference_ms = (time.perf_counter() - t0) * 1000.0
        if not matches:
            return TextPrediction(
                status="ready",
                category=Category.NO_DARK_PATTERN,
                confidence=0.0,
                evidence_spans=[],
                message=(
                    f"Exemplar NLP ready ({self._embedding._resolve_backend()}); "
                    "no semantic matches above threshold."
                ),
                backend="exemplar_similarity",
            )

        top = matches[0]
        pattern_cats = {
            "preselected_consent",
            "forced_action",
            "confirmshaming",
            "obstruction",
            "interface_interference",
        }
        dark = [m for m in matches if m["pattern"] in pattern_cats]
        if dark:
            top_dark = dark[0]
            cat_name = PATTERN_TO_CATEGORY.get(
                top_dark["pattern"], Category.COOKIE_CONSENT_MANIPULATION.value
            )
            category = Category.COOKIE_CONSENT_MANIPULATION
            for c in Category:
                if c.value == cat_name:
                    category = c
                    break
            conf = float(top_dark["confidence"])
        else:
            category = Category.NO_DARK_PATTERN
            conf = float(top["confidence"]) * 0.5

        spans = [
            {
                "predicted_pattern": m["pattern"],
                "confidence": m["confidence"],
                "embedding_similarity": m["embedding_similarity"],
                "matched_example": m["matched_example"],
                "source_text": m["source_text"],
            }
            for m in matches[:8]
        ]
        return TextPrediction(
            status="ready",
            category=category,
            confidence=conf,
            evidence_spans=spans,
            message=(
                f"NLP backend=exemplar_similarity "
                f"top={top['pattern']} sim={top['confidence']:.2f} "
                f"inference_ms={self._last_inference_ms:.1f}"
                + (
                    f" (MiniLM fallback: {self._finetuned_error})"
                    if self._finetuned_error
                    else ""
                )
            ),
            backend="exemplar_similarity",
            predicted_class=category.value if category else None,
        )

    def predict(self, input_data: ScanPayload) -> PredictionResult:
        pred = self.classify(input_data)
        return PredictionResult(status=pred.status, data=pred.model_dump(), message=pred.message)

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(status="not_implemented", message="Use apps/api/tests/test_phase25_minilm.py.")

    def load_model(self, path: str) -> None:
        try:
            self._finetuned = load_finetuned_minilm(path, device=self.config.device, force_reload=True)
            self._finetuned_error = None
        except Exception as exc:  # noqa: BLE001
            self._finetuned = None
            self._finetuned_error = str(exc)

    def save_model(self, path: str) -> None:
        return None

    def is_ready(self) -> bool:
        if self._finetuned and self._finetuned.get("loaded"):
            return True
        if self.config.stub_mode and self.config.backend == "auto":
            return False
        return self._embedding.is_ready()
