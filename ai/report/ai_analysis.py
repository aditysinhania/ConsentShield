"""Build AI Analysis block for reports (Phase 4 / 2.5)."""

from __future__ import annotations

from typing import Any

from ai.common.types import ExplainableReport, FusionOutput, TextPrediction, VisionFeatures
from ai.text.classifier.finetuned_minilm import top_k_probs


def _models_used_list(
    *,
    text: TextPrediction | None,
    vision: VisionFeatures | None,
) -> list[str]:
    models = ["Rule Engine"]
    if text and text.status == "ready":
        if (text.backend or "") == "finetuned_minilm" or (
            text.message and "Fine-tuned MiniLM" in text.message
        ):
            models.append("NLP (Fine-tuned MiniLM)")
        elif (text.backend or "") == "exemplar_similarity":
            models.append("Sentence Transformer")
        else:
            models.append("NLP (Fine-tuned MiniLM)" if text.predicted_class else "Sentence Transformer")
    elif text and text.status == "not_loaded":
        models.append("NLP (stub)")

    if vision and vision.status == "ready":
        layout = vision.layout or {}
        backend = layout.get("backend") if isinstance(layout, dict) else None
        if backend == "stub":
            models.append("Vision (CLIP / Stub)")
        else:
            models.append("Vision (CLIP / Stub)" if backend in (None, "lexical") else "Vision (CLIP / Stub)")
    elif vision and vision.status == "not_loaded":
        models.append("Vision (CLIP / Stub)")

    models.append("Fusion")
    return models


def build_ai_analysis(
    *,
    text: TextPrediction | None,
    vision: VisionFeatures | None,
    fusion: FusionOutput,
) -> dict[str, Any]:
    nlp_findings: list[dict[str, Any]] = []
    if text and text.status == "ready":
        for span in text.evidence_spans or []:
            if isinstance(span, dict):
                nlp_findings.append(span)

    probs = dict(text.class_probabilities) if text and text.class_probabilities else {}
    if not probs and nlp_findings:
        first = nlp_findings[0]
        if isinstance(first.get("class_probabilities"), dict):
            probs = dict(first["class_probabilities"])

    top3 = top_k_probs(probs, k=3) if probs else []
    predicted = (
        text.predicted_class
        if text and text.predicted_class
        else (text.category.value if text and text.category else None)
    )

    vision_findings: list[dict[str, Any]] = []
    vision_conf = 0.0
    if vision and vision.status == "ready":
        layout = vision.layout or {}
        dets = layout.get("detections") if isinstance(layout, dict) else []
        if isinstance(dets, list):
            vision_findings = [d for d in dets if isinstance(d, dict)]
        if vision.banner_detected:
            vision_conf = max(
                (float(d.get("confidence") or 0) for d in vision_findings),
                default=0.6,
            )
        vision_findings = [
            {
                "banner_detected": vision.banner_detected,
                "accept_button": vision.accept_button,
                "reject_button": vision.reject_button,
                "detections": vision_findings,
                "backend": layout.get("backend") if isinstance(layout, dict) else None,
                "model": layout.get("model") if isinstance(layout, dict) else None,
                "load_ms": layout.get("load_ms") if isinstance(layout, dict) else None,
                "inference_ms": layout.get("inference_ms") if isinstance(layout, dict) else None,
            }
        ]

    bd = fusion.confidence_breakdown
    contribution = {
        "rules": bd.rules if bd else 0.0,
        "nlp": bd.nlp if bd else 0.0,
        "vision": bd.vision_model if bd else 0.0,
        "fusion": bd.fusion if bd else fusion.confidence,
    }

    return {
        "status": "ready",
        "nlp": {
            "status": text.status if text else "unavailable",
            "backend": text.backend if text else None,
            "name": (
                "Fine-tuned MiniLM"
                if text and (text.backend == "finetuned_minilm" or (text.message and "Fine-tuned MiniLM" in text.message))
                else "Sentence Transformer"
                if text and text.status == "ready"
                else None
            ),
            "predicted_class": predicted,
            "category": text.category.value if text and text.category else None,
            "confidence": text.confidence if text else None,
            "class_probabilities": probs,
            "top_3_class_probabilities": top3,
            "logits": list(text.logits) if text and text.logits else [],
            "findings": nlp_findings,
            "message": text.message if text else None,
        },
        "vision": {
            "status": vision.status if vision else "unavailable",
            "confidence": vision_conf,
            "findings": vision_findings,
            "message": vision.message if vision else None,
        },
        "similarity_scores": [
            {
                "pattern": f.get("predicted_pattern") or f.get("predicted_class"),
                "similarity": f.get("embedding_similarity"),
                "probability": f.get("confidence"),
            }
            for f in nlp_findings
            if f.get("embedding_similarity") is not None or f.get("confidence") is not None
        ],
        "model_confidence": {
            "nlp": text.confidence if text and text.status == "ready" else 0.0,
            "vision": vision_conf,
        },
        "fusion_contribution": contribution,
        "models_used": _models_used_list(text=text, vision=vision),
        "note": "Rules remain the source of truth; Fine-tuned MiniLM findings are assistive and DOM-grounded.",
    }


def attach_ai_analysis(
    report: ExplainableReport,
    *,
    text: TextPrediction | None,
    vision: VisionFeatures | None,
    fusion: FusionOutput,
) -> ExplainableReport:
    analysis = build_ai_analysis(text=text, vision=vision, fusion=fusion)
    return report.model_copy(
        update={
            "ai_analysis": analysis,
            "models_used": list(analysis.get("models_used") or []),
        }
    )
