"""Build AI Analysis block for reports (Phase 4)."""

from __future__ import annotations

from typing import Any

from ai.common.types import ExplainableReport, FusionOutput, TextPrediction, VisionFeatures


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
            "category": text.category.value if text and text.category else None,
            "confidence": text.confidence if text else None,
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
                "pattern": f.get("predicted_pattern"),
                "similarity": f.get("embedding_similarity"),
            }
            for f in nlp_findings
            if f.get("embedding_similarity") is not None
        ],
        "model_confidence": {
            "nlp": text.confidence if text and text.status == "ready" else 0.0,
            "vision": vision_conf,
        },
        "fusion_contribution": contribution,
        "note": "Rules remain the source of truth; AI findings are assistive and DOM-grounded.",
    }


def attach_ai_analysis(
    report: ExplainableReport,
    *,
    text: TextPrediction | None,
    vision: VisionFeatures | None,
    fusion: FusionOutput,
) -> ExplainableReport:
    return report.model_copy(
        update={"ai_analysis": build_ai_analysis(text=text, vision=vision, fusion=fusion)}
    )
