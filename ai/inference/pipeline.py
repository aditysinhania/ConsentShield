"""End-to-end inference orchestration."""

from __future__ import annotations

from ai.common.types import ExplainableReport, FusionInput, ScanPayload
from ai.explanation.generator import ExplanationGenerator
from ai.fusion.aggregator.fusion_engine import FusionEngine
from ai.performance.metrics import MetricStage, PerformanceCollector
from ai.performance.instrumentation import PerformanceInstrumentation
from ai.registry.model_registry import ModelRegistry
from ai.report.enricher import enrich_report
from ai.rules import RuleEngine
from ai.timeline.recorder import TimelineEventName, TimelineRecorder


def _detect_banner_cmp(payload: ScanPayload, timeline: TimelineRecorder) -> None:
    existing = {e.event for e in timeline.events()}
    css = payload.css_snapshot or {}
    banner = css.get("banner")
    if (
        TimelineEventName.BANNER_FOUND not in existing
        and isinstance(banner, dict)
        and (banner.get("width") or banner.get("xpath"))
    ):
        timeline.mark(TimelineEventName.BANNER_FOUND)
    cmp_meta = css.get("cmp") or css.get("cmpMetadata") or {}
    if (
        TimelineEventName.CMP_FOUND not in existing
        and isinstance(cmp_meta, dict)
        and (cmp_meta.get("name") or cmp_meta.get("detected"))
    ):
        timeline.mark(TimelineEventName.CMP_FOUND, metadata={"cmp": str(cmp_meta.get("name", "detected"))})


class InferencePipeline:
    """
    Chrome Extension payload → Rules + Vision stub + Text stub → Fusion → Explanation.
    """

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry.default()
        self.rules = RuleEngine()
        self.vision = self.registry.vision
        self.text = self.registry.text
        self.fusion = FusionEngine()
        self.explainer = ExplanationGenerator()

    def run(
        self,
        payload: ScanPayload,
        *,
        timeline: TimelineRecorder | None = None,
        metrics: PerformanceCollector | None = None,
    ) -> ExplainableReport:
        tl = timeline or TimelineRecorder()
        perf = metrics or PerformanceInstrumentation()

        if not any(e.event == TimelineEventName.PAGE_LOADED for e in tl.events()):
            tl.mark(
                TimelineEventName.PAGE_LOADED,
                at=payload.collected_at,
                duration_ms=0.0,
            )
        _detect_banner_cmp(payload, tl)

        with perf.measure(MetricStage.RULE_ENGINE):
            rule_result = self.rules.evaluate_payload(payload)
        tl.mark(TimelineEventName.RULE_ENGINE, metadata={"hits": len(rule_result.hits)})

        with perf.measure("vision"):
            vision_features = self.vision.extract_features(payload)
        with perf.measure("text"):
            text_pred = self.text.classify(payload)

        with perf.measure(MetricStage.FUSION):
            fusion_out = self.fusion.fuse(
                FusionInput(
                    vision=vision_features,
                    text=text_pred,
                    rules=rule_result,
                    dom_features=payload.css_snapshot or {},
                )
            )
        tl.mark(TimelineEventName.FUSION, metadata={"risk": fusion_out.risk_score})

        with perf.measure(MetricStage.REPORT):
            report = self.explainer.generate(
                fusion_out,
                rules=rule_result,
                vision=vision_features,
                text=text_pred,
                payload=payload,
            )
            report = enrich_report(report, payload=payload, fusion=fusion_out)
        tl.mark(TimelineEventName.EXPLANATION, metadata={"evidence_count": len(report.evidence)})

        perf.set(MetricStage.LLM, 0.0)

        perf_payload = perf.finalize() if isinstance(perf, PerformanceInstrumentation) else perf.to_dict()

        return report.model_copy(
            update={
                "timeline": tl.to_list(),
                "performance": perf_payload,
            }
        )
