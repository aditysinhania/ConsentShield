"""Performance metrics for scan pipeline stages."""

from __future__ import annotations

import time
from contextlib import contextmanager
from enum import Enum

from pydantic import BaseModel, Field


class MetricStage(str, Enum):
    COLLECTION = "collection"
    RULE_ENGINE = "rule_engine"
    FUSION = "fusion"
    SCREENSHOT = "screenshot"
    ANNOTATION = "annotation"
    REPORT = "report"
    LLM = "llm"
    TOTAL = "total"


class PerformanceMetrics(BaseModel):
    collection_ms: float = 0.0
    rule_engine_ms: float = 0.0
    fusion_ms: float = 0.0
    screenshot_ms: float = 0.0
    annotation_ms: float = 0.0
    report_ms: float = 0.0
    llm_ms: float = 0.0
    total_ms: float = 0.0
    stages: dict[str, float] = Field(default_factory=dict)


class PerformanceCollector:
    """Collect per-stage millisecond timings."""

    def __init__(self) -> None:
        self._values: dict[str, float] = {}
        self._total_start: float | None = None

    def start_total(self) -> None:
        self._total_start = time.perf_counter()

    def finish_total(self) -> float:
        if self._total_start is None:
            return 0.0
        elapsed = (time.perf_counter() - self._total_start) * 1000.0
        self._values[MetricStage.TOTAL.value] = round(elapsed, 2)
        return self._values[MetricStage.TOTAL.value]

    def set(self, stage: MetricStage | str, ms: float) -> None:
        key = stage.value if isinstance(stage, MetricStage) else stage
        self._values[key] = round(max(ms, 0.0), 2)

    @contextmanager
    def measure(self, stage: MetricStage | str):
        key = stage.value if isinstance(stage, MetricStage) else stage
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.set(key, (time.perf_counter() - t0) * 1000.0)

    def to_model(self) -> PerformanceMetrics:
        return PerformanceMetrics(
            collection_ms=self._values.get(MetricStage.COLLECTION.value, 0.0),
            rule_engine_ms=self._values.get(MetricStage.RULE_ENGINE.value, 0.0),
            fusion_ms=self._values.get(MetricStage.FUSION.value, 0.0),
            screenshot_ms=self._values.get(MetricStage.SCREENSHOT.value, 0.0),
            annotation_ms=self._values.get(MetricStage.ANNOTATION.value, 0.0),
            report_ms=self._values.get(MetricStage.REPORT.value, 0.0),
            llm_ms=self._values.get(MetricStage.LLM.value, 0.0),
            total_ms=self._values.get(MetricStage.TOTAL.value, 0.0),
            stages=dict(self._values),
        )

    def to_dict(self) -> dict:
        return self.to_model().model_dump()
