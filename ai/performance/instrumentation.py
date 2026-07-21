"""Extended performance instrumentation and benchmarking."""

from __future__ import annotations

import statistics
import time
from typing import Any

from ai.common.types import ScanPayload
from ai.performance.metrics import MetricStage, PerformanceCollector

PIPELINE_VERSION = "0.5.0"


class PerformanceInstrumentation(PerformanceCollector):
    """Performance collector with span summaries and percent-of-total."""

    def __init__(self) -> None:
        super().__init__()
        self._spans: list[dict[str, Any]] = []

    def record_span(self, name: str, duration_ms: float, **metadata: Any) -> None:
        self._spans.append(
            {
                "name": name,
                "duration_ms": round(max(duration_ms, 0.0), 2),
                **metadata,
            }
        )

    def finalize(self) -> dict[str, Any]:
        total = self.finish_total() or self._values.get(MetricStage.TOTAL.value, 0.0)
        stages = dict(self._values)
        percentages: dict[str, float] = {}
        if total > 0:
            for key, ms in stages.items():
                if key != MetricStage.TOTAL.value:
                    percentages[key] = round((ms / total) * 100.0, 2)

        accounted = sum(stages.get(s.value, 0.0) for s in MetricStage if s != MetricStage.TOTAL)
        return {
            **self.to_dict(),
            "pipeline_version": PIPELINE_VERSION,
            "spans": list(self._spans),
            "percent_of_total": percentages,
            "accounted_ms": round(accounted, 2),
            "unaccounted_ms": round(max(total - accounted, 0.0), 2),
        }


def benchmark_pipeline(
    payload: ScanPayload,
    *,
    runs: int = 3,
    registry=None,
) -> dict[str, Any]:
    """Run pipeline multiple times; return latency stats per stage."""
    from ai.inference.pipeline import InferencePipeline

    pipe = InferencePipeline(registry=registry)
    totals: list[float] = []
    stage_samples: dict[str, list[float]] = {}

    for _ in range(runs):
        perf = PerformanceInstrumentation()
        perf.start_total()
        pipe.run(payload, metrics=perf)
        finalized = perf.finalize()
        totals.append(finalized.get("total_ms", 0.0))
        for key, ms in (finalized.get("stages") or {}).items():
            stage_samples.setdefault(key, []).append(float(ms))

    def _stats(values: list[float]) -> dict[str, float]:
        if not values:
            return {"min": 0.0, "max": 0.0, "mean": 0.0, "p95": 0.0}
        ordered = sorted(values)
        p95_idx = min(len(ordered) - 1, int(len(ordered) * 0.95))
        return {
            "min": round(min(values), 2),
            "max": round(max(values), 2),
            "mean": round(statistics.mean(values), 2),
            "p95": round(ordered[p95_idx], 2),
        }

    return {
        "runs": runs,
        "pipeline_version": PIPELINE_VERSION,
        "total_ms": _stats(totals),
        "stages": {k: _stats(v) for k, v in stage_samples.items()},
        "benchmark_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
