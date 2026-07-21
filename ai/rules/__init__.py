"""Rule engine — configurable, evidence-producing consent dark-pattern rules."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from ai.common.base import BaseDetector
from ai.common.types import (
    EvalMetrics,
    PredictionResult,
    RuleHit,
    RuleResult,
    ScanPayload,
    TrainResult,
)
from ai.rules.evaluators.registry import EVALUATORS
from ai.rules.scoring.scorer import score_hits
from ai.rules.traceability import attach_traces

CONFIG_DIR = Path(__file__).parent / "config"


class RuleEngine(BaseDetector[ScanPayload, PredictionResult]):
    """Evaluates DOM/CSS/text payloads against YAML-configured rules."""

    name = "rule_engine"
    version = "0.1.0"

    def __init__(self, config_paths: list[Path] | None = None) -> None:
        self._rules: list[dict[str, Any]] = []
        paths = config_paths or sorted(CONFIG_DIR.glob("*.yaml"))
        for path in paths:
            self._load_config(path)

    def _load_config(self, path: Path) -> None:
        with path.open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        for rule in data.get("rules", []):
            self._rules.append(rule)

    def list_rules(self) -> list[dict[str, Any]]:
        return list(self._rules)

    def evaluate_payload(self, payload: ScanPayload) -> RuleResult:
        hits: list[RuleHit] = []
        context = {
            "url": payload.url,
            "title": payload.title or "",
            "html": payload.html or "",
            "css": payload.css_snapshot or {},
            "text": payload.visible_text or "",
            "viewport": payload.viewport or {},
        }
        for rule in self._rules:
            evaluator_name = rule.get("evaluator")
            fn = EVALUATORS.get(evaluator_name)
            if fn is None:
                continue
            result = fn(context, rule)
            if result and result.get("triggered"):
                hits.append(
                    RuleHit(
                        rule_id=rule["id"],
                        name=rule["name"],
                        category=rule["category"],
                        severity=float(rule.get("severity", 0.5)),
                        score=float(result.get("score", rule.get("severity", 0.5))),
                        evidence=str(result.get("evidence", rule.get("description", ""))),
                        metadata=result.get("metadata", {}),
                        features_used=list(result.get("features_used") or []),
                        visual_score=float(result.get("visual_score") or 0.0),
                        text_score=float(result.get("text_score") or 0.0),
                        layout_score=float(result.get("layout_score") or 0.0),
                    )
                )
        return attach_traces(score_hits(hits))

    def train(self, dataset: Any, **kwargs: Any) -> TrainResult:
        return TrainResult(
            status="not_applicable",
            message="Rule engine is configuration-driven; use YAML to extend rules.",
        )

    def predict(self, input_data: ScanPayload) -> PredictionResult:
        result = self.evaluate_payload(input_data)
        return PredictionResult(status="ready", data=result.model_dump())

    def evaluate(self, dataset: Any) -> EvalMetrics:
        return EvalMetrics(
            status="not_implemented",
            message="Provide labeled scan fixtures to evaluate rule precision/recall.",
        )

    def load_model(self, path: str) -> None:
        self._rules = []
        p = Path(path)
        if p.is_dir():
            for yaml_path in sorted(p.glob("*.yaml")):
                self._load_config(yaml_path)
        else:
            self._load_config(p)

    def save_model(self, path: str) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"rules": self._rules}, f, sort_keys=False)

    def is_ready(self) -> bool:
        return len(self._rules) > 0
