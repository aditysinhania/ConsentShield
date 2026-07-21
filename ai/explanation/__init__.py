"""Explainability module — evidence-backed reports only."""

from ai.explanation.catalog import all_rule_explanations, explanation_for_rule
from ai.explanation.generator import ExplanationGenerator

__all__ = ["ExplanationGenerator", "explanation_for_rule", "all_rule_explanations"]
