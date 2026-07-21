"""Evidence helpers for rule outputs."""

from __future__ import annotations

from ai.common.types import EvidenceItem, RuleResult, ScanPayload
from ai.evidence.enrichment import enrich_evidence_item
from ai.explanation.catalog import explanation_for_rule


def evidence_from_rules(
    rules: RuleResult,
    payload: ScanPayload | None = None,
) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for hit in rules.hits:
        expl = explanation_for_rule(hit.rule_id, fallback_evidence=hit.evidence)
        item = EvidenceItem(
            id=f"rule:{hit.rule_id}",
            statement=hit.evidence,
            severity=hit.severity,
            source="rules",
            rule_id=hit.rule_id,
            metadata=hit.metadata,
            explanation=expl.explanation,
            user_impact=expl.user_impact,
            gdpr_relevance=expl.gdpr_relevance,
            recommendation=expl.recommendation,
        )
        if payload is not None:
            item = enrich_evidence_item(item, payload)
        items.append(item)
    return items
