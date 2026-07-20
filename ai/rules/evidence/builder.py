"""Evidence helpers for rule outputs."""

from __future__ import annotations

from ai.common.types import EvidenceItem, RuleResult


def evidence_from_rules(rules: RuleResult) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for hit in rules.hits:
        items.append(
            EvidenceItem(
                id=f"rule:{hit.rule_id}",
                statement=hit.evidence,
                severity=hit.severity,
                source="rules",
                rule_id=hit.rule_id,
                metadata=hit.metadata,
            )
        )
    return items
