"""Group rule findings into Phase 3.1 pattern clusters."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field

from ai.common.types import EvidenceItem


class PatternClusterName(str, Enum):
    VISUAL_MANIPULATION = "Visual Manipulation"
    HIDDEN_CHOICE = "Hidden Choice"
    OBSTRUCTION = "Obstruction"
    FORCED_ACTION = "Forced Action"
    INTERFACE_INTERFERENCE = "Interface Interference"
    PRIVACY_FRICTION = "Privacy Friction"


_RULE_CLUSTER_MAP: dict[str, PatternClusterName] = {
    "cookie.accept_visually_dominant": PatternClusterName.VISUAL_MANIPULATION,
    "cookie.unequal_emphasis": PatternClusterName.VISUAL_MANIPULATION,
    "cookie.hidden_reject": PatternClusterName.HIDDEN_CHOICE,
    "cookie.multi_click_reject": PatternClusterName.PRIVACY_FRICTION,
    "cookie.banner_obstruction": PatternClusterName.OBSTRUCTION,
    "cookie.preselected_marketing": PatternClusterName.FORCED_ACTION,
    "sub.preselected_plan": PatternClusterName.FORCED_ACTION,
    "sub.auto_renew": PatternClusterName.PRIVACY_FRICTION,
    "sub.misleading_free_trial": PatternClusterName.PRIVACY_FRICTION,
    "sub.hidden_billing": PatternClusterName.INTERFACE_INTERFERENCE,
    "sub.small_font_disclaimer": PatternClusterName.INTERFACE_INTERFERENCE,
    "sub.confirmshaming": PatternClusterName.INTERFACE_INTERFERENCE,
}


_CLUSTER_ORDER = list(PatternClusterName)


class PatternCluster(BaseModel):
    cluster: PatternClusterName
    finding_ids: list[str] = Field(default_factory=list)
    rule_ids: list[str] = Field(default_factory=list)
    count: int = 0
    max_severity: float = Field(ge=0.0, le=1.0, default=0.0)


def cluster_for_rule(rule_id: str | None) -> PatternClusterName | None:
    if not rule_id:
        return None
    return _RULE_CLUSTER_MAP.get(rule_id)


def cluster_evidence(evidence: list[EvidenceItem]) -> list[PatternCluster]:
    buckets: dict[PatternClusterName, PatternCluster] = {}

    for item in evidence:
        if item.source != "rules" or not item.rule_id:
            continue
        cluster_name = cluster_for_rule(item.rule_id)
        if cluster_name is None:
            continue
        if cluster_name not in buckets:
            buckets[cluster_name] = PatternCluster(cluster=cluster_name)
        bucket = buckets[cluster_name]
        bucket.finding_ids.append(item.id)
        if item.rule_id not in bucket.rule_ids:
            bucket.rule_ids.append(item.rule_id)
        bucket.count += 1
        bucket.max_severity = max(bucket.max_severity, float(item.severity))

    return [buckets[name] for name in _CLUSTER_ORDER if name in buckets]
