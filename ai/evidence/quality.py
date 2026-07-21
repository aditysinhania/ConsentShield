"""Evidence completeness and audit quality scoring."""

from __future__ import annotations

from ai.common.types import EvidenceItem, ScanPayload
from ai.evidence.enrichment import enrich_evidence_item


_ENRICHMENT_FIELDS = (
    "xpath",
    "css_selector",
    "dom_path",
    "bounding_rect",
    "html_snippet",
    "computed_styles",
    "viewport",
    "scroll_position",
    "timestamp",
    "url",
    "page_title",
)
_EXPLANATION_FIELDS = ("explanation", "user_impact", "gdpr_relevance", "recommendation")


def score_evidence_item(item: EvidenceItem) -> float:
    """0–1 quality score from enrichment + explanation completeness."""
    if item.source != "rules":
        return 0.5
    enrich_present = sum(1 for f in _ENRICHMENT_FIELDS if getattr(item, f, None))
    enrich_score = min(1.0, enrich_present / max(len(_ENRICHMENT_FIELDS) * 0.45, 1))
    expl_present = sum(1 for f in _EXPLANATION_FIELDS if getattr(item, f, None))
    expl_score = expl_present / len(_EXPLANATION_FIELDS)
    meta = dict(item.metadata or {})
    if meta.get("accept_text") or meta.get("label") or meta.get("coverage"):
        enrich_score = min(1.0, enrich_score + 0.1)
    return round(0.55 * enrich_score + 0.45 * expl_score, 3)


def apply_evidence_quality(
    items: list[EvidenceItem],
    payload: ScanPayload | None = None,
) -> list[EvidenceItem]:
    """Attach quality_score and quality_tier to each evidence metadata."""
    out: list[EvidenceItem] = []
    for item in items:
        enriched = enrich_evidence_item(item, payload) if payload is not None else item
        score = score_evidence_item(enriched)
        tier = "high" if score >= 0.75 else "medium" if score >= 0.45 else "low"
        meta = dict(enriched.metadata or {})
        meta["quality_score"] = score
        meta["quality_tier"] = tier
        out.append(enriched.model_copy(update={"metadata": meta}))
    return out
