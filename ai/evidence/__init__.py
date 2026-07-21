"""Evidence package — enrichment of EvidenceItem with real DOM context."""

from ai.evidence.enrichment import enrich_evidence_item
from ai.evidence.quality import apply_evidence_quality, score_evidence_item

__all__ = ["enrich_evidence_item", "apply_evidence_quality", "score_evidence_item"]
