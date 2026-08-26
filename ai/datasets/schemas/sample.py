"""Dataset schemas for the custom multimodal corpus."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai.common.types import Category


class MultimodalSample(BaseModel):
    id: str
    url: str = ""
    screenshot_path: str | None = None
    html_path: str | None = None
    css_path: str | None = None
    ocr_text: str | None = None
    visible_text: str | None = None
    label: Category = Category.UNKNOWN
    verified_by: str | None = None
    notes: str | None = None
    metadata: dict = Field(default_factory=dict)


class UnifiedSample(BaseModel):
    """Schema for datasets/unified/*.jsonl rows."""

    id: str
    source: str
    modality: str
    image_path: str | None = None
    text: str | None = None
    ocr_text: str | None = None
    label_binary: str
    label_fine: str
    label_consentshield: str
    source_labels: list[str] = Field(default_factory=list)
    bboxes: list[dict] = Field(default_factory=list)
    components: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)

    def to_multimodal(self) -> MultimodalSample:
        try:
            label = Category(self.label_consentshield)
        except ValueError:
            label = Category.UNKNOWN
        return MultimodalSample(
            id=self.id,
            url=str(self.metadata.get("file_url") or self.metadata.get("site_name") or ""),
            screenshot_path=self.image_path,
            ocr_text=self.ocr_text,
            visible_text=self.text,
            label=label,
            notes=f"{self.source}:{self.label_fine}",
            metadata={
                **self.metadata,
                "label_binary": self.label_binary,
                "label_fine": self.label_fine,
                "source_labels": self.source_labels,
                "bboxes": self.bboxes,
                "components": self.components,
                "modality": self.modality,
                "source": self.source,
            },
        )
