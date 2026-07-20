"""Dataset schemas for the custom multimodal corpus."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ai.common.types import Category


class MultimodalSample(BaseModel):
    id: str
    url: str
    screenshot_path: str | None = None
    html_path: str | None = None
    css_path: str | None = None
    ocr_text: str | None = None
    visible_text: str | None = None
    label: Category
    verified_by: str | None = None
    notes: str | None = None
    metadata: dict = Field(default_factory=dict)
