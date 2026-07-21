from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str | None = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: UUID
    email: EmailStr
    full_name: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ScanCreate(BaseModel):
    url: str
    title: str | None = None
    html: str | None = None
    css_snapshot: dict | None = None
    visible_text: str | None = None
    screenshot_base64: str | None = None
    viewport: dict[str, int] | None = None
    scroll_position: dict[str, int] | None = None
    collected_at: str | None = None
    collection_duration_ms: float | None = None


class ScanOut(BaseModel):
    id: UUID
    url: str
    title: str | None
    status: str
    created_at: datetime
    completed_at: datetime | None = None

    model_config = {"from_attributes": True}


class ReportOut(BaseModel):
    scan_id: UUID
    risk_score: float
    category: str
    confidence: float
    confidence_breakdown: dict | None = None
    evidence: list[dict]
    rule_traces: list[dict] = []
    severity: dict | None = None
    pattern_clusters: list[dict] = []
    accessibility: dict | None = None
    timeline: list[dict] = []
    performance: dict | None = None
    vision: dict | None = None
    text: dict | None = None
    rules: dict | None = None
    fusion: dict | None = None
    pipeline_notes: list[str] = []
    screenshot_path: str | None = None
    annotated_screenshot_path: str | None = None
    export_paths: dict[str, str] | None = None


class FeedbackCreate(BaseModel):
    scan_id: UUID
    label: str
    comment: str | None = None


class FeedbackOut(BaseModel):
    id: UUID
    scan_id: UUID
    label: str
    comment: str | None
    created_at: datetime

    model_config = {"from_attributes": True}
