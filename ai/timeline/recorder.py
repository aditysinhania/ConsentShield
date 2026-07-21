"""Pipeline event timeline with ISO timestamps."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class TimelineEventName(str, Enum):
    PAGE_LOADED = "page_loaded"
    BANNER_FOUND = "banner_found"
    CMP_FOUND = "cmp_found"
    RULE_ENGINE = "rule_engine"
    FUSION = "fusion"
    EXPLANATION = "explanation"
    REPORT = "report"


class TimelineEvent(BaseModel):
    event: TimelineEventName
    timestamp: str
    duration_ms: float | None = None
    metadata: dict[str, str | float | int | bool] = Field(default_factory=dict)


class TimelineRecorder:
    """Records ordered pipeline events."""

    def __init__(self, *, start_time: datetime | None = None) -> None:
        self._start = start_time or datetime.now(UTC)
        self._events: list[TimelineEvent] = []
        self._last_mark = self._start

    @property
    def start_time(self) -> datetime:
        return self._start

    def mark(
        self,
        name: TimelineEventName | str,
        *,
        at: datetime | str | None = None,
        duration_ms: float | None = None,
        metadata: dict | None = None,
    ) -> TimelineEvent:
        if isinstance(name, str):
            name = TimelineEventName(name)
        if isinstance(at, str):
            ts = at
            now = datetime.fromisoformat(at.replace("Z", "+00:00"))
        elif isinstance(at, datetime):
            now = at if at.tzinfo else at.replace(tzinfo=UTC)
            ts = now.isoformat().replace("+00:00", "Z")
        else:
            now = datetime.now(UTC)
            ts = now.isoformat().replace("+00:00", "Z")

        if duration_ms is None and self._events:
            delta = (now - self._last_mark).total_seconds() * 1000.0
            duration_ms = round(max(delta, 0.0), 2)
        elif duration_ms is None and name == TimelineEventName.PAGE_LOADED:
            duration_ms = 0.0

        event = TimelineEvent(
            event=name,
            timestamp=ts,
            duration_ms=duration_ms,
            metadata=metadata or {},
        )
        self._events.append(event)
        self._last_mark = now
        return event

    def to_list(self) -> list[dict]:
        return [e.model_dump() for e in self._events]

    def events(self) -> list[TimelineEvent]:
        return list(self._events)
