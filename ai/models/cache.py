"""Process-wide lazy model cache — load once, reuse."""

from __future__ import annotations

import threading
import time
from typing import Any

_lock = threading.Lock()
_store: dict[str, Any] = {}
_timings: dict[str, float] = {}


def get_cached(key: str) -> Any | None:
    with _lock:
        return _store.get(key)


def set_cached(key: str, value: Any, *, load_ms: float = 0.0) -> Any:
    with _lock:
        _store[key] = value
        if load_ms:
            _timings[key] = load_ms
        return value


def get_load_ms(key: str) -> float:
    with _lock:
        return float(_timings.get(key, 0.0))


def clear_cache() -> None:
    with _lock:
        _store.clear()
        _timings.clear()


def timed_load(key: str, factory):
    """Load via factory if missing; return (value, load_ms, cache_hit)."""
    existing = get_cached(key)
    if existing is not None:
        return existing, get_load_ms(key), True
    with _lock:
        existing = _store.get(key)
        if existing is not None:
            return existing, float(_timings.get(key, 0.0)), True
        t0 = time.perf_counter()
        value = factory()
        load_ms = (time.perf_counter() - t0) * 1000.0
        _store[key] = value
        _timings[key] = load_ms
        return value, load_ms, False
