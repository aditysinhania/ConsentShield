"""Training run logger (JSONL + console)."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _gpu_usage() -> dict[str, Any] | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return {
        "device": device,
        "name": torch.cuda.get_device_name(device),
        "memory_allocated_mb": round(torch.cuda.memory_allocated(device) / (1024**2), 2),
        "memory_reserved_mb": round(torch.cuda.memory_reserved(device) / (1024**2), 2),
        "max_memory_allocated_mb": round(torch.cuda.max_memory_allocated(device) / (1024**2), 2),
    }


class TrainingLogger:
    """Write epoch metrics to logs/ as JSONL and optional summary JSON."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        run_name: str = "run",
        also_print: bool = True,
    ) -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.also_print = also_print
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{run_name}_{stamp}"
        self.jsonl_path = self.log_dir / f"{self.run_id}.jsonl"
        self.summary_path = self.log_dir / f"{self.run_id}_summary.json"
        self.history: list[dict[str, Any]] = []
        self._epoch_start: float | None = None

    def log_event(self, event: str, payload: dict[str, Any] | None = None) -> None:
        row = {
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            **(payload or {}),
        }
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
            f.flush()
        if self.also_print:
            print(f"[{event}] {json.dumps(payload or {}, default=str)}", flush=True)

    def epoch_start(self, epoch: int) -> None:
        self._epoch_start = time.perf_counter()
        self.log_event("epoch_start", {"epoch": epoch})

    def epoch_end(
        self,
        epoch: int,
        *,
        train_loss: float | None = None,
        val_loss: float | None = None,
        metrics: dict[str, float] | None = None,
        learning_rate: float | None = None,
    ) -> dict[str, Any]:
        elapsed = None if self._epoch_start is None else time.perf_counter() - self._epoch_start
        row: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "metrics": metrics or {},
            "learning_rate": learning_rate,
            "epoch_time_sec": None if elapsed is None else round(elapsed, 3),
            "gpu": _gpu_usage(),
        }
        self.history.append(row)
        self.log_event("epoch_end", row)
        self.summary_path.write_text(json.dumps({"run_id": self.run_id, "history": self.history}, indent=2), encoding="utf-8")
        return row
