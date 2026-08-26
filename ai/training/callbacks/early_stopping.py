"""Early stopping callback."""

from __future__ import annotations

from typing import Literal


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(
        self,
        *,
        monitor: str = "val_loss",
        mode: Literal["min", "max"] = "min",
        patience: int = 3,
        min_delta: float = 0.0,
    ) -> None:
        self.monitor = monitor
        self.mode = mode
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.best: float | None = None
        self.bad_epochs = 0
        self.should_stop = False

    def step(self, metrics: dict[str, float]) -> bool:
        if self.monitor not in metrics:
            return False
        value = float(metrics[self.monitor])
        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        improved = (
            value < self.best - self.min_delta
            if self.mode == "min"
            else value > self.best + self.min_delta
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        if self.bad_epochs >= self.patience:
            self.should_stop = True
        return self.should_stop

    def state_dict(self) -> dict:
        return {
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "best": self.best,
            "bad_epochs": self.bad_epochs,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.best = state.get("best")
        self.bad_epochs = int(state.get("bad_epochs", 0))
        self.should_stop = bool(state.get("should_stop", False))
