"""Abstract base trainer — shared fit / eval / checkpoint loop."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ai.training.callbacks.checkpoint import CheckpointCallback
from ai.training.callbacks.early_stopping import EarlyStopping
from ai.training.callbacks.logging import TrainingLogger
from ai.training.metrics.classification import compute_classification_metrics
from ai.training.trainers.utils import (
    build_optimizer,
    build_scheduler,
    ensure_run_dirs,
    get_lr,
    resolve_device,
    set_seed,
)


class PhaseGateError(RuntimeError):
    """Raised when Phase 2+ model training is requested before it is enabled."""


class BaseTrainer(ABC):
    """
    Production training loop scaffolding.

    Subclasses implement ``build_model``, ``training_step``, and ``predict_batch``.
    Phase 1 ships TextTrainer / VisionTrainer stubs that refuse to train models
    until Phase 2/3 explicitly enable them.
    """

    task_name: str = "base"

    def __init__(self, config: dict[str, Any], *, allow_train: bool = False) -> None:
        self.config = config
        self.allow_train = allow_train
        self.device = resolve_device((config.get("hardware") or {}).get("device"))
        seed = int((config.get("reproducibility") or {}).get("seed", 42))
        set_seed(seed)
        self.seed = seed

        self.model: Any | None = None
        self.optimizer: Any | None = None
        self.scheduler: Any | None = None
        self.label_vocab: dict[str, int] = {}
        self.run_dirs = ensure_run_dirs(config)
        self.logger = TrainingLogger(
            self.run_dirs["logs"],
            run_name=str(config.get("run_name") or self.task_name),
        )
        es_cfg = config.get("early_stopping") or {}
        self.early_stopping = EarlyStopping(
            monitor=str(es_cfg.get("monitor", "val_loss")),
            mode=str(es_cfg.get("mode", "min")),  # type: ignore[arg-type]
            patience=int(es_cfg.get("patience", 3)),
            min_delta=float(es_cfg.get("min_delta", 0.0)),
        )
        ckpt_cfg = config.get("checkpoint") or {}
        self.checkpoint_cb = CheckpointCallback(
            self.run_dirs["models"],
            monitor=str(ckpt_cfg.get("monitor", "val_f1")),
            mode=str(ckpt_cfg.get("mode", "max")),
            filename_best=str(ckpt_cfg.get("best_name", "best.pt")),
            filename_latest=str(ckpt_cfg.get("latest_name", "latest.pt")),
        )

    @abstractmethod
    def build_model(self, num_labels: int) -> Any:
        raise NotImplementedError

    @abstractmethod
    def training_step(self, batch: Any) -> tuple[Any, float]:
        """Return (loss_tensor, loss_value)."""
        raise NotImplementedError

    @abstractmethod
    def predict_batch(self, batch: Any) -> tuple[list[Any], list[Any], list[str]]:
        """Return (y_true, y_pred, ids)."""
        raise NotImplementedError

    def _assert_trainable(self) -> None:
        if not self.allow_train:
            raise PhaseGateError(
                "Model training is disabled in Phase 1. "
                "Re-run with --allow-train only after Phase 2/3 approval "
                f"(task={self.task_name})."
            )

    def setup(self, *, num_labels: int, label_vocab: dict[str, int]) -> None:
        self._assert_trainable()
        self.label_vocab = dict(label_vocab)
        self.model = self.build_model(num_labels)
        self.model.to(self.device)
        self.optimizer = build_optimizer(self.model, self.config)
        self.logger.log_event(
            "setup",
            {
                "device": self.device,
                "seed": self.seed,
                "num_labels": num_labels,
                "run_dirs": {k: str(v) for k, v in self.run_dirs.items()},
            },
        )

    def load_training_state(self, checkpoint_path: str | Path) -> int:
        """
        Restore model/optimizer/scheduler from a Phase 2 checkpoint.

        Returns the next epoch index to train (last completed epoch + 1).
        """
        from pathlib import Path

        from ai.training.callbacks.checkpoint import load_checkpoint, restore_rng_state

        self._assert_trainable()
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Call setup() before load_training_state().")

        ckpt = load_checkpoint(checkpoint_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt and self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("label_vocab"):
            self.label_vocab = dict(ckpt["label_vocab"])
        if ckpt.get("rng_state"):
            try:
                restore_rng_state(ckpt["rng_state"])
            except Exception:
                pass

        last_epoch = int(ckpt.get("epoch") or 0)
        metrics = ckpt.get("metrics") or {}
        monitor = self.checkpoint_cb.monitor
        if monitor in metrics:
            self.checkpoint_cb.best_metric = float(metrics[monitor])
        elif f"val_{monitor}" in metrics:
            self.checkpoint_cb.best_metric = float(metrics[f"val_{monitor}"])
        elif "val_f1" in metrics:
            self.checkpoint_cb.best_metric = float(metrics["val_f1"])

        self.logger.log_event(
            "resume",
            {
                "checkpoint": str(checkpoint_path),
                "last_epoch": last_epoch,
                "next_epoch": last_epoch + 1,
                "best_metric": self.checkpoint_cb.best_metric,
            },
        )
        self._resume_scheduler_state = ckpt.get("scheduler_state_dict")
        return last_epoch + 1

    def fit(
        self,
        train_loader: Any,
        val_loader: Any | None = None,
        *,
        start_epoch: int = 1,
    ) -> dict[str, Any]:
        self._assert_trainable()
        if self.model is None or self.optimizer is None:
            raise RuntimeError("Call setup() before fit().")

        train_cfg = self.config.get("train") or {}
        epochs = int(train_cfg.get("epochs", 1))
        start_epoch = max(1, int(start_epoch))
        if start_epoch > epochs:
            self.logger.log_event(
                "resume_skip",
                {"start_epoch": start_epoch, "epochs": epochs, "message": "Nothing left to train"},
            )
            return {"history": [], "run_dirs": {k: str(v) for k, v in self.run_dirs.items()}}

        self.scheduler = build_scheduler(
            self.optimizer,
            self.config,
            steps_per_epoch=max(1, len(train_loader)),
        )
        resume_sched = getattr(self, "_resume_scheduler_state", None)
        if self.scheduler is not None and resume_sched is not None:
            try:
                self.scheduler.load_state_dict(resume_sched)
            except Exception:
                pass
            self._resume_scheduler_state = None

        history: list[dict[str, Any]] = []
        for epoch in range(start_epoch, epochs + 1):
            self.logger.epoch_start(epoch)
            train_loss = self._run_train_epoch(train_loader)
            metrics: dict[str, float] = {"train_loss": train_loss}
            val_loss = None
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                val_loss = val_metrics.get("loss")
                metrics.update({f"val_{k}": v for k, v in val_metrics.items()})

            lr = get_lr(self.optimizer)
            row = self.logger.epoch_end(
                epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                metrics=metrics,
                learning_rate=lr,
            )
            history.append(row)

            self.checkpoint_cb.on_epoch_end(
                epoch=epoch,
                metrics=metrics,
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                config=self.config,
                label_vocab=self.label_vocab,
            )

            if self.scheduler is not None:
                self.scheduler.step()

            if self.early_stopping.step(metrics):
                self.logger.log_event("early_stop", {"epoch": epoch, "monitor": self.early_stopping.monitor})
                break

        return {"history": history, "run_dirs": {k: str(v) for k, v in self.run_dirs.items()}}

    def _run_train_epoch(self, loader: Any) -> float:
        import torch

        assert self.model is not None and self.optimizer is not None
        self.model.train()
        total = 0.0
        n = 0
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, loss_value = self.training_step(batch)
            loss.backward()
            max_norm = (self.config.get("train") or {}).get("max_grad_norm")
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
            self.optimizer.step()
            total += loss_value
            n += 1
        return total / max(1, n)

    def evaluate(self, loader: Any) -> dict[str, float]:
        self._assert_trainable()
        detailed = getattr(self, "evaluate_detailed", None)
        if callable(detailed):
            details = detailed(loader)
            return dict(details.get("metrics") or {})

        assert self.model is not None
        import torch

        self.model.eval()
        y_true: list[Any] = []
        y_pred: list[Any] = []
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                yt, yp, _ids = self.predict_batch(batch)
                y_true.extend(yt)
                y_pred.extend(yp)
                if isinstance(batch, dict) and "_loss" in batch:
                    losses.append(float(batch["_loss"]))

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
        return metrics
