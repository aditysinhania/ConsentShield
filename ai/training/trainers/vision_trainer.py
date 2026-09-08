"""Vision trainer — CLIP fine-tuning (Phase 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ai.training.callbacks.checkpoint import load_checkpoint, save_checkpoint
from ai.training.models.clip_classifier import CLIPClassifier
from ai.training.trainers.base_trainer import BaseTrainer


class VisionTrainer(BaseTrainer):
    """
    Fine-tunes openai/clip-vit-base-patch32 vision encoder + classification head.

    Expects collated batches from ``collate_image_batch`` with ``pixel_values``
    and ``label_ids``.
    """

    task_name = "vision"

    def __init__(self, config: dict[str, Any], *, allow_train: bool = True) -> None:
        super().__init__(config, allow_train=allow_train)
        model_cfg = config.get("model") or {}
        self.model_name = str(model_cfg.get("name", "openai/clip-vit-base-patch32"))
        self.image_size = int(model_cfg.get("image_size", 224))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.freeze_text_encoder = bool(model_cfg.get("freeze_text_encoder", True))
        self.freeze_projection = bool(model_cfg.get("freeze_projection", True))
        self.use_amp = bool((config.get("train") or {}).get("mixed_precision", True)) and (
            str(self.device).startswith("cuda")
        )
        self.accum_steps = max(1, int((config.get("train") or {}).get("gradient_accumulation_steps", 1)))
        self.scaler = None
        if self.use_amp:
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self._last_val_details: dict[str, Any] = {}
        self._global_step = 0
        self._accum_counter = 0

    def build_model(self, num_labels: int) -> CLIPClassifier:
        model = CLIPClassifier(
            self.model_name,
            num_labels=num_labels,
            dropout=self.dropout,
            freeze_text_encoder=self.freeze_text_encoder,
            freeze_projection=self.freeze_projection,
        )
        counts = model.trainable_parameter_counts()
        self.logger.log_event("clip_params", counts)
        return model

    def _forward(self, batch: dict[str, Any]) -> dict[str, Any]:
        assert self.model is not None
        pixel_values = batch["pixel_values"].to(self.device)
        labels = torch.tensor(batch["label_ids"], dtype=torch.long, device=self.device)
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(pixel_values=pixel_values, labels=labels)
        else:
            out = self.model(pixel_values=pixel_values, labels=labels)
        out["_labels"] = labels
        return out

    def training_step(self, batch: Any) -> tuple[Any, float]:
        out = self._forward(batch)
        loss = out["loss"]
        if self.accum_steps > 1:
            loss = loss / self.accum_steps
        return loss, float(out["loss"].detach().item())

    def validation_step(self, batch: Any) -> dict[str, Any]:
        out = self._forward(batch)
        labels = out["_labels"]
        preds = torch.argmax(out["logits"], dim=-1)
        return {
            "loss": float(out["loss"].detach().item()) if out["loss"] is not None else 0.0,
            "y_true": labels.detach().cpu().tolist(),
            "y_pred": preds.detach().cpu().tolist(),
            "ids": [str(x) for x in batch.get("ids", [])],
            "probs": out["probs"].detach().cpu(),
        }

    def predict_batch(self, batch: Any) -> tuple[list[Any], list[Any], list[str]]:
        result = self.validation_step(batch)
        if isinstance(batch, dict):
            batch["_loss"] = result["loss"]
        return result["y_true"], result["y_pred"], result["ids"]

    def save_checkpoint(
        self,
        path: str | Path,
        *,
        epoch: int = 0,
        metrics: dict[str, float] | None = None,
        best_metric: float | None = None,
    ) -> Path:
        return save_checkpoint(
            path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=epoch,
            global_step=self._global_step,
            best_metric=best_metric,
            metrics=metrics or {},
            config=self.config,
            label_vocab=self.label_vocab,
        )

    def load_checkpoint(self, path: str | Path, *, map_location: str | None = None) -> dict[str, Any]:
        ckpt = load_checkpoint(path, map_location=map_location or self.device)
        if self.model is not None and "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
        if self.optimizer is not None and "optimizer_state_dict" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if self.scheduler is not None and "scheduler_state_dict" in ckpt:
            try:
                self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception:
                pass
        if ckpt.get("label_vocab"):
            self.label_vocab = dict(ckpt["label_vocab"])
        self._global_step = int(ckpt.get("global_step") or 0)
        return ckpt

    def _run_train_epoch(self, loader: Any) -> float:
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        total = 0.0
        n = 0
        max_norm = (self.config.get("train") or {}).get("max_grad_norm")
        log_every = int((self.config.get("train") or {}).get("log_every_steps", 20))
        total_steps = max(1, len(loader))
        self.optimizer.zero_grad(set_to_none=True)
        self._accum_counter = 0

        for batch in loader:
            loss, loss_value = self.training_step(batch)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            self._accum_counter += 1
            step_now = self._accum_counter >= self.accum_steps
            if step_now:
                if self.scaler is not None:
                    if max_norm is not None:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    if max_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                self._accum_counter = 0
                self._global_step += 1

            total += loss_value
            n += 1
            if log_every > 0 and (n % log_every == 0 or n == total_steps):
                self.logger.log_event(
                    "train_step",
                    {
                        "step": n,
                        "total_steps": total_steps,
                        "loss": round(loss_value, 6),
                        "avg_loss": round(total / n, 6),
                    },
                )

        # Flush leftover grads
        if self._accum_counter > 0:
            if self.scaler is not None:
                if max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._global_step += 1
            self._accum_counter = 0

        return total / max(1, n)

    def evaluate_detailed(self, loader: Any) -> dict[str, Any]:
        from ai.training.metrics.classification import compute_classification_metrics
        from ai.training.metrics.confusion import confusion_matrix_dict

        assert self.model is not None
        self.model.eval()
        y_true: list[Any] = []
        y_pred: list[Any] = []
        ids: list[str] = []
        losses: list[float] = []
        with torch.no_grad():
            for batch in loader:
                result = self.validation_step(batch)
                y_true.extend(result["y_true"])
                y_pred.extend(result["y_pred"])
                ids.extend(result["ids"])
                losses.append(float(result["loss"]))

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
        metrics["f1"] = float(metrics.get("macro_f1") or metrics.get("f1") or 0.0)
        id2label = {i: lab for lab, i in self.label_vocab.items()}
        label_ids = sorted(self.label_vocab.values())
        cm = confusion_matrix_dict(
            y_true,
            y_pred,
            labels=label_ids,
            label_names=[id2label[i] for i in label_ids],
        )
        details = {
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "ids": ids,
            "confusion_matrix": cm,
            "id2label": id2label,
        }
        self._last_val_details = details
        return details
