"""Text trainer — MiniLM fine-tuning (Phase 2)."""

from __future__ import annotations

from typing import Any

import torch

from ai.training.models.minilm_classifier import MiniLMClassifier, build_tokenizer, tokenize_batch
from ai.training.trainers.base_trainer import BaseTrainer


class TextTrainer(BaseTrainer):
    """
    Fine-tunes sentence-transformers/all-MiniLM-L6-v2 with a classification head.

    Expects collated batches from ``collate_text_batch``:
    ``texts``, ``label_ids``, ``ids``.
    """

    task_name = "text"

    def __init__(self, config: dict[str, Any], *, allow_train: bool = True) -> None:
        super().__init__(config, allow_train=allow_train)
        model_cfg = config.get("model") or {}
        self.model_name = str(model_cfg.get("name", "sentence-transformers/all-MiniLM-L6-v2"))
        self.max_length = int(model_cfg.get("max_length", 256))
        self.dropout = float(model_cfg.get("dropout", 0.1))
        self.freeze_encoder = bool(model_cfg.get("freeze_encoder", False))
        self.tokenizer = build_tokenizer(self.model_name, max_length=self.max_length)
        self.use_amp = bool((config.get("train") or {}).get("mixed_precision", True)) and (
            str(self.device).startswith("cuda")
        )
        self.scaler = None
        if self.use_amp:
            # torch.amp.GradScaler is preferred on recent PyTorch builds
            try:
                self.scaler = torch.amp.GradScaler("cuda", enabled=True)
            except Exception:
                self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.criterion = torch.nn.CrossEntropyLoss()
        self._last_val_details: dict[str, Any] = {}

    def build_model(self, num_labels: int) -> MiniLMClassifier:
        return MiniLMClassifier(
            self.model_name,
            num_labels=num_labels,
            dropout=self.dropout,
            freeze_encoder=self.freeze_encoder,
        )

    def _encode(self, batch: dict[str, Any]) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        texts = list(batch["texts"])
        labels = torch.tensor(batch["label_ids"], dtype=torch.long, device=self.device)
        encoded = tokenize_batch(
            self.tokenizer,
            texts,
            max_length=self.max_length,
            device=self.device,
        )
        return encoded, labels

    def training_step(self, batch: Any) -> tuple[Any, float]:
        assert self.model is not None
        encoded, labels = self._encode(batch)
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=labels,
                )
                loss = out["loss"]
        else:
            out = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )
            loss = out["loss"]
        return loss, float(loss.detach().item())

    def predict_batch(self, batch: Any) -> tuple[list[Any], list[Any], list[str]]:
        assert self.model is not None
        encoded, labels = self._encode(batch)
        if self.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                out = self.model(
                    input_ids=encoded["input_ids"],
                    attention_mask=encoded["attention_mask"],
                    labels=labels,
                )
        else:
            out = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            )
        logits = out["logits"]
        preds = torch.argmax(logits, dim=-1)
        # Stash loss for evaluate()
        if isinstance(batch, dict):
            batch["_loss"] = float(out["loss"].detach().item()) if out["loss"] is not None else 0.0
        ids = [str(x) for x in batch.get("ids", [])]
        return labels.detach().cpu().tolist(), preds.detach().cpu().tolist(), ids

    def _run_train_epoch(self, loader: Any) -> float:
        """Override to support mixed-precision GradScaler + step progress logs."""
        assert self.model is not None and self.optimizer is not None
        self.model.train()
        total = 0.0
        n = 0
        max_norm = (self.config.get("train") or {}).get("max_grad_norm")
        log_every = int((self.config.get("train") or {}).get("log_every_steps", 50))
        total_steps = max(1, len(loader))
        for batch in loader:
            self.optimizer.zero_grad(set_to_none=True)
            loss, loss_value = self.training_step(batch)
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                if max_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                if max_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(max_norm))
                self.optimizer.step()
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
        return total / max(1, n)

    def evaluate_detailed(self, loader: Any) -> dict[str, Any]:
        """Full validation pass returning metrics + predictions."""
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
                yt, yp, batch_ids = self.predict_batch(batch)
                y_true.extend(yt)
                y_pred.extend(yp)
                ids.extend(batch_ids)
                if isinstance(batch, dict) and "_loss" in batch:
                    losses.append(float(batch["_loss"]))

        metrics = compute_classification_metrics(y_true, y_pred)
        metrics["loss"] = sum(losses) / len(losses) if losses else 0.0
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
