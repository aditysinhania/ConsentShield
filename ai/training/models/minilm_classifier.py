"""MiniLM encoder + classification head for ConsentShield text training."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MiniLMClassifier(nn.Module):
    """
    MiniLM encoder → mean-pool → Dropout → Linear → logits.

    Softmax is applied at inference (``predict_proba``); training uses
    CrossEntropyLoss on logits. Configurable ``num_labels`` for binary or multiclass.

    ``freeze_layers`` freezes the first N encoder transformer layers (embeddings
    stay trainable unless ``freeze_encoder`` is True).
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        num_labels: int = 2,
        dropout: float = 0.1,
        freeze_encoder: bool = False,
        freeze_layers: int = 0,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = int(num_labels)
        self.freeze_layers = max(0, int(freeze_layers))
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = int(self.config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, self.num_labels)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
        elif self.freeze_layers > 0:
            self._freeze_bottom_layers(self.freeze_layers)

    def _freeze_bottom_layers(self, n: int) -> None:
        """Freeze the first ``n`` transformer layers of the MiniLM encoder."""
        layers = getattr(self.encoder, "encoder", None)
        layer_list = getattr(layers, "layer", None) if layers is not None else None
        if layer_list is None:
            return
        for i, layer in enumerate(layer_list):
            if i < n:
                for p in layer.parameters():
                    p.requires_grad = False

    @staticmethod
    def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pool(outputs.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        return {"loss": loss, "logits": logits, "probs": probs}

    def predict_proba(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(input_ids=input_ids, attention_mask=attention_mask)["probs"]


def build_tokenizer(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    *,
    max_length: int = 256,
):
    """Return tokenizer configured for dynamic padding + truncation."""
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.model_max_length = max_length
    return tok


def tokenize_batch(
    tokenizer,
    texts: list[str],
    *,
    max_length: int = 256,
    device: str | torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Tokenize with padding, attention masks, and truncation (dynamic batch padding)."""
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if device is not None:
        encoded = {k: v.to(device) for k, v in encoded.items()}
    return encoded
