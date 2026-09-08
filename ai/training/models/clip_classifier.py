"""CLIP vision encoder + classification head for ConsentShield Phase 3."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor


class CLIPClassifier(nn.Module):
    """
    Fine-tune CLIP ViT for 5-class dark-pattern vision classification.

    Frozen:
      - text encoder (``clip.text_model``)
      - projection layers (``visual_projection``, ``text_projection``, ``logit_scale``)

    Trainable:
      - vision encoder (``clip.vision_model``)
      - classification head on pooled vision features
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        *,
        num_labels: int = 5,
        dropout: float = 0.1,
        freeze_text_encoder: bool = True,
        freeze_projection: bool = True,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.num_labels = int(num_labels)
        self.clip = CLIPModel.from_pretrained(model_name)
        hidden = int(self.clip.config.vision_config.hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, self.num_labels)

        if freeze_text_encoder:
            for p in self.clip.text_model.parameters():
                p.requires_grad = False
            if self.clip.text_projection is not None:
                for p in self.clip.text_projection.parameters():
                    p.requires_grad = False

        if freeze_projection:
            if self.clip.visual_projection is not None:
                for p in self.clip.visual_projection.parameters():
                    p.requires_grad = False
            if self.clip.text_projection is not None:
                for p in self.clip.text_projection.parameters():
                    p.requires_grad = False
            if hasattr(self.clip, "logit_scale"):
                self.clip.logit_scale.requires_grad = False

    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        vision_outputs = self.clip.vision_model(pixel_values=pixel_values)
        pooled = vision_outputs.pooler_output
        return pooled

    def forward(
        self,
        pixel_values: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        pooled = self.encode_images(pixel_values)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        probs = torch.softmax(logits, dim=-1)
        return {"loss": loss, "logits": logits, "probs": probs, "pooled": pooled}

    def trainable_parameter_counts(self) -> dict[str, int]:
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def build_clip_processor(model_name: str = "openai/clip-vit-base-patch32") -> CLIPProcessor:
    return CLIPProcessor.from_pretrained(model_name)


def clip_image_mean_std(processor: CLIPProcessor) -> tuple[list[float], list[float]]:
    img = getattr(processor, "image_processor", None) or processor
    mean = list(getattr(img, "image_mean", [0.48145466, 0.4578275, 0.40821073]))
    std = list(getattr(img, "image_std", [0.26862954, 0.26130258, 0.27577711]))
    return mean, std
