"""Image transforms for CLIP fine-tuning (no flips)."""

from __future__ import annotations

from typing import Any, Callable

from PIL import Image


def build_train_transform(
    *,
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> Callable[[Image.Image], Any]:
    from torchvision import transforms

    mean = mean or [0.48145466, 0.4578275, 0.40821073]
    std = std or [0.26862954, 0.26130258, 0.27577711]
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))],
                p=0.15,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def build_eval_transform(
    *,
    image_size: int = 224,
    mean: list[float] | None = None,
    std: list[float] | None = None,
) -> Callable[[Image.Image], Any]:
    from torchvision import transforms

    mean = mean or [0.48145466, 0.4578275, 0.40821073]
    std = std or [0.26862954, 0.26130258, 0.27577711]
    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def load_rgb_image(path: str) -> Image.Image:
    img = Image.open(path).convert("RGB")
    return img
