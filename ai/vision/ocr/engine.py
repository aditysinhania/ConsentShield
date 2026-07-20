"""OCR stubs — PaddleOCR preferred; EasyOCR / Tesseract as fallbacks later."""

from __future__ import annotations

from typing import Any


class OCREngine:
    name = "ocr_stub"
    version = "0.1.0"

    def extract_text(self, image: Any) -> dict[str, Any]:
        return {
            "status": "not_loaded",
            "text": "",
            "boxes": [],
            "message": "OCR not configured. Install PaddleOCR and wire ai/vision/ocr.",
        }
