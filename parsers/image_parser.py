"""
parsers/image_parser.py
Parses scanned documents and images.
Layer 1: pytesseract OCR (fast, free)
Layer 2: Claude Vision via AgentRouter (for complex drawings or failed OCR)
"""

from __future__ import annotations
import base64
import json
from pathlib import Path
from core.logger import get_logger

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp"}


def parse_image(file_path: str, job_id: str = "system", use_vision: bool = True) -> dict:
    """
    Parse an image/scan file and extract text and dimensional data.

    Args:
        file_path: Path to the image file
        job_id: For logging
        use_vision: If True, falls back to Claude Vision when OCR confidence is low

    Returns same structure as other parsers:
        { filename, document_type, extracted_text, tables, confidence, flagged_gaps, ... }
    """
    log_ctx = get_logger("image_parser", job_id)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Image not found: {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image format: {path.suffix}")

    log_ctx.info(f"Parsing image: {path.name}")

    # ── Layer 1: pytesseract OCR ─────────────────────────────────
    ocr_text, ocr_confidence = _run_ocr(path, log_ctx)

    flagged_gaps = []
    text = ocr_text

    # ── Layer 2: Claude Vision (if OCR was poor) ─────────────────
    if ocr_confidence < 0.4 and use_vision:
        log_ctx.info(f"OCR confidence low ({ocr_confidence:.2f}) — trying Claude Vision")
        vision_text = _run_claude_vision(path, log_ctx, job_id)
        if vision_text and len(vision_text) > len(ocr_text):
            text = vision_text
            log_ctx.info(f"Claude Vision returned {len(vision_text)} chars (better than OCR)")
        else:
            flagged_gaps.append("Both OCR and Claude Vision returned limited text — image quality may be poor")

    if len(text.strip()) < 50:
        flagged_gaps.append("Very little text extracted from image — may need manual review")

    confidence = min(ocr_confidence + 0.1, 1.0) if text else 0.1

    log_ctx.info(f"Image parsed: chars={len(text)} confidence={confidence:.2f}")

    return {
        "filename": path.name,
        "document_type": "image",
        "page_count": 1,
        "extracted_text": text,
        "tables": [],
        "is_scanned": True,
        "confidence": confidence,
        "flagged_gaps": flagged_gaps,
    }


def _run_ocr(path: Path, log_ctx) -> tuple[str, float]:
    """Run pytesseract OCR and return (text, confidence_score)."""
    try:
        import pytesseract
        from PIL import Image
        import numpy as np

        img = Image.open(str(path))

        # Preprocess: convert to grayscale, increase contrast
        img_gray = img.convert("L")

        # Get text + confidence data
        data = pytesseract.image_to_data(img_gray, output_type=pytesseract.Output.DICT)
        text = pytesseract.image_to_string(img_gray, lang="eng")

        # Calculate mean confidence (ignore -1 entries which are layout items)
        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        mean_conf = (sum(confidences) / len(confidences) / 100.0) if confidences else 0.0

        log_ctx.debug(f"OCR: chars={len(text)} mean_conf={mean_conf:.2f}")
        return text, mean_conf

    except ImportError:
        log_ctx.warning("pytesseract not installed — skipping OCR")
        return "", 0.0
    except Exception as e:
        log_ctx.warning(f"OCR failed: {e}")
        return "", 0.0


def _run_claude_vision(path: Path, log_ctx, job_id: str) -> str:
    """Use Claude Vision (via AgentRouter) to extract text from image."""
    try:
        from openai import OpenAI
        from core.config import settings, MODELS

        # Encode image as base64
        with open(str(path), "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        suffix = path.suffix.lower().lstrip(".")
        media_type_map = {
            "jpg": "image/jpeg", "jpeg": "image/jpeg",
            "png": "image/png", "tiff": "image/tiff",
            "tif": "image/tiff", "webp": "image/webp",
        }
        media_type = media_type_map.get(suffix, "image/jpeg")

        client = OpenAI(
            api_key=settings.agentrouter_api_key,
            base_url=settings.agentrouter_base_url,
        )

        response = client.chat.completions.create(
            model=MODELS["opus"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}"
                            }
                        },
                        {
                            "type": "text",
                            "text": (
                                "This is a construction document scan. "
                                "Extract ALL text, numbers, dimensions, room labels, "
                                "material specifications, and area measurements you can see. "
                                "Format as plain text, preserving all numbers and units exactly."
                            )
                        }
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.0,
        )
        return response.choices[0].message.content or ""

    except Exception as e:
        log_ctx.warning(f"Claude Vision failed: {e}")
        return ""
