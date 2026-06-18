"""
parsers/pdf_parser.py
Layered PDF parser: PyMuPDF (primary) → pdfplumber (tables) → Claude Vision (scanned).
Returns raw text, tables, and page metadata.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from core.logger import get_logger

log = get_logger("pdf_parser")


def parse_pdf(file_path: str, job_id: str = "system") -> dict:
    """
    Parse a PDF file and return structured content.

    Returns:
        {
            "filename": str,
            "document_type": "pdf",
            "page_count": int,
            "extracted_text": str,
            "tables": [...],
            "is_scanned": bool,
            "confidence": float,
            "flagged_gaps": [...]
        }
    """
    log_ctx = get_logger("pdf_parser", job_id)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    log_ctx.info(f"Parsing PDF: {path.name}")

    # ── Layer 1: PyMuPDF — fast text extraction ──────────────────
    text, page_count, is_scanned = _extract_with_pymupdf(path, log_ctx)

    # ── Layer 2: pdfplumber — table extraction ───────────────────
    tables = _extract_tables_pdfplumber(path, log_ctx)

    # ── Layer 3: pytesseract OCR if scanned ─────────────────────
    if is_scanned and len(text.strip()) < 100:
        log_ctx.info("Scanned PDF detected — running OCR")
        text = _ocr_pdf(path, log_ctx)

    flagged_gaps = []
    if len(text.strip()) < 50:
        flagged_gaps.append("Very little text extracted — document may be image-only or encrypted")
    if not tables:
        flagged_gaps.append("No tables detected — manual review recommended for BOQ data")

    confidence = _score_confidence(text, tables, is_scanned)

    log_ctx.info(
        f"PDF parsed: pages={page_count} tables={len(tables)} "
        f"chars={len(text)} scanned={is_scanned} confidence={confidence:.2f}"
    )

    return {
        "filename": path.name,
        "document_type": "pdf",
        "page_count": page_count,
        "extracted_text": text,
        "tables": tables,
        "is_scanned": is_scanned,
        "confidence": confidence,
        "flagged_gaps": flagged_gaps,
    }


def _extract_with_pymupdf(path: Path, log_ctx) -> tuple[str, int, bool]:
    """Extract text with PyMuPDF. Returns (text, page_count, is_scanned)."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    pages_text = []
    image_page_count = 0

    for page in doc:
        text = page.get_text("text")
        pages_text.append(text)
        # Detect scanned pages (image-heavy, little text)
        if len(text.strip()) < 20 and len(page.get_images()) > 0:
            image_page_count += 1

    page_count = len(doc)
    doc.close()

    full_text = "\n\n--- PAGE BREAK ---\n\n".join(pages_text)
    is_scanned = image_page_count > (page_count * 0.5)

    return full_text, page_count, is_scanned


def _extract_tables_pdfplumber(path: Path, log_ctx) -> list[dict]:
    """Extract tables using pdfplumber."""
    import pdfplumber

    all_tables = []
    try:
        with pdfplumber.open(str(path)) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                tables = page.extract_tables()
                for t_idx, table in enumerate(tables):
                    if not table:
                        continue
                    # Convert to list of dicts using first row as headers
                    if len(table) < 2:
                        continue
                    headers = [
                        str(h).strip() if h else f"col_{i}"
                        for i, h in enumerate(table[0])
                    ]
                    rows = []
                    for row in table[1:]:
                        row_dict = {}
                        for i, cell in enumerate(row):
                            key = headers[i] if i < len(headers) else f"col_{i}"
                            row_dict[key] = str(cell).strip() if cell else ""
                        rows.append(row_dict)

                    all_tables.append({
                        "page": page_num,
                        "table_index": t_idx,
                        "headers": headers,
                        "rows": rows,
                        "row_count": len(rows),
                    })
    except Exception as e:
        log_ctx.warning(f"pdfplumber table extraction error: {e}")

    return all_tables


def _ocr_pdf(path: Path, log_ctx) -> str:
    """OCR a scanned PDF using pytesseract."""
    import fitz
    import pytesseract
    from PIL import Image
    import io

    doc = fitz.open(str(path))
    pages_text = []

    for page_num, page in enumerate(doc):
        # Render page as image at 200 DPI
        mat = fitz.Matrix(200 / 72, 200 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        img = Image.open(io.BytesIO(img_bytes))

        try:
            text = pytesseract.image_to_string(img, lang="eng")
            pages_text.append(text)
            log_ctx.debug(f"OCR page {page_num + 1}: {len(text)} chars")
        except Exception as e:
            log_ctx.warning(f"OCR failed on page {page_num + 1}: {e}")
            pages_text.append("")

    doc.close()
    return "\n\n--- PAGE BREAK ---\n\n".join(pages_text)


def _score_confidence(text: str, tables: list, is_scanned: bool) -> float:
    """Score extraction confidence 0.0–1.0 based on richness of output."""
    score = 0.0
    if len(text) > 500:
        score += 0.4
    elif len(text) > 100:
        score += 0.2
    if tables:
        score += 0.3
    if not is_scanned:
        score += 0.2
    if len(text) > 2000:
        score += 0.1
    return min(score, 1.0)
