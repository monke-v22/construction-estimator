"""
parsers/dispatcher.py
Auto-detects file type and routes to the correct parser.
Single entry point for all document parsing in Agent #1.
"""

from __future__ import annotations
from pathlib import Path
from core.logger import get_logger

# Extension → parser mapping
_EXT_MAP = {
    ".pdf":  "pdf",
    ".docx": "docx",
    ".doc":  "docx",
    ".xlsx": "excel",
    ".xls":  "excel",
    ".csv":  "excel",
    ".dxf":  "dxf",
    ".dwg":  "dxf",
    ".jpg":  "image",
    ".jpeg": "image",
    ".png":  "image",
    ".tiff": "image",
    ".tif":  "image",
    ".bmp":  "image",
    ".webp": "image",
}


def dispatch(file_path: str, job_id: str = "system") -> dict:
    """
    Auto-detect file type and parse it.

    Args:
        file_path: Absolute or relative path to the file
        job_id: For logging traceability

    Returns:
        Parsed document dict (same schema across all parsers)

    Raises:
        ValueError: If file type is unsupported
        FileNotFoundError: If file does not exist
    """
    log_ctx = get_logger("dispatcher", job_id)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    suffix = path.suffix.lower()
    parser_type = _EXT_MAP.get(suffix)

    if not parser_type:
        raise ValueError(
            f"Unsupported file type: '{suffix}' for file '{path.name}'\n"
            f"Supported: {', '.join(sorted(_EXT_MAP.keys()))}"
        )

    log_ctx.info(f"Dispatching '{path.name}' → {parser_type} parser")

    if parser_type == "pdf":
        from parsers.pdf_parser import parse_pdf
        return parse_pdf(file_path, job_id=job_id)

    elif parser_type == "docx":
        from parsers.docx_parser import parse_docx
        return parse_docx(file_path, job_id=job_id)

    elif parser_type == "excel":
        from parsers.excel_parser import parse_excel
        return parse_excel(file_path, job_id=job_id)

    elif parser_type == "dxf":
        from parsers.dxf_parser import parse_dxf
        return parse_dxf(file_path, job_id=job_id)

    elif parser_type == "image":
        from parsers.image_parser import parse_image
        return parse_image(file_path, job_id=job_id)


def dispatch_many(file_paths: list[str], job_id: str = "system") -> list[dict]:
    """
    Parse multiple files. Continues on individual failures (logs errors).

    Returns list of successfully parsed document dicts.
    """
    log_ctx = get_logger("dispatcher", job_id)
    results = []

    for fp in file_paths:
        try:
            result = dispatch(fp, job_id=job_id)
            results.append(result)
        except Exception as e:
            log_ctx.error(f"Failed to parse '{fp}': {type(e).__name__}: {e}")
            # Return a minimal error record so the pipeline can continue
            results.append({
                "filename": Path(fp).name,
                "document_type": "error",
                "page_count": 0,
                "extracted_text": "",
                "tables": [],
                "is_scanned": False,
                "confidence": 0.0,
                "flagged_gaps": [f"Parse error: {type(e).__name__}: {str(e)}"],
            })

    return results
