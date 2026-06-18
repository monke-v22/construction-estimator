"""
parsers/excel_parser.py
Parses .xlsx / .xls / .csv files.
Handles multi-sheet BOQs, merged cells, and summary tabs.
"""

from __future__ import annotations
import json
from pathlib import Path
from core.logger import get_logger


def parse_excel(file_path: str, job_id: str = "system") -> dict:
    """
    Parse an Excel or CSV file and return structured content.

    Returns:
        {
            "filename": str,
            "document_type": "xlsx" | "csv",
            "page_count": int (number of sheets),
            "extracted_text": str,
            "tables": [...],
            "is_scanned": False,
            "confidence": float,
            "flagged_gaps": [...]
        }
    """
    import pandas as pd

    log_ctx = get_logger("excel_parser", job_id)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Excel file not found: {file_path}")

    log_ctx.info(f"Parsing Excel: {path.name}")
    suffix = path.suffix.lower()

    if suffix == ".csv":
        return _parse_csv(path, log_ctx)

    # ── Read all sheets ──────────────────────────────────────────
    try:
        all_sheets = pd.read_excel(str(path), sheet_name=None, header=None, dtype=str)
    except Exception as e:
        raise ValueError(f"Failed to open Excel file {path.name}: {e}") from e

    tables = []
    text_parts = []

    for sheet_name, df in all_sheets.items():
        if df.empty:
            continue

        log_ctx.debug(f"Sheet: '{sheet_name}' — {df.shape[0]} rows × {df.shape[1]} cols")

        # Drop completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        df = df.fillna("")

        if df.empty:
            continue

        # Find the header row — first row with >50% non-empty cells
        header_row_idx = _find_header_row(df)
        headers = [str(c).strip() or f"col_{i}" for i, c in enumerate(df.iloc[header_row_idx])]
        data_df = df.iloc[header_row_idx + 1:].reset_index(drop=True)

        rows = []
        for _, row in data_df.iterrows():
            row_dict = {headers[i]: str(v).strip() for i, v in enumerate(row) if i < len(headers)}
            # Skip completely empty rows
            if any(v for v in row_dict.values()):
                rows.append(row_dict)

        if rows:
            tables.append({
                "sheet_name": sheet_name,
                "page": None,
                "table_index": len(tables),
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
            })

        # Build text summary for this sheet
        text_parts.append(f"[Sheet: {sheet_name}]")
        text_parts.append(" | ".join(headers))
        for row in rows[:5]:  # First 5 rows as text preview
            text_parts.append(" | ".join(str(v) for v in row.values()))
        if len(rows) > 5:
            text_parts.append(f"... ({len(rows) - 5} more rows)")

    full_text = "\n".join(text_parts)
    flagged_gaps = []

    if not tables:
        flagged_gaps.append("No data tables found in Excel file")
    if len(all_sheets) > 1:
        text_parts.append(f"Note: {len(all_sheets)} sheets found — all processed")

    confidence = _score_confidence(tables)

    log_ctx.info(
        f"Excel parsed: sheets={len(all_sheets)} tables={len(tables)} "
        f"total_rows={sum(t['row_count'] for t in tables)} confidence={confidence:.2f}"
    )

    return {
        "filename": path.name,
        "document_type": suffix.lstrip("."),
        "page_count": len(all_sheets),
        "extracted_text": full_text,
        "tables": tables,
        "is_scanned": False,
        "confidence": confidence,
        "flagged_gaps": flagged_gaps,
    }


def _parse_csv(path: Path, log_ctx) -> dict:
    """Parse a CSV file."""
    import pandas as pd

    df = pd.read_csv(str(path), dtype=str).fillna("")
    headers = list(df.columns)
    rows = df.to_dict(orient="records")

    tables = [{
        "sheet_name": "CSV",
        "page": None,
        "table_index": 0,
        "headers": headers,
        "rows": rows,
        "row_count": len(rows),
    }]

    text = f"[CSV: {path.name}]\n"
    text += " | ".join(headers) + "\n"
    for row in rows[:5]:
        text += " | ".join(str(v) for v in row.values()) + "\n"

    return {
        "filename": path.name,
        "document_type": "csv",
        "page_count": 1,
        "extracted_text": text,
        "tables": tables,
        "is_scanned": False,
        "confidence": 0.9 if rows else 0.1,
        "flagged_gaps": [] if rows else ["CSV appears empty"],
    }


def _find_header_row(df) -> int:
    """Find the index of the most likely header row."""
    import pandas as pd
    best_idx = 0
    best_score = 0
    for i in range(min(10, len(df))):
        row = df.iloc[i]
        non_empty = sum(1 for v in row if str(v).strip())
        score = non_empty / len(row) if len(row) > 0 else 0
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _score_confidence(tables: list) -> float:
    if not tables:
        return 0.1
    total_rows = sum(t["row_count"] for t in tables)
    if total_rows > 50:
        return 0.95
    if total_rows > 10:
        return 0.8
    return 0.6
