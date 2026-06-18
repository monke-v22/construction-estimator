"""
parsers/dxf_parser.py
Parses AutoCAD DXF files using ezdxf.
Extracts room dimensions, area labels, and layer information.
DWG files require pre-conversion to DXF via ODA File Converter.
"""

from __future__ import annotations
from pathlib import Path
from core.logger import get_logger


def parse_dxf(file_path: str, job_id: str = "system") -> dict:
    """
    Parse a DXF file and extract dimensional/area data.

    Returns:
        {
            "filename": str,
            "document_type": "dxf",
            "page_count": 1,
            "extracted_text": str,
            "tables": [],
            "dimensions": [...],
            "floor_areas": [...],
            "is_scanned": False,
            "confidence": float,
            "flagged_gaps": [...]
        }
    """
    import ezdxf

    log_ctx = get_logger("dxf_parser", job_id)
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"DXF not found: {file_path}")

    suffix = path.suffix.lower()
    if suffix == ".dwg":
        raise ValueError(
            f"DWG files cannot be read directly. "
            f"Convert '{path.name}' to DXF using ODA File Converter first. "
            f"Download: https://www.opendesign.com/guestfiles/oda_file_converter"
        )

    log_ctx.info(f"Parsing DXF: {path.name}")

    try:
        doc = ezdxf.readfile(str(path))
    except Exception as e:
        raise ValueError(f"Failed to open DXF file {path.name}: {e}") from e

    msp = doc.modelspace()

    dimensions = []
    text_items = []
    floor_areas = []
    layers = set()

    for entity in msp:
        layer = entity.dxf.layer if hasattr(entity.dxf, "layer") else "0"
        layers.add(layer)

        # ── DIMENSION entities ───────────────────────────────────
        if entity.dxftype() == "DIMENSION":
            try:
                measurement = entity.dxf.get("actual_measurement", None)
                if measurement and measurement > 0:
                    dimensions.append({
                        "element": f"dim_{len(dimensions)+1}",
                        "measurement": round(measurement, 3),
                        "unit": "drawing_units",
                        "layer": layer,
                        "notes": f"DIMENSION entity on layer '{layer}'",
                    })
            except Exception:
                pass

        # ── TEXT and MTEXT entities ──────────────────────────────
        elif entity.dxftype() in ("TEXT", "MTEXT"):
            try:
                if entity.dxftype() == "TEXT":
                    text = entity.dxf.text.strip()
                else:
                    text = entity.text.strip()

                if text:
                    text_items.append({"text": text, "layer": layer})

                    # Detect area labels like "50.5 m2", "120 sqm", "ROOM: 35 M2"
                    import re
                    area_match = re.search(
                        r"(\d+\.?\d*)\s*(m2|sqm|sq\.m|m²|sqft|ft2)", text, re.IGNORECASE
                    )
                    if area_match:
                        area_val = float(area_match.group(1))
                        unit = area_match.group(2).lower()
                        # Convert sqft to m2 if needed
                        if unit in ("sqft", "ft2"):
                            area_val = round(area_val * 0.0929, 2)
                            unit = "m2 (converted from sqft)"
                        floor_areas.append({
                            "floor_name": f"area_{len(floor_areas)+1}",
                            "area_m2": area_val,
                            "area_source": "dxf_text_label",
                            "original_text": text,
                            "layer": layer,
                            "notes": "",
                        })
            except Exception:
                pass

        # ── HATCH entities (floor areas from closed polylines) ───
        elif entity.dxftype() == "HATCH":
            try:
                for path_obj in entity.paths:
                    area = getattr(path_obj, "area", None)
                    if area and area > 0.1:
                        floor_areas.append({
                            "floor_name": f"hatch_area_{len(floor_areas)+1}",
                            "area_m2": round(area, 2),
                            "area_source": "dxf_hatch",
                            "original_text": "",
                            "layer": layer,
                            "notes": "Area calculated from HATCH boundary",
                        })
            except Exception:
                pass

    # Build text summary
    text_summary_parts = [f"DXF File: {path.name}"]
    text_summary_parts.append(f"Layers: {', '.join(sorted(layers))}")
    text_summary_parts.append(f"Dimension entities: {len(dimensions)}")
    text_summary_parts.append(f"Text labels: {len(text_items)}")
    text_summary_parts.append(f"Area labels found: {len(floor_areas)}")

    if text_items:
        text_summary_parts.append("\n── Text Labels ──")
        for item in text_items[:50]:
            text_summary_parts.append(f"  [{item['layer']}] {item['text']}")

    flagged_gaps = []
    if not dimensions and not floor_areas:
        flagged_gaps.append("No dimension or area data found in DXF — file may be 3D or use non-standard layers")
    if not floor_areas:
        flagged_gaps.append("No area labels detected — areas may need manual measurement")

    confidence = _score_confidence(dimensions, floor_areas, text_items)

    log_ctx.info(
        f"DXF parsed: layers={len(layers)} dims={len(dimensions)} "
        f"areas={len(floor_areas)} confidence={confidence:.2f}"
    )

    return {
        "filename": path.name,
        "document_type": "dxf",
        "page_count": 1,
        "extracted_text": "\n".join(text_summary_parts),
        "tables": [],
        "dimensions": dimensions,
        "floor_areas": floor_areas,
        "is_scanned": False,
        "confidence": confidence,
        "flagged_gaps": flagged_gaps,
    }


def _score_confidence(dimensions: list, floor_areas: list, text_items: list) -> float:
    score = 0.2  # Base score for successful parse
    if dimensions:
        score += 0.3
    if floor_areas:
        score += 0.4
    if text_items:
        score += 0.1
    return min(score, 1.0)
