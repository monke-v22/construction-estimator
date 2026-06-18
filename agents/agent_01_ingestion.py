"""
agents/agent_01_ingestion.py
Agent #1 — Document Ingestion Agent
Model: claude-opus-4-6 (via AgentRouter)

Responsibilities:
  1. Receive uploaded file paths from EstimatorState
  2. Dispatch each file to the correct parser (PDF/DOCX/XLSX/DXF/Image)
  3. Call Claude Opus to analyze extracted content per document
  4. Merge all outputs into unified ExtractedData
  5. Flag confidence issues for HITL Checkpoint 1
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from core.state import EstimatorState, ExtractedData, DocumentInfo
from core.llm_client import llm
from core.prompt_loader import prompt_loader
from core.logger import get_logger
from parsers.dispatcher import dispatch_many

AGENT_NAME = "agent_01_ingestion"


def run(state: EstimatorState) -> EstimatorState:
    """
    LangGraph node entry point.
    Reads: state["uploaded_file_paths"], state["project_context"]
    Writes: state["extracted_data"], state["status"]
    """
    job_id = state.get("job_id", "unknown")
    log = get_logger(AGENT_NAME, job_id)
    t_start = time.time()

    log.info("Agent #1 started — Document Ingestion")
    state["status"] = "ingestion"

    file_paths = state.get("uploaded_file_paths", [])
    project_context = state.get("project_context", {})

    if not file_paths:
        log.warning("No files to process — skipping ingestion")
        state["extracted_data"] = _empty_extraction()
        state["warnings"] = state.get("warnings", []) + [
            "No documents uploaded — QTO will rely on project description only"
        ]
        return state

    log.info(f"Processing {len(file_paths)} files")

    # ── Step 1: Parse all files ──────────────────────────────────
    parsed_docs = dispatch_many(file_paths, job_id=job_id)
    log.info(f"Parsed {len(parsed_docs)} documents")

    # ── Step 2: LLM analysis per document ────────────────────────
    analyzed_docs: list[DocumentInfo] = []

    for doc in parsed_docs:
        if doc["document_type"] == "error":
            # Pass through error docs without LLM call
            analyzed_docs.append(_build_error_doc(doc))
            continue

        log.info(f"Analyzing: {doc['filename']} ({doc['document_type']})")

        try:
            analysis = _analyze_document(doc, project_context, job_id, log)
            doc_info = _build_document_info(doc, analysis)
            analyzed_docs.append(doc_info)
        except Exception as e:
            log.error(f"LLM analysis failed for {doc['filename']}: {e}")
            analyzed_docs.append(_build_error_doc(doc, str(e)))

    # ── Step 3: Merge all into ExtractedData ─────────────────────
    extracted = _merge_documents(analyzed_docs, log)

    state["extracted_data"] = extracted
    state["status"] = "checkpoint_1_pending"

    elapsed = time.time() - t_start
    state.setdefault("agent_timings", {})[AGENT_NAME] = round(elapsed, 2)

    log.info(
        f"Agent #1 complete in {elapsed:.1f}s | "
        f"docs={len(analyzed_docs)} | "
        f"confidence={extracted.get('overall_confidence', 0):.2f} | "
        f"gaps={len(extracted.get('flagged_gaps', []))}"
    )

    return state


def _analyze_document(
    doc: dict,
    project_context: dict,
    job_id: str,
    log,
) -> dict:
    """Call Claude Opus to extract structured data from a parsed document."""

    # Truncate text to avoid token limits (~12k chars = ~3k tokens)
    text_excerpt = doc.get("extracted_text", "")[:12000]

    # Format tables as JSON string (truncated)
    tables = doc.get("tables", [])
    tables_json = json.dumps(tables[:10], indent=2)[:6000]  # Max 10 tables, 6k chars

    messages = prompt_loader.get_messages(
        "agent_01_ingestion.yaml",
        context={
            "project_type":    project_context.get("project_type", "unknown"),
            "grade":           project_context.get("grade", "standard"),
            "region":          project_context.get("region", "GCC"),
            "included_trades": ", ".join(project_context.get("included_trades", ["civil", "mep", "finishing"])),
            "filename":        doc["filename"],
            "document_type":   doc["document_type"],
            "page_count":      doc.get("page_count", 1),
            "extracted_text":  text_excerpt,
            "tables_json":     tables_json,
        }
    )

    result = llm.chat_json(
        agent_name=AGENT_NAME,
        messages=messages,
        job_id=job_id,
        temperature=0.0,
        max_tokens=4096,
    )
    return result


def _build_document_info(raw_doc: dict, analysis: dict) -> DocumentInfo:
    """Merge parser output + LLM analysis into a DocumentInfo dict."""
    return {
        "file_path":           raw_doc.get("filename", ""),
        "document_type":       raw_doc.get("document_type", "unknown"),
        "raw_text":            raw_doc.get("extracted_text", ""),
        "tables":              raw_doc.get("tables", []),
        "dimensions":          analysis.get("dimensions", []),
        "material_specs":      analysis.get("material_specs", []),
        "floor_areas":         analysis.get("floor_areas", []),
        "boq_items":           analysis.get("boq_items", []),
        "scope_statements":    analysis.get("scope_statements", []),
        "images_extracted":    [],
        "confidence":          analysis.get("overall_confidence", raw_doc.get("confidence", 0.5)),
        "flagged_gaps":        (
            raw_doc.get("flagged_gaps", []) +
            analysis.get("flagged_gaps", [])
        ),
    }


def _build_error_doc(raw_doc: dict, error_msg: str = "") -> DocumentInfo:
    gaps = raw_doc.get("flagged_gaps", [])
    if error_msg:
        gaps.append(f"LLM analysis error: {error_msg}")
    return {
        "file_path":        raw_doc.get("filename", ""),
        "document_type":    raw_doc.get("document_type", "error"),
        "raw_text":         raw_doc.get("extracted_text", ""),
        "tables":           [],
        "dimensions":       [],
        "material_specs":   [],
        "floor_areas":      [],
        "boq_items":        [],
        "scope_statements": [],
        "images_extracted": [],
        "confidence":       0.0,
        "flagged_gaps":     gaps,
    }


def _merge_documents(docs: list[DocumentInfo], log) -> ExtractedData:
    """Combine all DocumentInfo objects into a single ExtractedData."""
    combined_text_parts = []
    all_tables = []
    all_dimensions = []
    all_material_specs = []
    all_floor_areas = []
    all_gaps = []
    confidence_scores = []

    for doc in docs:
        if doc.get("raw_text"):
            combined_text_parts.append(
                f"=== {doc['file_path']} ===\n{doc['raw_text']}"
            )
        all_tables.extend(doc.get("tables", []))
        all_dimensions.extend(doc.get("dimensions", []))
        all_material_specs.extend(doc.get("material_specs", []))
        all_floor_areas.extend(doc.get("floor_areas", []))
        all_gaps.extend(doc.get("flagged_gaps", []))
        if doc.get("confidence", 0) > 0:
            confidence_scores.append(doc["confidence"])

    overall_confidence = (
        sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
    )

    log.info(
        f"Merged: tables={len(all_tables)} dims={len(all_dimensions)} "
        f"areas={len(all_floor_areas)} specs={len(all_material_specs)} "
        f"gaps={len(all_gaps)} avg_confidence={overall_confidence:.2f}"
    )

    return {
        "documents":         docs,
        "combined_text":     "\n\n".join(combined_text_parts),
        "all_tables":        all_tables,
        "all_dimensions":    all_dimensions,
        "all_material_specs": all_material_specs,
        "all_floor_areas":   all_floor_areas,
        "overall_confidence": round(overall_confidence, 3),
        "flagged_gaps":      list(set(all_gaps)),  # Deduplicate
    }


def _empty_extraction() -> ExtractedData:
    return {
        "documents":          [],
        "combined_text":      "",
        "all_tables":         [],
        "all_dimensions":     [],
        "all_material_specs": [],
        "all_floor_areas":    [],
        "overall_confidence": 0.0,
        "flagged_gaps":       ["No documents provided"],
    }
