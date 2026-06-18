"""
agents/agent_02_qto.py
Agent #2 — Quantity Takeoff (QTO) Agent
Model: claude-opus-4-6 (via AgentRouter)

Responsibilities:
  1. Read extracted document data from state
  2. Cross-reference all document sources
  3. Calculate quantities per trade and line item
  4. Flag low-confidence and missing items
  5. Output structured QTO table for HITL Checkpoint 2
"""

from __future__ import annotations
import json
import time
from core.state import EstimatorState, QTOItem
from core.llm_client import llm
from core.prompt_loader import prompt_loader
from core.logger import get_logger

AGENT_NAME = "agent_02_qto"

# Trade letter prefixes for item IDs
TRADE_PREFIX = {
    "civil": "A",
    "mep": "B",
    "finishing": "C",
    "ffe": "D",
    "external": "E",
}


def run(state: EstimatorState) -> EstimatorState:
    """
    LangGraph node entry point.
    Reads:  state["extracted_data"], state["project_context"]
    Writes: state["qto_table"], state["status"]
    """
    job_id = state.get("job_id", "unknown")
    log = get_logger(AGENT_NAME, job_id)
    t_start = time.time()

    log.info("Agent #2 started — Quantity Takeoff")
    state["status"] = "qto"

    project_context = state.get("project_context", {})
    extracted = state.get("extracted_data", {})

    if not project_context:
        log.error("No project_context — Agent #0 must run first")
        state["status"] = "error"
        return state

    # ── Prepare context variables for prompt ─────────────────
    area = project_context.get("total_area_m2")
    area_note = f"{area} m² (from documents)" if area else "Not specified — use document data"

    # Gather all extracted data across all documents
    all_floor_areas   = extracted.get("all_floor_areas", [])
    all_dimensions    = extracted.get("all_dimensions", [])
    all_material_specs = extracted.get("all_material_specs", [])
    all_gaps          = extracted.get("flagged_gaps", [])

    # Collect boq_items and scope_statements from per-document analysis
    all_boq_items = []
    all_scope_stmts = []
    for doc in extracted.get("documents", []):
        all_boq_items.extend(doc.get("boq_items", []))
        all_scope_stmts.extend(doc.get("scope_statements", []))

    log.info(
        f"Input: area={area} floors={len(all_floor_areas)} "
        f"dims={len(all_dimensions)} boq_items={len(all_boq_items)}"
    )

    # ── Call Claude Opus ─────────────────────────────────────
    messages = prompt_loader.get_messages(
        "agent_02_qto.yaml",
        context={
            "project_type":       project_context.get("project_type", "unknown"),
            "grade":              project_context.get("grade", "standard"),
            "region":             project_context.get("region", "KSA"),
            "currency":           project_context.get("currency", "SAR"),
            "total_area_m2":      area or "Not stated",
            "area_note":          area_note,
            "floors":             ", ".join(project_context.get("floors", [])) or "Not specified",
            "included_trades":    ", ".join(project_context.get("included_trades", [])),
            "excluded_trades":    ", ".join(project_context.get("excluded_trades", [])) or "None",
            "special_requirements": project_context.get("special_requirements", "None"),
            "project_context_json": json.dumps(project_context, indent=2),
            "floor_areas_json":   json.dumps(all_floor_areas, indent=2)[:3000] or "[]",
            "dimensions_json":    json.dumps(all_dimensions, indent=2)[:2000] or "[]",
            "material_specs_json": json.dumps(all_material_specs, indent=2)[:2000] or "[]",
            "boq_items_json":     json.dumps(all_boq_items, indent=2)[:5000] or "[]",
            "scope_statements":   "\n".join(f"- {s}" for s in all_scope_stmts) or "None found",
            "flagged_gaps":       "\n".join(f"- {g}" for g in all_gaps) or "None",
        }
    )

    raw = llm.chat_json(
        agent_name=AGENT_NAME,
        messages=messages,
        job_id=job_id,
        temperature=0.0,
        max_tokens=6000,
    )

    # ── Parse and validate QTO output ────────────────────────
    qto_items = _parse_qto_items(raw.get("qto_items", []), log)
    assumptions = raw.get("assumptions", [])
    gaps = raw.get("flagged_gaps", [])
    overall_confidence = raw.get("overall_confidence", 0.0)

    # Append any new gaps to existing ones
    state["warnings"] = state.get("warnings", []) + gaps

    state["qto_table"] = qto_items
    state["status"] = "checkpoint_2_pending"

    elapsed = time.time() - t_start
    state.setdefault("agent_timings", {})[AGENT_NAME] = round(elapsed, 2)

    log.info(
        f"Agent #2 complete in {elapsed:.1f}s | "
        f"items={len(qto_items)} confidence={overall_confidence:.2f} "
        f"gaps={len(gaps)} assumptions={len(assumptions)}"
    )
    return state


def _parse_qto_items(raw_items: list, log) -> list[QTOItem]:
    """Validate and normalise raw QTO items from LLM output."""
    valid_units = {"m2", "m3", "lm", "nr", "kg", "ls"}
    valid_trades = {"civil", "mep", "finishing", "ffe", "external"}
    valid_confidence = {"high", "medium", "low"}

    items: list[QTOItem] = []
    trade_counters = {t: 0 for t in valid_trades}

    for raw in raw_items:
        try:
            trade = str(raw.get("trade", "civil")).lower()
            if trade not in valid_trades:
                log.warning(f"Unknown trade '{trade}' in QTO item — skipping")
                continue

            unit = str(raw.get("unit", "nr")).lower().replace("sqm", "m2").replace("sq.m", "m2")
            if unit not in valid_units:
                log.warning(f"Unknown unit '{unit}' — defaulting to 'nr'")
                unit = "nr"

            qty = raw.get("quantity", 0)
            try:
                qty = float(qty)
            except (TypeError, ValueError):
                qty = 0.0

            confidence = str(raw.get("confidence", "medium")).lower()
            if confidence not in valid_confidence:
                confidence = "medium"

            # Auto-generate item_id if missing or reformat
            trade_counters[trade] += 1
            prefix = TRADE_PREFIX.get(trade, "X")
            item_id = raw.get("item_id") or f"{prefix}.{trade_counters[trade]:02d}"

            items.append({
                "item_id":    item_id,
                "trade":      trade,
                "description": str(raw.get("description", "")).strip(),
                "unit":       unit,
                "quantity":   round(qty, 3),
                "source_doc": str(raw.get("source_doc", "estimated")),
                "confidence": confidence,
                "notes":      str(raw.get("notes", "")),
            })

        except Exception as e:
            log.warning(f"Skipping invalid QTO item: {e} | raw={raw}")

    log.info(f"Parsed {len(items)} valid QTO items from {len(raw_items)} raw items")
    return items
