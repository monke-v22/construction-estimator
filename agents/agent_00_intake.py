"""
agents/agent_00_intake.py
Agent #0 — Project Intake Agent
Model: claude-opus-4-6 (via AgentRouter)

Responsibilities:
  1. Accept free-text project description from user
  2. Classify project type, grade, region, trades
  3. Extract area, floors, constraints
  4. Produce project_context JSON
  5. This context flows to ALL downstream agents
"""

from __future__ import annotations
import time
from core.state import EstimatorState, ProjectContext
from core.llm_client import llm
from core.prompt_loader import prompt_loader
from core.logger import get_logger
from core.config import PROJECT_TYPES, GRADES, TRADES

AGENT_NAME = "agent_00_intake"


def run(state: EstimatorState) -> EstimatorState:
    """
    LangGraph node entry point.
    Reads:  state["user_description"]
    Writes: state["project_context"], state["status"]
    """
    job_id = state.get("job_id", "unknown")
    log = get_logger(AGENT_NAME, job_id)
    t_start = time.time()

    log.info("Agent #0 started — Project Intake")
    state["status"] = "intake"

    description = state.get("user_description", "").strip()
    if not description:
        log.error("No user_description provided")
        state.setdefault("errors", []).append({
            "agent": AGENT_NAME,
            "error_type": "missing_input",
            "message": "user_description is empty",
        })
        state["status"] = "error"
        return state

    log.info(f"Classifying: '{description[:80]}...'")

    # ── Call Claude Opus ─────────────────────────────────────
    messages = prompt_loader.get_messages(
        "agent_00_intake.yaml",
        context={
            "project_types": "\n  ".join(f"- {pt}" for pt in PROJECT_TYPES),
            "user_description": description,
        }
    )

    raw = llm.chat_json(
        agent_name=AGENT_NAME,
        messages=messages,
        job_id=job_id,
        temperature=0.0,
        max_tokens=1024,
    )

    context = _validate_and_normalise(raw, log)
    state["project_context"] = context
    state["status"] = "intake_complete"

    elapsed = time.time() - t_start
    state.setdefault("agent_timings", {})[AGENT_NAME] = round(elapsed, 2)

    log.info(
        f"Agent #0 complete in {elapsed:.1f}s | "
        f"type={context['project_type']} grade={context['grade']} "
        f"area={context.get('total_area_m2')} confidence={context['intake_confidence']}"
    )
    return state


def _validate_and_normalise(raw: dict, log) -> ProjectContext:
    """Validate LLM output and apply safe defaults for any missing fields."""

    # Project type
    project_type = raw.get("project_type", "other")
    if project_type not in PROJECT_TYPES:
        log.warning(f"Unknown project_type '{project_type}' — defaulting to 'other'")
        project_type = "other"

    # Grade
    grade = raw.get("grade", "standard")
    if grade not in GRADES:
        log.warning(f"Unknown grade '{grade}' — defaulting to 'standard'")
        grade = "standard"

    # Area
    area = raw.get("total_area_m2")
    if area is not None:
        try:
            area = float(area)
            if area <= 0:
                area = None
        except (TypeError, ValueError):
            area = None

    # Trades
    included = [t for t in raw.get("included_trades", []) if t in TRADES]
    excluded = [t for t in raw.get("excluded_trades", []) if t in TRADES]

    # If no trades detected, apply project-type defaults
    if not included:
        included = _default_trades(project_type)
        log.info(f"No trades in description — using defaults for {project_type}: {included}")

    # Region / currency
    region = raw.get("region", "KSA")
    currency = raw.get("currency", "SAR")
    if currency == "unknown":
        currency = "AED" if region == "UAE" else "SAR"

    confidence = raw.get("intake_confidence", "medium")
    if confidence not in ("high", "medium", "low"):
        confidence = "medium"

    return {
        "project_type":        project_type,
        "grade":               grade,
        "total_area_m2":       area,
        "floors":              raw.get("floors", []),
        "included_trades":     included,
        "excluded_trades":     excluded,
        "region":              region,
        "currency":            currency,
        "special_requirements": str(raw.get("special_requirements", "")),
        "client_notes":        str(raw.get("client_notes", "")),
        "intake_confidence":   confidence,
        "confidence_reason":   str(raw.get("confidence_reason", "")),
    }


def _default_trades(project_type: str) -> list[str]:
    """Return the standard set of trades for a given project type."""
    defaults = {
        "office_fitout":  ["civil", "mep", "finishing", "ffe"],
        "retail":         ["civil", "mep", "finishing", "ffe"],
        "hospitality":    ["civil", "mep", "finishing", "ffe"],
        "residential":    ["civil", "mep", "finishing", "ffe"],
        "healthcare":     ["civil", "mep", "finishing"],
        "renovation":     ["civil", "mep", "finishing"],
        "mep_only":       ["mep"],
        "finishing_only": ["finishing"],
        "full_fitout":    ["civil", "mep", "finishing", "ffe", "external"],
        "other":          ["civil", "mep", "finishing"],
    }
    return defaults.get(project_type, ["civil", "mep", "finishing"])
