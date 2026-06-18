"""
agents/agent_04_timeline.py
Agent #4 — Timeline & Schedule Agent
Model: deepseek-v4-pro (via AgentRouter)
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from core.state import EstimatorState, Timeline
from core.llm_client import llm
from core.prompt_loader import prompt_loader
from core.logger import get_logger

AGENT_NAME = "agent_04_timeline"
_ROOT = Path(__file__).parent.parent
_PRODUCTIVITY_PATH = _ROOT / "knowledge_base" / "productivity_rates.json"


def run(state: EstimatorState) -> EstimatorState:
    job_id = state.get("job_id", "unknown")
    log = get_logger(AGENT_NAME, job_id)
    t_start = time.time()

    log.info("Agent #4 started — Timeline & Schedule")
    state["status"] = "timeline"

    qto_table = state.get("qto_table", [])
    cost_summary = state.get("cost_summary", {})
    project_context = state.get("project_context", {})

    # Load productivity rates
    productivity_rates = {}
    if _PRODUCTIVITY_PATH.exists():
        with open(_PRODUCTIVITY_PATH) as f:
            productivity_rates = json.load(f)

    messages = prompt_loader.get_messages(
        "agent_04_timeline.yaml",
        context={
            "project_type":          project_context.get("project_type", "unknown"),
            "grade":                 project_context.get("grade", "standard"),
            "region":                project_context.get("region", "KSA"),
            "total_area_m2":         project_context.get("total_area_m2", "unknown"),
            "included_trades":       ", ".join(project_context.get("included_trades", [])),
            "productivity_rates_json": json.dumps(productivity_rates, indent=2)[:3000],
            "qto_items_json":        json.dumps(qto_table, indent=2)[:5000],
            "cost_summary_json":     json.dumps(cost_summary, indent=2)[:1000],
        }
    )

    raw = llm.chat_json(
        agent_name=AGENT_NAME,
        messages=messages,
        job_id=job_id,
        temperature=0.1,
        max_tokens=3000,
    )

    timeline = _parse_timeline(raw, log)
    state["timeline"] = timeline
    state["status"] = "timeline_complete"

    elapsed = time.time() - t_start
    state.setdefault("agent_timings", {})[AGENT_NAME] = round(elapsed, 2)

    log.info(
        f"Agent #4 complete in {elapsed:.1f}s | "
        f"phases={len(timeline.get('phases', []))} "
        f"total={timeline.get('total_weeks', 0)} weeks"
    )
    return state


def _parse_timeline(raw: dict, log) -> Timeline:
    try:
        return {
            "phases":        raw.get("phases", []),
            "milestones":    raw.get("milestones", []),
            "total_weeks":   int(raw.get("total_weeks", 0)),
            "total_months":  float(raw.get("total_months", 0)),
            "critical_path": raw.get("critical_path", []),
            "assumptions":   raw.get("assumptions", []),
        }
    except Exception as e:
        log.warning(f"Timeline parse error: {e}")
        return {
            "phases": [], "milestones": [], "total_weeks": 0,
            "total_months": 0.0, "critical_path": [], "assumptions": [],
        }
