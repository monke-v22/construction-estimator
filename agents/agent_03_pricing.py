"""
agents/agent_03_pricing.py
Agent #3 — Pricing Research Agent
Model: deepseek-v4-pro (via AgentRouter)

Responsibilities:
  1. For each QTO item: query ChromaDB RAG for historical rates
  2. Run Tavily web search for current market pricing
  3. Call DeepSeek to synthesize RAG + web into unit rates
  4. Calculate line totals and cost summary
  5. Output fully priced BOQ
"""

from __future__ import annotations
import json
import time
from core.state import EstimatorState, PricedBOQItem, CostSummary
from core.llm_client import llm
from core.prompt_loader import prompt_loader
from core.logger import get_logger
from core.config import settings

AGENT_NAME = "agent_03_pricing"
TRADES = ["civil", "mep", "finishing", "ffe", "external"]


def run(state: EstimatorState) -> EstimatorState:
    job_id = state.get("job_id", "unknown")
    log = get_logger(AGENT_NAME, job_id)
    t_start = time.time()

    log.info("Agent #3 started — Pricing Research")
    state["status"] = "pricing"

    qto_table = state.get("qto_table", [])
    project_context = state.get("project_context", {})

    if not qto_table:
        log.warning("Empty QTO table — nothing to price")
        state["priced_boq"] = []
        state["cost_summary"] = _empty_summary(project_context)
        return state

    log.info(f"Pricing {len(qto_table)} QTO items")

    # ── Step 1: RAG retrieval ────────────────────────────────
    rag_results = _run_rag(qto_table, project_context, log)

    # ── Step 2: Web search for market prices ─────────────────
    web_results = _run_web_search(qto_table, project_context, log)

    # ── Step 3: DeepSeek pricing synthesis ───────────────────
    messages = prompt_loader.get_messages(
        "agent_03_pricing.yaml",
        context={
            "project_type":    project_context.get("project_type", "unknown"),
            "grade":           project_context.get("grade", "standard"),
            "region":          project_context.get("region", "KSA"),
            "currency":        project_context.get("currency", "SAR"),
            "qto_items_json":  json.dumps(qto_table, indent=2)[:6000],
            "rag_results_json": json.dumps(rag_results, indent=2)[:4000],
            "web_results_json": json.dumps(web_results, indent=2)[:3000],
        }
    )

    raw = llm.chat_json(
        agent_name=AGENT_NAME,
        messages=messages,
        job_id=job_id,
        temperature=0.1,
        max_tokens=8192,
    )

    # ── Step 4: Parse and validate output ────────────────────
    priced_items = _parse_priced_items(raw.get("priced_items", []), log)
    cost_summary = _build_cost_summary(priced_items, raw.get("cost_summary", {}), project_context)

    state["priced_boq"] = priced_items
    state["cost_summary"] = cost_summary
    state["status"] = "pricing_complete"

    elapsed = time.time() - t_start
    state.setdefault("agent_timings", {})[AGENT_NAME] = round(elapsed, 2)

    log.info(
        f"Agent #3 complete in {elapsed:.1f}s | "
        f"items={len(priced_items)} total={cost_summary.get('grand_total', 0):,.0f} "
        f"{cost_summary.get('currency', 'SAR')}"
    )
    return state


def _run_rag(qto_table: list, project_context: dict, log) -> dict:
    """Attempt RAG retrieval — gracefully skip if ChromaDB not ready."""
    try:
        from core.rag_retriever import retrieve_batch
        results = retrieve_batch(qto_table, project_context)
        log.info(f"RAG: retrieved results for {len(results)} items")
        return results
    except Exception as e:
        log.warning(f"RAG unavailable (run chroma_ingest.py first): {e}")
        return {}


def _run_web_search(qto_table: list, project_context: dict, log) -> list:
    """Run Tavily web searches for current market pricing."""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=settings.tavily_api_key)

        region = project_context.get("region", "KSA")
        grade = project_context.get("grade", "standard")
        currency = project_context.get("currency", "SAR")

        # Build focused search queries (one per trade, not per item)
        trades_in_qto = list({item["trade"] for item in qto_table})
        results = []

        for trade in trades_in_qto[:4]:  # Max 4 searches to save quota
            query = (
                f"{trade} fit-out construction unit rates {region} "
                f"{grade} grade {currency} 2024 2025"
            )
            log.debug(f"Tavily search: {query}")
            resp = client.search(query=query, max_results=3)
            for r in resp.get("results", []):
                results.append({
                    "trade": trade,
                    "title": r.get("title", ""),
                    "content": r.get("content", "")[:500],
                    "url": r.get("url", ""),
                })

        log.info(f"Web search: {len(results)} results across {len(trades_in_qto)} trades")
        return results

    except Exception as e:
        log.warning(f"Web search unavailable: {e}")
        return []


def _parse_priced_items(raw_items: list, log) -> list[PricedBOQItem]:
    items = []
    for r in raw_items:
        try:
            qty = float(r.get("quantity", 0))
            rate = float(r.get("unit_rate", 0))
            items.append({
                "item_id":       str(r.get("item_id", "")),
                "description":   str(r.get("description", "")),
                "trade":         str(r.get("trade", "")),
                "unit":          str(r.get("unit", "nr")),
                "quantity":      round(qty, 3),
                "unit_rate":     round(rate, 2),
                "line_total":    round(qty * rate, 2),
                "confidence":    str(r.get("confidence", "medium")),
                "source":        str(r.get("source", "estimated")),
                "rate_range_low":  float(r.get("rate_range_low", rate * 0.8)),
                "rate_range_high": float(r.get("rate_range_high", rate * 1.2)),
                "notes":         str(r.get("notes", "")),
            })
        except Exception as e:
            log.warning(f"Skipping invalid priced item: {e}")
    return items


def _build_cost_summary(items: list, raw_summary: dict, ctx: dict) -> CostSummary:
    currency = ctx.get("currency", "SAR")
    cont_pct = settings.default_contingency_pct
    oh_pct = settings.default_overhead_pct
    margin_pct = settings.default_margin_pct

    by_trade = {t: 0.0 for t in TRADES}
    for item in items:
        trade = item.get("trade", "civil")
        if trade in by_trade:
            by_trade[trade] += item.get("line_total", 0)

    subtotal = sum(by_trade.values())
    cont_amt = round(subtotal * cont_pct / 100, 2)
    oh_amt = round(subtotal * oh_pct / 100, 2)
    margin_amt = round(subtotal * margin_pct / 100, 2)
    grand_total = round(subtotal + cont_amt + oh_amt + margin_amt, 2)

    return {
        "by_trade":         {k: round(v, 2) for k, v in by_trade.items()},
        "subtotal":         round(subtotal, 2),
        "contingency_pct":  cont_pct,
        "contingency_amount": cont_amt,
        "overhead_pct":     oh_pct,
        "overhead_amount":  oh_amt,
        "margin_pct":       margin_pct,
        "margin_amount":    margin_amt,
        "grand_total":      grand_total,
        "currency":         currency,
    }


def _empty_summary(ctx: dict) -> CostSummary:
    return {
        "by_trade": {t: 0.0 for t in TRADES},
        "subtotal": 0.0,
        "contingency_pct": settings.default_contingency_pct,
        "contingency_amount": 0.0,
        "overhead_pct": settings.default_overhead_pct,
        "overhead_amount": 0.0,
        "margin_pct": settings.default_margin_pct,
        "margin_amount": 0.0,
        "grand_total": 0.0,
        "currency": ctx.get("currency", "SAR"),
    }
