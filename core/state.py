"""
core/state.py
Shared state schema for the LangGraph orchestrator.
All agents read from and write to this single state object.
Uses Pydantic TypedDict for type-safe access across the entire pipeline.
"""

from __future__ import annotations
from typing import Any, Optional
from typing_extensions import TypedDict


# ------------------------------------------------------------------
# Sub-schemas (nested data structures per agent output)
# ------------------------------------------------------------------

class ProjectContext(TypedDict, total=False):
    """Output of Agent #0 — Project Intake Agent."""
    project_type: str           # office_fitout | retail | hospitality | etc.
    grade: str                  # standard | premium | luxury
    total_area_m2: Optional[float]
    floors: list[str]
    included_trades: list[str]  # civil | mep | finishing | ffe | external
    excluded_trades: list[str]
    special_requirements: str
    client_notes: str
    intake_confidence: str      # high | medium | low


class DocumentInfo(TypedDict, total=False):
    """Metadata for a single parsed document."""
    file_path: str
    document_type: str          # pdf | docx | xlsx | dxf | dwg | scan | image
    raw_text: str
    tables: list[dict]
    dimensions: list[dict]
    material_specs: list[dict]
    floor_areas: list[dict]
    images_extracted: list[str]
    confidence: float
    flagged_gaps: list[str]


class ExtractedData(TypedDict, total=False):
    """Output of Agent #1 — Document Ingestion Agent."""
    documents: list[DocumentInfo]
    combined_text: str
    all_tables: list[dict]
    all_dimensions: list[dict]
    all_material_specs: list[dict]
    all_floor_areas: list[dict]
    overall_confidence: float
    flagged_gaps: list[str]


class QTOItem(TypedDict, total=False):
    """A single line in the Quantity Takeoff table."""
    item_id: str
    trade: str                  # civil | mep | finishing | ffe | external
    description: str
    unit: str                   # m2 | m3 | lm | nr | kg | ls
    quantity: float
    source_doc: str
    confidence: str             # high | medium | low
    notes: str


class PricedBOQItem(TypedDict, total=False):
    """A single priced line item — output of Agent #3."""
    item_id: str
    description: str
    trade: str
    unit: str
    quantity: float
    unit_rate: float
    line_total: float
    confidence: str             # high | medium | low
    source: str                 # rag | web | combined
    rate_range_low: float
    rate_range_high: float
    notes: str


class CostSummary(TypedDict, total=False):
    """Rolled-up cost summary — output of Agent #3."""
    by_trade: dict[str, float]
    subtotal: float
    contingency_pct: float
    contingency_amount: float
    overhead_pct: float
    overhead_amount: float
    margin_pct: float
    margin_amount: float
    grand_total: float
    currency: str


class TimelinePhase(TypedDict, total=False):
    """A single phase in the project timeline."""
    phase_id: str
    name: str
    trades: list[str]
    start_week: int
    end_week: int
    duration_weeks: int
    milestones: list[str]


class Timeline(TypedDict, total=False):
    """Output of Agent #4 — Timeline & Schedule Agent."""
    phases: list[TimelinePhase]
    milestones: list[dict]
    total_weeks: int
    total_months: float
    critical_path: list[str]
    assumptions: list[str]


class AuditFlag(TypedDict, total=False):
    """A single audit finding from Agent #5."""
    item_id: str
    flag_type: str              # price_outlier | missing_item | unit_error | scope_gap
    description: str
    severity: str               # critical | warning | info
    suggestion: str


class AuditReport(TypedDict, total=False):
    """Output of Agent #5 — QA / Auditor Agent."""
    flagged_items: list[AuditFlag]
    anomalies: list[str]
    missing_items: list[str]
    price_outliers: list[dict]
    scope_gaps: list[str]
    overall_confidence_score: float  # 0.0 – 1.0
    recommendation: str


class HITLEdit(TypedDict, total=False):
    """Tracks human edits made at any HITL checkpoint."""
    checkpoint: int             # 1 | 2 | 3
    edited_at: str              # ISO timestamp
    changes: list[dict]         # list of {field, old_value, new_value}
    approved_by: str
    notes: str


# ------------------------------------------------------------------
# Master workflow state — passed to every LangGraph node
# ------------------------------------------------------------------

class EstimatorState(TypedDict, total=False):
    """
    Master state object shared across all 7 agents in the LangGraph workflow.
    Each agent reads what it needs and writes its output fields.
    Fields marked Optional may not exist early in the pipeline.
    """

    # ---- Job metadata ----
    job_id: str
    created_at: str
    status: str                 # intake | ingestion | qto | pricing | timeline | audit | proposal | complete | error

    # ---- User inputs ----
    user_description: str       # Free text from user (input to Agent #0)
    uploaded_file_paths: list[str]  # Paths to uploaded documents

    # ---- Agent #0 output ----
    project_context: Optional[ProjectContext]
    intake_confirmed: bool      # True after user confirms classification

    # ---- Agent #1 output ----
    extracted_data: Optional[ExtractedData]
    checkpoint_1_approved: bool
    checkpoint_1_edits: Optional[HITLEdit]

    # ---- Agent #2 output ----
    qto_table: Optional[list[QTOItem]]
    checkpoint_2_approved: bool
    checkpoint_2_edits: Optional[HITLEdit]

    # ---- Agent #3 output ----
    priced_boq: Optional[list[PricedBOQItem]]
    cost_summary: Optional[CostSummary]

    # ---- Agent #4 output ----
    timeline: Optional[Timeline]

    # ---- Agent #5 output ----
    audit_report: Optional[AuditReport]
    checkpoint_3_approved: bool
    checkpoint_3_edits: Optional[HITLEdit]

    # ---- Agent #6 output ----
    proposal_html_path: Optional[str]
    proposal_pdf_path: Optional[str]

    # ---- Error tracking ----
    errors: list[dict]          # list of {agent, error_type, message, timestamp}
    warnings: list[str]

    # ---- Audit trail ----
    agent_timings: dict[str, float]  # {agent_name: elapsed_seconds}
