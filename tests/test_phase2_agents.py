"""
tests/test_phase2_agents.py
Tests for Agent #0 (Intake) and Agent #2 (QTO).
All tests run WITHOUT API calls — tests logic, validation, and state flow.
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ─────────────────────────────────────────────
# Agent #0 — Intake — Unit Tests
# ─────────────────────────────────────────────

def test_agent00_validate_known_project_type():
    """_validate_and_normalise accepts valid project types."""
    from agents.agent_00_intake import _validate_and_normalise
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw = {
        "project_type": "office_fitout",
        "grade": "premium",
        "total_area_m2": 500.0,
        "floors": ["Floor 3"],
        "included_trades": ["civil", "mep", "finishing", "ffe"],
        "excluded_trades": [],
        "region": "KSA",
        "currency": "SAR",
        "special_requirements": "Open plan layout",
        "client_notes": "500m2 office fitout",
        "intake_confidence": "high",
        "confidence_reason": "Area clearly stated",
    }
    ctx = _validate_and_normalise(raw, log)
    assert ctx["project_type"] == "office_fitout"
    assert ctx["grade"] == "premium"
    assert ctx["total_area_m2"] == 500.0
    assert "civil" in ctx["included_trades"]
    assert ctx["intake_confidence"] == "high"
    assert ctx["currency"] == "SAR"
    print(f"  Validation OK: type={ctx['project_type']} confidence={ctx['intake_confidence']}")


def test_agent00_defaults_unknown_project_type():
    """_validate_and_normalise defaults unknown project type to 'other'."""
    from agents.agent_00_intake import _validate_and_normalise
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw = {
        "project_type": "underwater_palace",  # Unknown
        "grade": "luxury",
        "total_area_m2": None,
        "included_trades": [],
        "excluded_trades": [],
        "region": "UAE",
        "currency": "AED",
        "intake_confidence": "low",
    }
    ctx = _validate_and_normalise(raw, log)
    assert ctx["project_type"] == "other"
    assert ctx["grade"] == "luxury"
    assert ctx["intake_confidence"] == "low"
    print(f"  Unknown type defaulted to: {ctx['project_type']}")


def test_agent00_defaults_unknown_grade():
    """_validate_and_normalise defaults unknown grade to 'standard'."""
    from agents.agent_00_intake import _validate_and_normalise
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw = {
        "project_type": "retail",
        "grade": "ultra-mega-deluxe",  # Unknown
        "total_area_m2": 200.0,
        "included_trades": ["civil", "finishing"],
        "excluded_trades": [],
        "region": "UAE",
        "currency": "AED",
        "intake_confidence": "medium",
    }
    ctx = _validate_and_normalise(raw, log)
    assert ctx["grade"] == "standard"
    print(f"  Unknown grade defaulted to: {ctx['grade']}")


def test_agent00_area_none_when_not_stated():
    """total_area_m2 is None when not in LLM output."""
    from agents.agent_00_intake import _validate_and_normalise
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw = {
        "project_type": "renovation",
        "grade": "standard",
        "total_area_m2": None,
        "included_trades": ["civil"],
        "excluded_trades": [],
        "region": "KSA",
        "currency": "SAR",
        "intake_confidence": "low",
    }
    ctx = _validate_and_normalise(raw, log)
    assert ctx["total_area_m2"] is None
    print("  area_m2=None correctly preserved")


def test_agent00_negative_area_becomes_none():
    """Negative area values are rejected and set to None."""
    from agents.agent_00_intake import _validate_and_normalise
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw = {
        "project_type": "office_fitout",
        "grade": "standard",
        "total_area_m2": -50.0,
        "included_trades": [],
        "excluded_trades": [],
        "region": "KSA",
        "currency": "SAR",
        "intake_confidence": "medium",
    }
    ctx = _validate_and_normalise(raw, log)
    assert ctx["total_area_m2"] is None
    print("  Negative area correctly rejected → None")


def test_agent00_default_trades_by_project_type():
    """_default_trades returns correct trades per project type."""
    from agents.agent_00_intake import _default_trades

    assert set(_default_trades("office_fitout")) == {"civil", "mep", "finishing", "ffe"}
    assert set(_default_trades("mep_only")) == {"mep"}
    assert set(_default_trades("finishing_only")) == {"finishing"}
    assert set(_default_trades("full_fitout")) == {"civil", "mep", "finishing", "ffe", "external"}
    assert "civil" in _default_trades("other")
    print("  Default trades per project type ✓")


def test_agent00_filters_invalid_trades():
    """Only valid trade names pass through _validate_and_normalise."""
    from agents.agent_00_intake import _validate_and_normalise
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw = {
        "project_type": "office_fitout",
        "grade": "standard",
        "total_area_m2": 300.0,
        "included_trades": ["civil", "mep", "magic_trade", "plumbing_only"],
        "excluded_trades": ["fantasy_trade"],
        "region": "KSA",
        "currency": "SAR",
        "intake_confidence": "medium",
    }
    ctx = _validate_and_normalise(raw, log)
    assert "magic_trade" not in ctx["included_trades"]
    assert "plumbing_only" not in ctx["included_trades"]
    assert "fantasy_trade" not in ctx["excluded_trades"]
    assert "civil" in ctx["included_trades"]
    assert "mep" in ctx["included_trades"]
    print(f"  Valid trades kept: {ctx['included_trades']}")


def test_agent00_currency_defaulted_from_region():
    """Currency defaults from region when set to 'unknown'."""
    from agents.agent_00_intake import _validate_and_normalise
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    for region, expected_currency in [("UAE", "AED"), ("KSA", "SAR")]:
        raw = {
            "project_type": "retail",
            "grade": "premium",
            "total_area_m2": 200.0,
            "included_trades": ["civil"],
            "excluded_trades": [],
            "region": region,
            "currency": "unknown",
            "intake_confidence": "medium",
        }
        ctx = _validate_and_normalise(raw, log)
        assert ctx["currency"] == expected_currency, f"Expected {expected_currency} for {region}"
    print("  Currency defaulted from region ✓")


def test_agent00_empty_description_sets_error():
    """Agent #0 sets error state when description is empty."""
    from core.state import EstimatorState
    from agents.agent_00_intake import run

    state: EstimatorState = {
        "job_id": "TEST001",
        "status": "intake",
        "user_description": "",  # Empty
        "uploaded_file_paths": [],
        "intake_confirmed": False,
        "checkpoint_1_approved": False,
        "checkpoint_2_approved": False,
        "checkpoint_3_approved": False,
        "errors": [],
        "warnings": [],
        "agent_timings": {},
    }
    result = run(state)
    assert result["status"] == "error"
    assert len(result["errors"]) > 0
    assert result["errors"][0]["agent"] == "agent_00_intake"
    print(f"  Empty description error: {result['errors'][0]['message']}")


# ─────────────────────────────────────────────
# Agent #2 — QTO — Unit Tests
# ─────────────────────────────────────────────

def test_agent02_parse_valid_qto_items():
    """_parse_qto_items correctly parses valid items."""
    from agents.agent_02_qto import _parse_qto_items
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw_items = [
        {
            "item_id": "A.01",
            "trade": "civil",
            "description": "Gypsum board partitions 75mm",
            "unit": "m2",
            "quantity": 420.0,
            "source_doc": "boq.xlsx",
            "confidence": "high",
            "notes": "Floor-to-slab",
        },
        {
            "item_id": "B.01",
            "trade": "mep",
            "description": "Split A/C units 2.5 ton",
            "unit": "nr",
            "quantity": 22.0,
            "source_doc": "calculated",
            "confidence": "medium",
            "notes": "1 per 25m2, total area 550m2",
        },
        {
            "item_id": "C.01",
            "trade": "finishing",
            "description": "Ceramic floor tiles 600x600",
            "unit": "m2",
            "quantity": 350.0,
            "source_doc": "spec.docx",
            "confidence": "high",
            "notes": "",
        },
    ]

    items = _parse_qto_items(raw_items, log)
    assert len(items) == 3
    assert items[0]["trade"] == "civil"
    assert items[0]["quantity"] == 420.0
    assert items[1]["trade"] == "mep"
    assert items[1]["unit"] == "nr"
    assert items[2]["confidence"] == "high"
    print(f"  Parsed {len(items)} QTO items ✓")


def test_agent02_rejects_invalid_trade():
    """_parse_qto_items skips items with unknown trades."""
    from agents.agent_02_qto import _parse_qto_items
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw_items = [
        {"trade": "plumbing", "description": "Pipes", "unit": "lm",
         "quantity": 50, "source_doc": "doc", "confidence": "high"},
        {"trade": "finishing", "description": "Paint", "unit": "m2",
         "quantity": 200, "source_doc": "doc", "confidence": "high"},
    ]
    items = _parse_qto_items(raw_items, log)
    assert len(items) == 1
    assert items[0]["trade"] == "finishing"
    print(f"  Invalid trade rejected, kept {len(items)} valid item")


def test_agent02_normalises_unit_aliases():
    """_parse_qto_items converts 'sqm' and 'sq.m' to 'm2'."""
    from agents.agent_02_qto import _parse_qto_items
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw_items = [
        {"trade": "finishing", "description": "Tiles", "unit": "sqm",
         "quantity": 100, "source_doc": "doc", "confidence": "high"},
        {"trade": "civil", "description": "Screed", "unit": "sq.m",
         "quantity": 80, "source_doc": "doc", "confidence": "medium"},
    ]
    items = _parse_qto_items(raw_items, log)
    assert items[0]["unit"] == "m2"
    assert items[1]["unit"] == "m2"
    print("  Unit aliases normalised: sqm→m2 ✓")


def test_agent02_defaults_unknown_unit():
    """_parse_qto_items defaults unknown units to 'nr'."""
    from agents.agent_02_qto import _parse_qto_items
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw_items = [
        {"trade": "mep", "description": "AC units", "unit": "units",
         "quantity": 5, "source_doc": "doc", "confidence": "high"},
    ]
    items = _parse_qto_items(raw_items, log)
    assert items[0]["unit"] == "nr"
    print(f"  Unknown unit defaulted to: {items[0]['unit']}")


def test_agent02_auto_generates_item_ids():
    """_parse_qto_items generates item_ids when missing."""
    from agents.agent_02_qto import _parse_qto_items
    from core.logger import get_logger
    log = get_logger("test", "TEST")

    raw_items = [
        {"trade": "finishing", "description": "Paint", "unit": "m2",
         "quantity": 500, "source_doc": "doc", "confidence": "high"},
        {"trade": "finishing", "description": "Ceiling grid", "unit": "m2",
         "quantity": 400, "source_doc": "doc", "confidence": "high"},
    ]
    items = _parse_qto_items(raw_items, log)
    assert items[0]["item_id"].startswith("C.")
    assert items[1]["item_id"].startswith("C.")
    assert items[0]["item_id"] != items[1]["item_id"]
    print(f"  Auto-IDs: {items[0]['item_id']}, {items[1]['item_id']}")


def test_agent02_handles_no_project_context():
    """Agent #2 returns error state when project_context is missing."""
    from core.state import EstimatorState
    from agents.agent_02_qto import run

    state: EstimatorState = {
        "job_id": "TEST002",
        "status": "qto",
        "user_description": "Office fitout",
        "uploaded_file_paths": [],
        "project_context": None,   # Missing
        "extracted_data": {},
        "intake_confirmed": True,
        "checkpoint_1_approved": True,
        "checkpoint_2_approved": False,
        "checkpoint_3_approved": False,
        "errors": [],
        "warnings": [],
        "agent_timings": {},
    }
    result = run(state)
    assert result["status"] == "error"
    print("  Missing project_context correctly sets error state ✓")


# ─────────────────────────────────────────────
# Knowledge Base Tests
# ─────────────────────────────────────────────

def test_pricing_data_csv_exists_and_valid():
    """pricing_data.csv exists and has required columns."""
    import csv
    path = Path(__file__).parent.parent / "knowledge_base" / "pricing_data.csv"
    assert path.exists(), "pricing_data.csv missing"

    with open(path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    required_cols = {"project_type", "trade", "item", "unit", "rate_low", "rate_high", "currency", "region"}
    assert required_cols.issubset(set(reader.fieldnames)), f"Missing columns: {required_cols - set(reader.fieldnames)}"
    assert len(rows) >= 30, f"Too few rows: {len(rows)}"

    # Check both regions and currencies present
    regions = {r["region"] for r in rows}
    currencies = {r["currency"] for r in rows}
    assert "KSA" in regions and "UAE" in regions
    assert "SAR" in currencies and "AED" in currencies
    print(f"  pricing_data.csv: {len(rows)} rows | regions={regions} | currencies={currencies}")


def test_productivity_rates_json_exists():
    """productivity_rates.json exists and has all required trades."""
    import json
    path = Path(__file__).parent.parent / "knowledge_base" / "productivity_rates.json"
    assert path.exists(), "productivity_rates.json missing"

    with open(path) as f:
        data = json.load(f)

    assert "civil" in data
    assert "mep" in data
    assert "finishing" in data
    assert "ffe" in data
    assert "trade_sequence" in data
    assert len(data["trade_sequence"]) >= 5
    print(f"  productivity_rates.json: {len(data['trade_sequence'])} sequence steps ✓")


def test_benchmark_ranges_json_exists():
    """benchmark_ranges.json exists and covers key project types."""
    import json
    path = Path(__file__).parent.parent / "knowledge_base" / "benchmark_ranges.json"
    assert path.exists(), "benchmark_ranges.json missing"

    with open(path) as f:
        data = json.load(f)

    for pt in ["office_fitout", "retail", "hospitality", "residential"]:
        assert pt in data, f"Missing project type: {pt}"
        for grade in ["standard", "premium"]:
            assert grade in data[pt], f"Missing grade {grade} for {pt}"
            assert "sar_min" in data[pt][grade]
            assert "aed_min" in data[pt][grade]
    print(f"  benchmark_ranges.json: {len(data)} project types ✓")


def test_synthetic_samples_exist():
    """All 3 synthetic sample files were generated."""
    root = Path(__file__).parent.parent
    files = [
        root / "samples" / "excel" / "SYNTHETIC_office_fitout_boq_riyadh.xlsx",
        root / "samples" / "word" / "SYNTHETIC_office_fitout_specification_riyadh.docx",
        root / "samples" / "pdf" / "SYNTHETIC_scope_of_works_dubai_aed.pdf",
    ]
    for f in files:
        assert f.exists(), f"Missing: {f}"
        assert f.stat().st_size > 1000, f"File too small (possible corruption): {f}"
    print(f"  All 3 synthetic samples exist and non-empty ✓")


def test_agent00_prompt_yaml_renders():
    """Agent #0 prompt YAML renders without errors."""
    from core.prompt_loader import PromptLoader
    loader = PromptLoader()
    messages = loader.get_messages(
        "agent_00_intake.yaml",
        context={
            "project_types": "- office_fitout\n  - retail",
            "user_description": "Office fit-out 500m2 premium grade in Riyadh KSA",
        }
    )
    assert len(messages) == 2
    assert "EstimatePRO" in messages[0]["content"]
    assert "500m2" in messages[1]["content"]
    print("  Agent #0 prompt renders correctly ✓")


def test_agent02_prompt_yaml_renders():
    """Agent #2 prompt YAML renders without errors."""
    from core.prompt_loader import PromptLoader
    loader = PromptLoader()
    messages = loader.get_messages(
        "agent_02_qto.yaml",
        context={
            "project_type": "office_fitout",
            "grade": "premium",
            "region": "KSA",
            "currency": "SAR",
            "total_area_m2": "650",
            "area_note": "From documents",
            "floors": "Floor 12",
            "included_trades": "civil, mep, finishing, ffe",
            "excluded_trades": "None",
            "special_requirements": "Open plan",
            "project_context_json": "{}",
            "floor_areas_json": "[]",
            "dimensions_json": "[]",
            "material_specs_json": "[]",
            "boq_items_json": "[]",
            "scope_statements": "None",
            "flagged_gaps": "None",
        }
    )
    assert len(messages) == 2
    assert "office_fitout" in messages[0]["content"]
    assert "KSA" in messages[0]["content"]
    print("  Agent #2 prompt renders correctly ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
