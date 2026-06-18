"""
tests/test_phase0_structure.py
Phase 0 structural tests — no API keys needed.
Verifies all core modules import, config loads, state is valid,
prompt loader works, and job manager CRUD functions correctly.
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


# ─────────────────────────────────────────────
# 1. Config
# ─────────────────────────────────────────────

def test_config_loads():
    """Config loads from .env without crashing."""
    from core.config import settings, MODELS, AGENT_MODELS, PROJECT_TYPES
    assert settings.agentrouter_base_url == "https://agentrouter.org/v1"
    assert "claude-opus-4-6" in MODELS.values()
    assert "deepseek-v4-pro" in MODELS.values()
    assert "agent_00_intake" in AGENT_MODELS
    assert "agent_06_proposal" in AGENT_MODELS
    assert "office_fitout" in PROJECT_TYPES
    print(f"  Config OK | env={settings.environment}")


def test_model_assignments():
    """Blueprint model assignments are enforced correctly."""
    from core.config import AGENT_MODELS, MODELS
    # Accuracy-critical agents must use Opus
    for agent in ["agent_00_intake", "agent_01_ingestion", "agent_02_qto", "agent_05_auditor"]:
        assert AGENT_MODELS[agent] == MODELS["opus"], \
            f"{agent} should use Opus, got {AGENT_MODELS[agent]}"
    # Efficiency agents use DeepSeek
    for agent in ["agent_03_pricing", "agent_04_timeline", "agent_06_proposal"]:
        assert AGENT_MODELS[agent] == MODELS["deep"], \
            f"{agent} should use DeepSeek, got {AGENT_MODELS[agent]}"
    print("  Model assignments match blueprint ✓")


# ─────────────────────────────────────────────
# 2. Logger
# ─────────────────────────────────────────────

def test_logger_creates_bound_instance():
    """Logger binds agent name and job_id without errors."""
    from core.logger import get_logger
    log = get_logger("test_agent", "TEST001")
    log.debug("Phase 0 logger test")
    log.info("Logger test passed")
    print("  Logger OK")


# ─────────────────────────────────────────────
# 3. State Schema
# ─────────────────────────────────────────────

def test_state_schema_instantiation():
    """EstimatorState TypedDict can be created and accessed correctly."""
    from core.state import EstimatorState, ProjectContext, QTOItem

    state: EstimatorState = {
        "job_id": "TEST001",
        "status": "intake",
        "user_description": "Office fit-out 500m2 premium grade Dubai",
        "uploaded_file_paths": [],
        "intake_confirmed": False,
        "checkpoint_1_approved": False,
        "checkpoint_2_approved": False,
        "checkpoint_3_approved": False,
        "errors": [],
        "warnings": [],
        "agent_timings": {},
    }
    assert state["job_id"] == "TEST001"
    assert state["status"] == "intake"
    assert isinstance(state["errors"], list)
    print("  State schema OK")


def test_project_context_schema():
    """ProjectContext TypedDict has all required fields."""
    from core.state import ProjectContext
    ctx: ProjectContext = {
        "project_type": "office_fitout",
        "grade": "premium",
        "total_area_m2": 500.0,
        "floors": ["Ground", "First"],
        "included_trades": ["civil", "mep", "finishing"],
        "excluded_trades": ["ffe"],
        "special_requirements": "Open plan, glass partitions",
        "client_notes": "Fast track, 12 weeks",
        "intake_confidence": "high",
    }
    assert ctx["project_type"] == "office_fitout"
    assert ctx["total_area_m2"] == 500.0
    print("  ProjectContext schema OK")


# ─────────────────────────────────────────────
# 4. Prompt Loader
# ─────────────────────────────────────────────

def test_prompt_loader_lists_files(tmp_path):
    """Prompt loader lists available .yaml files."""
    from core.prompt_loader import PromptLoader
    loader = PromptLoader()
    # May be empty before Phase 1 prompts are added — just ensure no crash
    files = loader.list_prompts()
    assert isinstance(files, list)
    print(f"  Prompt files found: {len(files)}")


def test_prompt_loader_renders_template(tmp_path):
    """Prompt loader renders Jinja2 variables correctly."""
    import yaml
    from core.prompt_loader import PromptLoader

    # Create a temp prompt file
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_data = {
        "agent": "test_agent",
        "version": "1.0",
        "system": "You are a {{ role }} expert. Region: {{ region }}.",
        "user": "Estimate this {{ project_type }} project."
    }
    (prompts_dir / "test_prompt.yaml").write_text(yaml.dump(prompt_data))

    # Override prompts dir for this test
    loader = PromptLoader()
    loader._cache = {}
    import core.prompt_loader as pl_module
    original_dir = pl_module._PROMPTS_DIR
    pl_module._PROMPTS_DIR = prompts_dir

    try:
        messages = loader.get_messages(
            "test_prompt.yaml",
            context={"role": "construction", "region": "GCC", "project_type": "office_fitout"}
        )
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "construction" in messages[0]["content"]
        assert "GCC" in messages[0]["content"]
        assert "office_fitout" in messages[1]["content"]
        print(f"  Prompt rendered: '{messages[0]['content'][:50]}...'")
    finally:
        pl_module._PROMPTS_DIR = original_dir


def test_prompt_loader_raises_on_missing_variable(tmp_path):
    """Prompt loader raises ValueError when a template variable is missing."""
    import yaml
    from core.prompt_loader import PromptLoader
    import core.prompt_loader as pl_module

    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    prompt_data = {
        "agent": "test_agent",
        "system": "Hello {{ missing_var }}",
        "user": "test"
    }
    (prompts_dir / "bad_prompt.yaml").write_text(yaml.dump(prompt_data))

    loader = PromptLoader()
    loader._cache = {}
    original_dir = pl_module._PROMPTS_DIR
    pl_module._PROMPTS_DIR = prompts_dir

    try:
        with pytest.raises(ValueError, match="missing_var"):
            loader.get_messages("bad_prompt.yaml", context={})
        print("  Missing variable correctly raises ValueError ✓")
    finally:
        pl_module._PROMPTS_DIR = original_dir


# ─────────────────────────────────────────────
# 5. Job Manager
# ─────────────────────────────────────────────

def test_job_manager_create_and_retrieve(tmp_path):
    """Job manager creates a job and retrieves it correctly."""
    from core.job_manager import JobManager

    db = JobManager(db_path=tmp_path / "test_jobs.db")
    job_id = db.create_job(
        user_description="Office fit-out 500m2 premium Dubai",
        uploaded_file_paths=["sample.pdf"],
    )
    assert len(job_id) == 8

    state = db.get_job(job_id)
    assert state is not None
    assert state["job_id"] == job_id
    assert state["status"] == "intake"
    assert state["user_description"] == "Office fit-out 500m2 premium Dubai"
    assert state["uploaded_file_paths"] == ["sample.pdf"]
    print(f"  Job created & retrieved: job_id={job_id}")


def test_job_manager_update_status(tmp_path):
    """Job manager correctly updates status."""
    from core.job_manager import JobManager

    db = JobManager(db_path=tmp_path / "test_jobs.db")
    job_id = db.create_job("Test project", [])
    db.update_status(job_id, "ingestion")

    state = db.get_job(job_id)
    assert state["status"] == "ingestion"
    print(f"  Status update OK: {job_id} → ingestion")


def test_job_manager_save_and_reload_state(tmp_path):
    """Job manager saves partial agent output and reloads correctly."""
    from core.job_manager import JobManager
    from core.state import EstimatorState

    db = JobManager(db_path=tmp_path / "test_jobs.db")
    job_id = db.create_job("Test project", [])

    state = db.get_job(job_id)
    state["project_context"] = {
        "project_type": "office_fitout",
        "grade": "premium",
        "total_area_m2": 350.0,
        "included_trades": ["civil", "mep", "finishing"],
        "intake_confidence": "high",
    }
    state["status"] = "qto"
    db.save_state(job_id, state)

    reloaded = db.get_job(job_id)
    assert reloaded["project_context"]["project_type"] == "office_fitout"
    assert reloaded["project_context"]["total_area_m2"] == 350.0
    assert reloaded["status"] == "qto"
    print(f"  State persistence OK: project_type={reloaded['project_context']['project_type']}")


def test_job_manager_error_tracking(tmp_path):
    """Job manager records errors correctly."""
    from core.job_manager import JobManager

    db = JobManager(db_path=tmp_path / "test_jobs.db")
    job_id = db.create_job("Test project", [])
    db.add_error(job_id, "agent_01_ingestion", "parse_error", "Failed to parse PDF")

    state = db.get_job(job_id)
    assert len(state["errors"]) == 1
    assert state["errors"][0]["agent"] == "agent_01_ingestion"
    assert state["errors"][0]["error_type"] == "parse_error"
    print(f"  Error tracking OK: {state['errors'][0]['message']}")


def test_job_manager_list_jobs(tmp_path):
    """Job manager lists all jobs correctly."""
    from core.job_manager import JobManager

    db = JobManager(db_path=tmp_path / "test_jobs.db")
    ids = [db.create_job(f"Project {i}", []) for i in range(3)]
    jobs = db.list_jobs()
    assert len(jobs) == 3
    assert all("job_id" in j for j in jobs)
    print(f"  Listed {len(jobs)} jobs ✓")


# ─────────────────────────────────────────────
# 6. Folder Structure
# ─────────────────────────────────────────────

def test_required_folders_exist():
    """All required project folders exist."""
    root = Path(__file__).parent.parent
    required = [
        "agents", "orchestrator", "parsers", "prompts",
        "knowledge_base", "samples", "previous_estimations",
        "templates", "outputs", "core", "config", "tests", "ui",
        "knowledge_base/by_project_type", "knowledge_base/by_trade",
        "samples/pdf", "samples/word", "samples/excel",
    ]
    missing = [d for d in required if not (root / d).exists()]
    assert not missing, f"Missing folders: {missing}"
    print(f"  All {len(required)} required folders exist ✓")


def test_core_files_exist():
    """All core module files exist."""
    root = Path(__file__).parent.parent
    required_files = [
        "core/config.py",
        "core/logger.py",
        "core/state.py",
        "core/llm_client.py",
        "core/prompt_loader.py",
        "core/job_manager.py",
        "core/test_connections.py",
        ".env.example",
        "requirements.txt",
        "docker-compose.yml",
        "HUMAN_TODO.md",
        "config/company.yaml",
    ]
    missing = [f for f in required_files if not (root / f).exists()]
    assert not missing, f"Missing files: {missing}"
    print(f"  All {len(required_files)} core files exist ✓")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
