"""
core/job_manager.py
Manages job lifecycle and state persistence using SQLite.
Enables resume after HITL pause, crash recovery, and audit trail.
Used by FastAPI backend and LangGraph checkpointer.
"""

from __future__ import annotations
import json
import uuid
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from core.logger import get_logger
from core.state import EstimatorState

log = get_logger("job_manager")

_ROOT = Path(__file__).parent.parent
_DB_PATH = _ROOT / "jobs.db"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobManager:
    """
    SQLite-backed job store.
    Each job has a unique job_id and stores the full EstimatorState as JSON.
    Supports pause/resume at HITL checkpoints.
    """

    def __init__(self, db_path: Path = _DB_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id      TEXT PRIMARY KEY,
                    status      TEXT NOT NULL,
                    created_at  TEXT NOT NULL,
                    updated_at  TEXT NOT NULL,
                    state_json  TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status ON jobs(status)
            """)
            conn.commit()
        log.debug(f"Job DB initialized at {self.db_path}")

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def create_job(
        self,
        user_description: str,
        uploaded_file_paths: list[str],
    ) -> str:
        """Create a new job and return its job_id."""
        job_id = str(uuid.uuid4())[:8].upper()
        now = _now()

        initial_state: EstimatorState = {
            "job_id": job_id,
            "created_at": now,
            "status": "intake",
            "user_description": user_description,
            "uploaded_file_paths": uploaded_file_paths,
            "intake_confirmed": False,
            "checkpoint_1_approved": False,
            "checkpoint_2_approved": False,
            "checkpoint_3_approved": False,
            "errors": [],
            "warnings": [],
            "agent_timings": {},
        }

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, status, created_at, updated_at, state_json) "
                "VALUES (?, ?, ?, ?, ?)",
                (job_id, "intake", now, now, json.dumps(initial_state)),
            )
            conn.commit()

        log.info(f"Created job {job_id}")
        return job_id

    def get_job(self, job_id: str) -> Optional[EstimatorState]:
        """Load a job's state from DB. Returns None if not found."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT state_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()

        if not row:
            log.warning(f"Job {job_id} not found")
            return None

        return json.loads(row[0])

    def save_state(self, job_id: str, state: EstimatorState) -> None:
        """Persist the updated state for a job."""
        now = _now()
        status = state.get("status", "unknown")

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ?, state_json = ? "
                "WHERE job_id = ?",
                (status, now, json.dumps(state), job_id),
            )
            conn.commit()

        log.debug(f"Saved state for job {job_id} | status={status}")

    def update_status(self, job_id: str, status: str) -> None:
        """Update only the status field."""
        now = _now()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, updated_at = ? WHERE job_id = ?",
                (status, now, job_id),
            )
            # Also update status inside state_json
            row = conn.execute(
                "SELECT state_json FROM jobs WHERE job_id = ?", (job_id,)
            ).fetchone()
            if row:
                state = json.loads(row[0])
                state["status"] = status
                conn.execute(
                    "UPDATE jobs SET state_json = ? WHERE job_id = ?",
                    (json.dumps(state), job_id),
                )
            conn.commit()

        log.info(f"Job {job_id} → status={status}")

    def list_jobs(self, limit: int = 50) -> list[dict]:
        """Return recent jobs (metadata only, not full state)."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT job_id, status, created_at, updated_at "
                "FROM jobs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()

        return [
            {
                "job_id": r[0],
                "status": r[1],
                "created_at": r[2],
                "updated_at": r[3],
            }
            for r in rows
        ]

    def add_error(self, job_id: str, agent: str, error_type: str, message: str) -> None:
        """Append an error entry to the job's error list."""
        state = self.get_job(job_id)
        if not state:
            return
        state.setdefault("errors", []).append({
            "agent": agent,
            "error_type": error_type,
            "message": message,
            "timestamp": _now(),
        })
        self.save_state(job_id, state)

    def add_timing(self, job_id: str, agent: str, elapsed: float) -> None:
        """Record how long an agent took to run."""
        state = self.get_job(job_id)
        if not state:
            return
        state.setdefault("agent_timings", {})[agent] = round(elapsed, 2)
        self.save_state(job_id, state)


# Singleton
job_manager = JobManager()
