"""
core/logger.py
Structured logging for the Construction Estimator system.
Uses loguru for clean, color-coded, file-persisted logs.
Every agent logs with its name and job_id as context.
"""

import sys
import os
from pathlib import Path
from loguru import logger

_ROOT = Path(__file__).parent.parent
_LOGS_DIR = _ROOT / "logs"
_LOGS_DIR.mkdir(exist_ok=True)

# Remove default loguru handler
logger.remove()

# Console handler — colored, human-readable
logger.add(
    sys.stdout,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[agent]: <20}</cyan> | "
        "<cyan>{extra[job_id]: <12}</cyan> | "
        "{message}"
    ),
    level=os.getenv("LOG_LEVEL", "DEBUG"),
    colorize=True,
)

# File handler — full structured logs, rotated daily
logger.add(
    _LOGS_DIR / "estimator_{time:YYYY-MM-DD}.log",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{extra[agent]: <20} | {extra[job_id]: <12} | {message}"
    ),
    level="DEBUG",
    rotation="00:00",
    retention="14 days",
    compression="gz",
    enqueue=True,
)

# Error-only file — for quick error scanning
logger.add(
    _LOGS_DIR / "errors.log",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
        "{extra[agent]: <20} | {extra[job_id]: <12} | {message}\n{exception}"
    ),
    level="ERROR",
    rotation="50 MB",
    retention="30 days",
    enqueue=True,
)


def get_logger(agent_name: str, job_id: str = "system"):
    """
    Get a logger instance bound to a specific agent and job.

    Usage:
        log = get_logger("agent_00_intake", job_id="abc123")
        log.info("Project classified as office_fitout")
        log.error("Failed to parse document", exc_info=True)
    """
    return logger.bind(agent=agent_name, job_id=job_id)


def get_system_logger():
    """Get a logger for system-level operations (not tied to an agent)."""
    return logger.bind(agent="system", job_id="system")


# System logger for imports
system_log = get_system_logger()
