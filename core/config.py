"""
core/config.py
Centralized configuration loader. Pydantic v2 compatible.
"""

import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_ROOT = Path(__file__).parent.parent
load_dotenv(_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # AgentRouter
    agentrouter_api_key: str = Field(default="", alias="AGENTROUTER_API_KEY")
    agentrouter_base_url: str = Field(default="https://agentrouter.org/v1", alias="AGENTROUTER_BASE_URL")

    # OpenAI (embeddings only)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")

    # Tavily
    tavily_api_key: str = Field(default="", alias="TAVILY_API_KEY")

    # App
    environment: str = Field(default="development", alias="ENVIRONMENT")
    log_level: str = Field(default="DEBUG", alias="LOG_LEVEL")

    # Paths
    outputs_dir: str = Field(default="outputs", alias="OUTPUTS_DIR")
    knowledge_base_dir: str = Field(default="knowledge_base", alias="KNOWLEDGE_BASE_DIR")
    samples_dir: str = Field(default="samples", alias="SAMPLES_DIR")
    chroma_db_path: str = Field(default="chroma_db", alias="CHROMA_DB_PATH")

    # RAG
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    chroma_collection_name: str = Field(default="construction_kb", alias="CHROMA_COLLECTION_NAME")
    rag_top_k: int = Field(default=10, alias="RAG_TOP_K")

    # Pricing defaults
    default_contingency_pct: float = Field(default=10.0, alias="DEFAULT_CONTINGENCY_PCT")
    default_overhead_pct: float = Field(default=12.0, alias="DEFAULT_OVERHEAD_PCT")
    default_margin_pct: float = Field(default=15.0, alias="DEFAULT_MARGIN_PCT")
    default_currency: str = Field(default="USD", alias="DEFAULT_CURRENCY")
    default_region: str = Field(default="GCC", alias="DEFAULT_REGION")

    # Jobs
    max_concurrent_jobs: int = Field(default=1, alias="MAX_CONCURRENT_JOBS")
    job_timeout_seconds: int = Field(default=10800, alias="JOB_TIMEOUT_SECONDS")


# Model assignment — matches blueprint exactly
MODELS = {
    "opus":  "claude-opus-4-6",
    "deep":  "deepseek-v4-pro",
    "haiku": "claude-haiku-4-5-20251001",
    "flash": "deepseek-v4-flash",
}

AGENT_MODELS = {
    "agent_00_intake":    MODELS["opus"],
    "agent_01_ingestion": MODELS["opus"],
    "agent_02_qto":       MODELS["opus"],
    "agent_03_pricing":   MODELS["deep"],
    "agent_04_timeline":  MODELS["deep"],
    "agent_05_auditor":   MODELS["opus"],
    "agent_06_proposal":  MODELS["deep"],
    "orchestrator":       MODELS["haiku"],
}

PROJECT_TYPES = [
    "office_fitout", "retail", "hospitality", "residential",
    "healthcare", "renovation", "mep_only", "finishing_only",
    "full_fitout", "other",
]

GRADES  = ["standard", "premium", "luxury"]
TRADES  = ["civil", "mep", "finishing", "ffe", "external"]
UNITS   = ["m2", "m3", "lm", "nr", "kg", "ls"]


def get_settings() -> Settings:
    return Settings()

settings = get_settings()
