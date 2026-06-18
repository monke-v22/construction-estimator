"""
core/prompt_loader.py
Loads agent prompts from /prompts/*.yaml files.
Prompts are NEVER hardcoded in agent files — always loaded here.
Supports Jinja2-style variable substitution in prompt templates.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import Any
import yaml
from jinja2 import Environment, BaseLoader, StrictUndefined
from core.logger import get_logger

log = get_logger("prompt_loader")

_ROOT = Path(__file__).parent.parent
_PROMPTS_DIR = _ROOT / "prompts"


class PromptLoader:
    """
    Loads YAML prompt files and renders them with context variables.

    YAML prompt file format:
    ─────────────────────────
    agent: agent_00_intake
    version: "1.0"
    description: "Classifies project type from user description"

    system: |
      You are a senior construction estimator...
      Project types: {{ project_types }}

    user: |
      User description: {{ user_description }}
      Please classify this project.
    ─────────────────────────
    """

    def __init__(self):
        self._cache: dict[str, dict] = {}
        self._jinja_env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,  # Raise if a variable is missing
        )

    def _load_yaml(self, prompt_file: str) -> dict:
        """Load and cache a YAML prompt file."""
        if prompt_file in self._cache:
            return self._cache[prompt_file]

        path = _PROMPTS_DIR / prompt_file
        if not path.exists():
            raise FileNotFoundError(
                f"Prompt file not found: {path}\n"
                f"Expected location: prompts/{prompt_file}"
            )

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        self._cache[prompt_file] = data
        log.debug(f"Loaded prompt: {prompt_file} (agent={data.get('agent', 'unknown')})")
        return data

    def get_messages(
        self,
        prompt_file: str,
        context: dict[str, Any] = None,
    ) -> list[dict]:
        """
        Load a prompt YAML and return OpenAI-format messages list.
        Renders Jinja2 variables with the provided context dict.

        Args:
            prompt_file: filename like "agent_00_intake.yaml"
            context: dict of variables to inject into the prompt template

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        """
        context = context or {}
        data = self._load_yaml(prompt_file)

        messages = []

        if "system" in data:
            system_content = self._render(data["system"], context)
            messages.append({"role": "system", "content": system_content})

        if "user" in data:
            user_content = self._render(data["user"], context)
            messages.append({"role": "user", "content": user_content})

        if not messages:
            raise ValueError(
                f"Prompt file {prompt_file} has neither 'system' nor 'user' key."
            )

        return messages

    def get_system(self, prompt_file: str, context: dict[str, Any] = None) -> str:
        """Return only the system prompt string."""
        context = context or {}
        data = self._load_yaml(prompt_file)
        if "system" not in data:
            raise ValueError(f"No 'system' key in {prompt_file}")
        return self._render(data["system"], context)

    def get_user(self, prompt_file: str, context: dict[str, Any] = None) -> str:
        """Return only the user prompt string."""
        context = context or {}
        data = self._load_yaml(prompt_file)
        if "user" not in data:
            raise ValueError(f"No 'user' key in {prompt_file}")
        return self._render(data["user"], context)

    def _render(self, template_str: str, context: dict) -> str:
        """Render a Jinja2 template string with the given context."""
        try:
            tmpl = self._jinja_env.from_string(template_str)
            return tmpl.render(**context)
        except Exception as e:
            raise ValueError(
                f"Failed to render prompt template: {e}\n"
                f"Context keys provided: {list(context.keys())}"
            ) from e

    def list_prompts(self) -> list[str]:
        """List all available prompt YAML files."""
        if not _PROMPTS_DIR.exists():
            return []
        return [f.name for f in _PROMPTS_DIR.glob("*.yaml")]

    def reload(self):
        """Clear cache — forces reload from disk on next access."""
        self._cache.clear()
        log.info("Prompt cache cleared")


# Singleton — import this in all agents
prompt_loader = PromptLoader()
