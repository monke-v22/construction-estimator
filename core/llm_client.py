"""
core/llm_client.py
Unified LLM client for all agents.
All calls route through AgentRouter (OpenAI-compatible endpoint).
Never call Anthropic or DeepSeek APIs directly.
"""

from __future__ import annotations
import json
import time
from typing import Any, Optional
from openai import OpenAI
from core.config import settings, AGENT_MODELS
from core.logger import get_logger

log = get_logger("llm_client")


class LLMClient:
    """
    Thin wrapper around AgentRouter's OpenAI-compatible endpoint.
    All agents use this client — never import openai directly in agent files.
    """

    def __init__(self):
        self._client = OpenAI(
            api_key=settings.agentrouter_api_key,
            base_url=settings.agentrouter_base_url,
        )

    def chat(
        self,
        agent_name: str,
        messages: list[dict],
        job_id: str = "system",
        temperature: float = 0.1,
        max_tokens: int = 8192,
        response_format: Optional[dict] = None,
        retries: int = 3,
        retry_delay: float = 5.0,
    ) -> str:
        """
        Send a chat completion request for the given agent.
        Model is automatically selected based on agent_name via AGENT_MODELS.

        Args:
            agent_name: e.g. "agent_00_intake" — used to pick the correct model
            messages: OpenAI-format list of {role, content} dicts
            job_id: for logging traceability
            temperature: 0.1 default (deterministic for estimation tasks)
            max_tokens: max output tokens
            response_format: e.g. {"type": "json_object"} for structured output
            retries: number of retry attempts on transient errors
            retry_delay: seconds between retries

        Returns:
            The assistant's response as a string.
        """
        log_ctx = get_logger("llm_client", job_id)
        model = AGENT_MODELS.get(agent_name, AGENT_MODELS["orchestrator"])

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        for attempt in range(1, retries + 1):
            try:
                log_ctx.debug(
                    f"Calling model={model} | agent={agent_name} | "
                    f"messages={len(messages)} | attempt={attempt}"
                )
                t0 = time.time()
                response = self._client.chat.completions.create(**kwargs)
                elapsed = time.time() - t0

                content = response.choices[0].message.content or ""
                tokens_in = response.usage.prompt_tokens if response.usage else 0
                tokens_out = response.usage.completion_tokens if response.usage else 0

                log_ctx.info(
                    f"OK | model={model} | {elapsed:.1f}s | "
                    f"in={tokens_in} out={tokens_out} tokens"
                )
                return content

            except Exception as e:
                log_ctx.warning(
                    f"Attempt {attempt}/{retries} failed: {type(e).__name__}: {e}"
                )
                if attempt < retries:
                    time.sleep(retry_delay * attempt)
                else:
                    log_ctx.error(f"All {retries} attempts failed for {agent_name}")
                    raise

        return ""  # unreachable but satisfies type checker

    def chat_json(
        self,
        agent_name: str,
        messages: list[dict],
        job_id: str = "system",
        temperature: float = 0.1,
        max_tokens: int = 8192,
    ) -> dict:
        """
        Like chat() but parses the response as JSON.
        Strips markdown code fences if present.
        Raises ValueError if JSON parsing fails.
        """
        raw = self.chat(
            agent_name=agent_name,
            messages=messages,
            job_id=job_id,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        # Strip ```json fences if model added them
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("```")[1]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"LLM returned invalid JSON for {agent_name}:\n{raw[:500]}"
            ) from e


# Singleton client — import this in all agents
llm = LLMClient()
