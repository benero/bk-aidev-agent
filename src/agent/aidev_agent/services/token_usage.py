# -*- coding: utf-8 -*-
"""Token usage callback helpers for AIDev agent SDK."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from aidev_agent.packages.resource_manager.registry import ResourceManagerProtocol

logger = logging.getLogger(__name__)


@runtime_checkable
class TokenUsageSink(Protocol):
    def record_token_usage(self, payload: dict[str, Any]) -> None: ...


@dataclass
class BKAidevTokenUsageSink(TokenUsageSink):
    """Report token usage to the AIDev OpenAPI through the resource manager client."""

    resource_manager: ResourceManagerProtocol
    username: str | None = None

    def record_token_usage(self, payload: dict[str, Any]) -> None:
        session_code = payload.get("session_code") or ""
        message_id = payload.get("message_id") or ""
        if not session_code or not message_id:
            return

        try:
            self.resource_manager.get_client().api.create_chat_session_token_usage(
                json=payload,
                headers={"X-BKAIDEV-USER": self.username or ""},
            )
        except Exception:
            logger.exception("failed to report token usage: payload=%s", payload)


def _normalize_token_usage(response: LLMResult) -> dict[str, int] | None:
    usage = None
    if response.llm_output:
        for key in ("token_usage", "usage"):
            if key in response.llm_output and response.llm_output[key]:
                usage = response.llm_output[key]
                break

    if usage is None and response.generations:
        for generation_group in reversed(response.generations):
            for generation in reversed(generation_group):
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    break
            if usage:
                break

    if usage is None:
        return None

    if hasattr(usage, "to_dict_recursive"):
        usage = usage.to_dict_recursive()
    elif hasattr(usage, "model_dump"):
        usage = usage.model_dump()
    elif hasattr(usage, "dict"):
        usage = usage.dict()

    if not isinstance(usage, dict):
        return None

    input_tokens = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    output_tokens = int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or input_tokens + output_tokens)
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


class TokenUsageCallbackHandler(BaseCallbackHandler):
    """Collect token usage from LLM results and forward it to a sink."""

    def __init__(self, sink: TokenUsageSink, metadata: dict[str, Any] | None = None):
        super().__init__()
        self._sink = sink
        self._metadata = metadata or {}

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        usage = _normalize_token_usage(response)
        if not usage:
            return

        message_id = ""
        if response.generations and response.generations[-1]:
            last_generation = response.generations[-1][-1]
            message = getattr(last_generation, "message", None)
            message_id = getattr(message, "id", "") or ""

        payload = {
            **self._metadata,
            **usage,
            "message_id": message_id or kwargs.get("run_id", ""),
            "run_id": str(kwargs.get("run_id", "") or ""),
            "model": (response.llm_output or {}).get("model_name", ""),
        }
        try:
            self._sink.record_token_usage(payload)
        except Exception:
            # token 统计失败不应影响主流程
            return
