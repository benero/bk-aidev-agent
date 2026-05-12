# -*- coding: utf-8 -*-
"""Session 管理。

封装 chat session 的取/建、内容持久化；底层 client 通过
``resource_manager().get_client()`` 获取（应用态 + ``X-BKAIDEV-USER`` header），
不直接依赖 ``aidev_agent.api.bk_aidev.BKAidevApi``。
"""

from __future__ import annotations

import hashlib
from logging import getLogger
from typing import Iterable

from aidev_agent.enums import PromptRole
from aidev_agent.packages.resource_manager import resource_manager
from aidev_agent.pydantic_models import ChatPrompt
from bkapi_client_core.exceptions import HTTPResponseError
from django.conf import settings
from langchain_core.stores import ByteStore, InMemoryByteStore

from ..constants import AGUI_PROTOCOL_VERSION

logger = getLogger(__name__)


class SessionManager:
    """会话管理：取/建 session、持久化 chat history、保存 AI 回复。

    所有 HTTP 调用经由 ``resource_manager().get_client()``，业务侧 import 上不再耦合
    ``BKAidevApi``；用户名通过 ``X-BKAIDEV-USER`` header 透传给后端识别用户。
    """

    _shared_session_file_stores: dict[tuple[str, str, str], ByteStore] = {}

    def __init__(self, username: str, agent_code: str | None = None):
        self.username = username
        self.agent_code = agent_code or settings.APP_CODE

    @staticmethod
    def generate_session_code(username: str, agent_code: str, thread_id: str) -> str:
        """``MD5(username:agent_code:thread_id)``，确保 session_code 长度固定且对相同三元组稳定。"""
        raw_string = f"{username}:{agent_code}:{thread_id}"
        return hashlib.md5(raw_string.encode()).hexdigest()

    def _client(self):
        return resource_manager().get_client()

    def _user_headers(self) -> dict:
        return {"X-BKAIDEV-USER": self.username}

    def get_or_create_by_thread_id(self, thread_id: str) -> str:
        """根据 ``thread_id`` 取回 session_code；不存在则新建（``protocol_version`` 由模块常量决定）。

        404 之外的 HTTPResponseError 直接抛出，由调用方决定降级策略。
        """
        session_code = self.generate_session_code(self.username, self.agent_code, thread_id)
        client = self._client()
        try:
            result = client.api.retrieve_chat_session(
                path_params={"session_code": session_code},
                headers=self._user_headers(),
            )
            if result.get("data"):
                return session_code
        except HTTPResponseError as err:
            logger.warning("Error retrieving chat session: %s", err)
            if err.response_status_code != 404:
                raise
            client.api.create_chat_session(
                json={
                    "session_code": session_code,
                    "session_name": f"Thread-{thread_id[:8]}",
                    "protocol_version": AGUI_PROTOCOL_VERSION,
                },
                headers=self._user_headers(),
            )
            return session_code
        return session_code

    def save_content(
        self,
        session_code: str,
        role: str,
        content: str,
        *,
        extra: dict | None = None,
        status: str = "success",
    ) -> dict:
        """保存单条会话内容到 BKAidev；返回后端 ``data`` 字段。"""
        data: dict = {"session_code": session_code, "role": role, "content": content, "status": status}
        if extra:
            data["extra"] = extra
        client = self._client()
        result = client.api.create_chat_session_content(json=data, headers=self._user_headers())
        return result.get("data", {})

    def save_chat_history(self, session_code: str, chat_history: Iterable[ChatPrompt] | None) -> None:
        """按顺序持久化 ``chat_history`` 中的每条 prompt；空列表 / None 直接返回。"""
        for prompt in chat_history or []:
            self.save_content(session_code=session_code, role=prompt.role, content=prompt.content)

    def save_ai_response(self, session_code: str, result: dict) -> None:
        """从非流式 LLM 响应中提取 ``choices[0].delta.content`` 并写回；空内容跳过。"""
        content = ""
        if "choices" in result and result["choices"]:
            delta = result["choices"][0].get("delta", {})
            content = delta.get("content", "")

        if content:
            self.save_content(session_code=session_code, role=PromptRole.AI.value, content=content)

    def _file_store_key(self, session_code: str) -> tuple[str, str, str]:
        return (self.username, self.agent_code, session_code)

    def get_session_file_store(self, session_code: str) -> ByteStore:
        key = self._file_store_key(session_code)
        if key not in self._shared_session_file_stores:
            self._shared_session_file_stores[key] = InMemoryByteStore()
        return self._shared_session_file_stores[key]

    def add_file_to_store(self, session_code: str, file_name: str, content: bytes) -> None:
        store = self.get_session_file_store(session_code)
        store.mset([(file_name, content)])

    def clear_session_file_store(self, session_code: str) -> None:
        self._shared_session_file_stores.pop(self._file_store_key(session_code), None)
