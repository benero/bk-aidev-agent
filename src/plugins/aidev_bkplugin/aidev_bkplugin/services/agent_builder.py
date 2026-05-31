# -*- coding: utf-8 -*-
"""``ChatCompletionAgent`` 构建器。

``AgentBuilder`` 收敛 4 种构建路径（by_session_code / by_chat_history / by_thread_id /
by_thread_id_with_chat_history），统一通过私方法 ``_build_session_agent_for_thread`` 装配
SESSION 路径，``event_handler`` / ``checkpointer`` 集中注入。
"""

from __future__ import annotations

from logging import getLogger

from aidev_agent.enums import AgentBuildType, PromptRole
from aidev_agent.pydantic_models import ChatPrompt
from aidev_agent.services.agent import AgentInstanceFactory, ChatCompletionAgent
from aidev_agent.services.common_agent import common_agent_factory
from aidev_agent.services.event_handlers import AGUISessionWriter
from django.conf import settings

from .agent_config import AgentConfigFetcher
from .agent_helpers import AgentHelper
from .agent_session import SessionManager

logger = getLogger(__name__)


class AgentBuilder:
    """构建 ``ChatCompletionAgent`` 的 OO 入口。

    - ``by_session_code``：直接以已有 ``session_code`` 走 SESSION 路径。
    - ``by_chat_history``：DIRECT 路径，把 role_prompts 自动前置。
    - ``by_thread_id`` / ``by_thread_id_with_chat_history``：先 ensure session，再走 SESSION 路径。
    """

    def __init__(
        self,
        username: str = "",
        agent_code: str | None = None,
        session_manager: SessionManager | None = None,
    ):
        self.username = username
        self.agent_code = agent_code or settings.APP_CODE
        self.session_manager = session_manager or SessionManager(username=username, agent_code=agent_code)

    def by_session_code(
        self,
        session_code: str,
        *,
        version: str | None = None,
        channel_type: str | None = None,
    ) -> ChatCompletionAgent:
        return self._build_session_agent_for_thread(session_code, version=version, channel_type=channel_type)

    def by_chat_history(
        self,
        chat_history: list[ChatPrompt],
        *,
        version: str | None = None,
        channel_type: str | None = None,
    ) -> ChatCompletionAgent:
        role_contents = AgentConfigFetcher.get_role_info(username=self.username)
        if role_contents:
            chat_history = role_contents + list(chat_history)
        agent_cls = common_agent_factory.get()

        return AgentInstanceFactory.build_agent(
            build_type=AgentBuildType.DIRECT,
            session_context_data=[each.model_dump() for each in chat_history],
            agent_cls=agent_cls,
            checkpointer=AgentHelper.get_checkpointer(),
            username=self.username,
            version=version,
            channel_type=channel_type,
        )

    def by_thread_id(
        self,
        thread_id: str,
        input_text: str,
        *,
        save_content: bool = True,
        version: str | None = None,
        channel_type: str | None = None,
    ) -> tuple[ChatCompletionAgent, str]:
        session_code = self.session_manager.get_or_create_by_thread_id(thread_id)
        if save_content and input_text:
            self.session_manager.save_content(
                session_code=session_code,
                role=PromptRole.USER.value,
                content=input_text,
            )
        return self._build_session_agent_for_thread(
            session_code,
            version=version,
            channel_type=channel_type,
        ), session_code

    def by_thread_id_with_chat_history(
        self,
        thread_id: str,
        chat_history: list[ChatPrompt],
        *,
        version: str | None = None,
        channel_type: str | None = None,
    ) -> tuple[ChatCompletionAgent, str]:
        session_code = self.session_manager.get_or_create_by_thread_id(thread_id)
        self.session_manager.save_chat_history(session_code, chat_history)
        return self._build_session_agent_for_thread(
            session_code,
            version=version,
            channel_type=channel_type,
        ), session_code

    def _build_session_agent_for_thread(
        self,
        session_code: str,
        *,
        version: str | None = None,
        channel_type: str | None = None,
    ) -> ChatCompletionAgent:
        """SESSION 路径 agent 装配；client 取自 ``AgentHelper.get_client()``（应用态）。"""
        agent_cls = common_agent_factory.get()
        event_handler = AGUISessionWriter(
            session_code=session_code, client=AgentHelper.get_client(), username=self.username
        )
        return AgentInstanceFactory.build_agent(
            build_type=AgentBuildType.SESSION,
            session_code=session_code,
            agent_cls=agent_cls,
            checkpointer=AgentHelper.get_checkpointer(),
            event_handler=event_handler,
            username=self.username,
            version=version,
            channel_type=channel_type,
        )
