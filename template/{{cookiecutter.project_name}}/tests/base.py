# -*- coding: utf-8 -*-
"""
TencentBlueKing is pleased to support the open source community by making
蓝鲸智云 - AIDev (BlueKing - AIDev) available.
Copyright (C) 2025 THL A29 Limited,
a Tencent company. All rights reserved.
Licensed under the MIT License (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at http://opensource.org/licenses/MIT
Unless required by applicable law or agreed to in writing,
software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the License for the
specific language governing permissions and limitations under the License.
We undertake not to change the open source license (MIT license) applicable
to the current version of the project delivered to anyone in the future.
"""

from blueapps.utils.request_provider import get_local_request  # noqa
from django.conf import settings
from django.test import TestCase as _TestCase

from time import sleep
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from langchain_community.chat_models.fake import FakeMessagesListChatModel
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.base import LanguageModelInput
from langchain_core.messages.ai import AIMessage, AIMessageChunk, UsageMetadata
from langchain_core.messages.base import BaseMessage
from langchain_core.outputs import ChatGenerationChunk
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator


class TestCase(_TestCase):
    """
    Base Test Case
    """

    app_code = settings.APP_CODE
    app_secret = settings.SECRET_KEY


class EnhancedFakeMessagesListChatModel(FakeMessagesListChatModel):
    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    streaming: bool = True
    contents: list[str] = Field(default_factory=list)  # noqa
    i: int = 0
    iter_response_content: bool = True

    @model_validator(mode="before")
    def format_message(cls, v: dict[str, Any]):
        if "responses" not in v:
            v["responses"] = []
        for content in v.pop("contents", []):
            v["responses"].append(
                AIMessage(
                    content=content,
                    usage_metadata=UsageMetadata(
                        input_tokens=1, output_tokens=len(content), total_tokens=1 + len(content)
                    ),
                )
            )
        return v

    def predict_messages(
        self, messages: List[BaseMessage], *, stop: Sequence[str] | None = None, **kwargs: Any
    ) -> BaseMessage:
        kwargs["stream"] = self.streaming
        result = super().predict_messages(messages, stop=stop, **kwargs)
        return result

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return super().__call__(*args, **kwds)

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = self.responses[self.i]
        sleep(0.05)
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        if isinstance(response, AIMessage):
            if response.tool_calls:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=response.content, tool_calls=response.tool_calls)
                )
            elif response.additional_kwargs:
                if self.iter_response_content:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=response.content, additional_kwargs=response.additional_kwargs)
                    )
                else:
                    for each in self.responses:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=each.content, additional_kwargs=each.additional_kwargs)
                        )
            else:
                if self.iter_response_content:
                    for each in str(response.content):
                        yield ChatGenerationChunk(message=AIMessageChunk(content=each))
                else:
                    for each in self.responses:
                        yield ChatGenerationChunk(
                            message=AIMessageChunk(content=each.content, additional_kwargs=each.additional_kwargs)
                        )

    def get_token_ids(self, text: str) -> list[int]:
        return [0] * len(text)

    def bind_tools(
        self, tools: Sequence[Dict[str, Any] | type[BaseModel] | Callable[..., Any] | BaseTool], **kwargs: Any
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        return self


def build_mock_llm(responses: list[BaseMessage], iter_response_content: bool = True, model_name: str = "gpt-3.5-turbo"):
    return EnhancedFakeMessagesListChatModel(
        responses=responses, iter_response_content=iter_response_content, model_name=model_name
    )
