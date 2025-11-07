# -*- coding: utf-8 -*-

import logging
from unittest.mock import patch

import pytest
from django.test.client import Client
from langchain_community.embeddings.fake import FakeEmbeddings
from langchain_core.messages.ai import AIMessage

from tests.base import build_mock_llm

logger = logging.getLogger(__name__)


@pytest.fixture
def client():
    """管理员登录的客户端"""
    return Client()


@pytest.fixture
def mock_default_models():
    # 将默认返回的Embedding模型和总结模型都mock
    with (
        patch(
            "aidev_agent.core.extend.models.llm_gateway.Embeddings.get_setup_instance",
            return_value=FakeEmbeddings(size=1024),
        ),
        patch(
            "aidev_agent.core.extend.models.llm_gateway.ChatModel.get_setup_instance",
            return_value=build_mock_llm([AIMessage(content="mocked summary")]),
        ),
    ):
        yield


@pytest.fixture
def patch_eager_task_get():
    with patch("celery.result.assert_will_not_block"):
        yield
