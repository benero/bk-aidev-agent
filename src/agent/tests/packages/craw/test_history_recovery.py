# -*- coding: utf-8 -*-
"""真实 writer → factory 账本 → Craw 下一轮的失败恢复契约。"""

import json
from types import SimpleNamespace

import pytest
from ag_ui.core import (
    CustomEvent,
    EventType,
    RunErrorEvent,
    RunFinishedEvent,
    TextMessageContentEvent,
    TextMessageStartEvent,
)
from aidev_agent.core.ag_ui.types import SessionPersistenceEventNames
from aidev_agent.enums import AgentType
from aidev_agent.packages.craw import CrawCompletionAgent, OpenClawBackend
from aidev_agent.packages.craw.agent import build_openclaw_history
from aidev_agent.packages.craw.openclaw_ws import OpenClawWSError, OpenClawWSSession
from aidev_agent.services.agent.factory import _FACTORY_TOKEN, AgentInstanceFactory
from aidev_agent.services.agent.registry import AgentBuildContext
from aidev_agent.services.event_handlers.base import BaseSessionWriter
from aidev_agent.utils.event import RunId


class RecordingWriter(BaseSessionWriter):
    def __init__(self, records, turn_id):
        super().__init__(session_code="history-recovery", turn_id=turn_id)
        self.records = records

    def _do_create_content(self, payload, headers):
        self.records.append(payload)
        return len(self.records)


class HistoryResourceManager:
    def __init__(self, records):
        self.records = records

    def get_chat_session_contents(self, _session_code):
        return self.records

    def resolve_access_token(self, _username):
        return "test-token"


def build_from_saved_records(records, turn_id, model="openclaw"):
    manager = HistoryResourceManager(records)
    factory = AgentInstanceFactory(
        agent_code="test-agent",
        session_code="history-recovery",
        resource_manager=manager,
        username="test-user",
        _token=_FACTORY_TOKEN,
    )
    factory.get_agent_config = lambda _code: SimpleNamespace(command_agent_mapping={})
    base_args = factory._build_from_session()
    writer = RecordingWriter(records, turn_id)
    return CrawCompletionAgent(backend=OpenClawBackend(transport="ws", model=model)).build(
        AgentBuildContext(
            agent_code="test-agent",
            agent_type=AgentType.CHAT,
            session_code="history-recovery",
            username="test-user",
            resource_manager=manager,
            session_context_data=base_args["session_context_data"],
            event_handler=writer,
        )
    )


@pytest.mark.parametrize(
    "model,expected_key", [("openclaw", "history-recovery"), ("openclaw:ops", "agent:ops:history-recovery")]
)
def test_connection_failure_does_not_permanently_poison_next_turn(monkeypatch, model, expected_key):
    records = []
    kernel_history = []
    sent_messages = []
    connect_count = 0

    class Wire:
        def __init__(self):
            self.messages = []

        def settimeout(self, _timeout):
            pass

        def close(self):
            pass

        def send(self, text):
            request = json.loads(text)
            assert request["params"]["sessionKey"] == expected_key
            if request["method"] == "chat.history":
                self.messages.append(
                    {"type": "res", "id": request["id"], "ok": True, "payload": {"messages": list(kernel_history)}}
                )
            else:
                assert request["method"] == "chat.send"
                sent_messages.append(request["params"]["message"])
                run_id = request["params"]["idempotencyKey"]
                kernel_history.extend(
                    [{"role": "user", "content": sent_messages[-1]}, {"role": "assistant", "content": "answer"}]
                )
                self.messages.extend(
                    [
                        {
                            "event": "agent",
                            "payload": {"runId": run_id, "stream": "assistant", "data": {"delta": "answer"}},
                        },
                        {
                            "event": "agent",
                            "payload": {"runId": run_id, "stream": "lifecycle", "data": {"phase": "end"}},
                        },
                    ]
                )

        def recv(self):
            return json.dumps(self.messages.pop(0))

    def connect(session):
        nonlocal connect_count
        connect_count += 1
        if connect_count == 1:
            raise OpenClawWSError("connection refused")
        session._ws = Wire()

    monkeypatch.setattr(OpenClawWSSession, "connect", connect)
    first = build_from_saved_records(records, "turn-1", model)
    list(first._run_stream_with_mcp_lease("", SimpleNamespace(input="question", turn_id="turn-1")))
    assert records[0]["role"] == "user"
    assert records[1]["status"] == "error"
    assert records[1]["property"]["builtin_property"]["error_code"] == "craw_not_submitted"
    assert kernel_history == []

    retry = build_from_saved_records(records, "turn-2", model)
    list(retry._run_stream_with_mcp_lease("", SimpleNamespace(input="question", turn_id="turn-2")))
    assert sent_messages == ["question"]
    assert records[-1]["content"] == "answer"
    third = build_from_saved_records(records, "turn-3", model)
    list(third._run_stream_with_mcp_lease("", SimpleNamespace(input="next", turn_id="turn-3")))
    assert sent_messages == ["question", "next"]
    # 原失败提示仍然可展示，未篡改/删除平台账本。
    assert records[1]["status"] == "error"


def test_cancel_display_record_is_not_kernel_history():
    records = []
    writer = RecordingWriter(records, "turn-1")
    writer(
        CustomEvent(
            type=EventType.CUSTOM,
            name=SessionPersistenceEventNames.UserInputSaved.value,
            value={"content": "submitted question", "turn_id": "turn-1"},
        )
    )
    writer(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id="history-recovery", run_id=RunId.CANCELLED))
    assert any(record["status"] == "error" for record in records)
    agent = build_from_saved_records(records, "turn-2")
    assert build_openclaw_history(agent.session_context_data) == [{"role": "user", "content": "submitted question"}]


@pytest.mark.parametrize("status_after_storage", ["complete", "error"])
def test_cancel_preserves_received_partial_reply_in_kernel_history(status_after_storage):
    records = []
    writer = RecordingWriter(records, "turn-1")
    writer(
        CustomEvent(
            type=EventType.CUSTOM,
            name=SessionPersistenceEventNames.UserInputSaved.value,
            value={"content": "submitted question", "turn_id": "turn-1"},
        )
    )
    writer(TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id="kernel-message", role="assistant"))
    writer(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id="kernel-message", delta="partial"))
    writer(RunErrorEvent(type=EventType.RUN_ERROR, message=RunId.CANCELLED_MESSAGE))
    assert records[1]["content"] == "partial"
    assert records[1]["status"] == "complete"
    assert records[2]["property"]["builtin_property"]["local_notice"] is True
    # 即使存储层将部分输出标记 error，也不能把其实际内容当成本地提示丢弃。
    records[1]["status"] = status_after_storage
    expected = [{"role": "user", "content": "submitted question"}, {"role": "assistant", "content": "partial"}]
    agent = build_from_saved_records(records, "turn-2")
    history = build_openclaw_history(agent.session_context_data)
    assert history == expected

    class HistoryWire:
        def send(self, payload):
            self.request = json.loads(payload)

        def recv(self):
            return json.dumps({"type": "res", "id": self.request["id"], "ok": True, "payload": {"messages": expected}})

        def settimeout(self, _timeout):
            pass

    session = OpenClawWSSession("ws://localhost", "test-token")
    session._ws = HistoryWire()
    session.verify_history("history-recovery", history)
    session.send_chat("history-recovery", "continue")
    assert session._ws.request["method"] == "chat.send"
