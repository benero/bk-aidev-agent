# -*- coding: utf-8 -*-
"""Craw 工具事件通过真实AGUISessionWriter持久化，不重建到上一轮assistant。"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from ag_ui.core import EventType, RunFinishedEvent, TextMessageContentEvent, TextMessageStartEvent
from ag_ui.encoder import EventEncoder
from aidev_agent.packages.craw import CrawCompletionAgent
from aidev_agent.packages.craw.openclaw_ws import OpenClawEvent
from aidev_agent.services.event_handlers.agui_writer import AGUISessionWriter
from aidev_agent.utils.event import RunId


class SessionStorageApi:
    def __init__(self, previous):
        self.records = deepcopy(previous)
        self.reads = 0
        self.updates = 0

    def create_chat_session_content(self, *, json, headers):
        record = {"id": len(self.records) + 1, **deepcopy(json)}
        self.records.append(record)
        return {"data": {"id": record["id"]}}

    def update_chat_session_content(self, *, path_params, json, headers):
        self.updates += 1
        self.records[path_params["id"] - 1].update(deepcopy(json))
        return {"data": {}}

    def get_chat_session_contents(self, *, params, headers):
        self.reads += 1
        return {"data": deepcopy(self.records)}

    def get_chat_session(self, **kwargs):
        return {"data": {"property": {}}}

    def update_chat_session(self, **kwargs):
        return {"data": {}}


def make_agent(previous):
    api = SessionStorageApi(previous)
    writer = AGUISessionWriter(session_code="tool-session", client=SimpleNamespace(api=api), turn_id="current-turn")
    return CrawCompletionAgent(event_handler=writer), writer, api


def start_tool(agent, opened, started):
    output = list(
        agent._emit_tool(
            EventEncoder(),
            OpenClawEvent("tool", phase="start", tool_call_id="new-call", tool_name="search", args={"query": "hello"}),
            "body-message",
            opened,
            started,
        )
    )
    start = next(json.loads(frame.removeprefix("data: ").strip()) for frame in output if '"TOOL_CALL_START"' in frame)
    return start["parentMessageId"]


@pytest.mark.parametrize("has_previous", [False, True])
@pytest.mark.parametrize("is_error", [False, True])
def test_tool_call_owns_current_parent_and_result_replay_is_idempotent(has_previous, is_error):
    previous = (
        [
            {
                "id": 1,
                "role": "assistant",
                "content": "previous answer",
                "property": {"builtin_property": {"message_id": "old-assistant", "tool_calls": []}},
            }
        ]
        if has_previous
        else []
    )
    agent, writer, api = make_agent(previous)
    writer(TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id="before-body", role="assistant"))
    writer(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id="before-body", delta="before tool"))
    opened, started = set(), {}
    parent_id = start_tool(agent, opened, started)
    parent = api.records[-1]
    assert parent["property"]["turn_id"] == "current-turn"
    assert parent["property"]["builtin_property"]["message_id"] == parent_id
    calls = parent["property"]["builtin_property"]["tool_calls"]
    assert calls == [
        {"id": "new-call", "type": "function", "function": {"name": "search", "arguments": '{"query": "hello"}'}}
    ]
    result = OpenClawEvent(
        "tool", phase="result", tool_call_id="new-call", tool_name="search", result="result", is_error=is_error
    )
    for _ in range(2):
        list(agent._emit_tool(EventEncoder(), result, "other-body", opened, started))
    writer(TextMessageStartEvent(type=EventType.TEXT_MESSAGE_START, message_id="after-body", role="assistant"))
    writer(TextMessageContentEvent(type=EventType.TEXT_MESSAGE_CONTENT, message_id="after-body", delta="after tool"))
    writer(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id="tool-session", run_id="run"))
    if has_previous:
        assert api.records[0] == previous[0]
    assert len([record for record in api.records if record["role"] == "tool"]) == 1
    assert (
        len(
            [
                record
                for record in api.records
                if record["property"].get("builtin_property", {}).get("message_id") == parent_id
            ]
        )
        == 1
    )
    assert {record["content"] for record in api.records if record["role"] == "assistant"} >= {
        "before tool",
        "after tool",
    }
    assert parent["property"]["builtin_property"]["tool_calls"] == calls
    assert api.reads == 0
    assert api.updates == 0


def test_tool_parent_survives_cancel_before_result():
    agent, writer, api = make_agent([])
    parent_id = start_tool(agent, set(), {})
    writer(RunFinishedEvent(type=EventType.RUN_FINISHED, thread_id="tool-session", run_id=RunId.CANCELLED))
    parent = next(record for record in api.records if record["property"]["builtin_property"]["message_id"] == parent_id)
    assert parent["property"]["builtin_property"]["tool_calls"][0]["function"]["name"] == "search"
    assert not any(record["role"] == "tool" for record in api.records)
    assert api.reads == 0
