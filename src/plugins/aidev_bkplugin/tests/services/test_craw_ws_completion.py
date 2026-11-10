# -*- coding: utf-8 -*-
"""真实 factory/writer/executor 混合流式与同步 WS 会话契约。"""

import json
from copy import deepcopy
from types import SimpleNamespace

import pytest
from aidev_agent.enums import AgentType
from aidev_agent.packages.craw import CrawCompletionAgent, OpenClawBackend
from aidev_agent.packages.craw.openclaw_ws import OpenClawWSError, OpenClawWSSession
from aidev_agent.pydantic_models import ExecuteKwargs
from aidev_agent.services.agent.factory import _FACTORY_TOKEN, AgentInstanceFactory
from aidev_agent.services.agent.registry import AgentBuildContext
from aidev_agent.services.event_handlers.agui_writer import AGUISessionWriter
from aidev_agent.services.messages_handler import GeneratorStreamingHelper, message_handler_factory
from aidev_bkplugin.services.agent_execution import AgentExecutor


class Storage:
    def __init__(self):
        self.records = []
        self.fail = False
        self.legacy_saves = 0

    def create_chat_session_content(self, *, json, headers):
        if self.fail:
            raise RuntimeError("storage unavailable")
        self.records.append({"id": len(self.records) + 1, **deepcopy(json)})
        return {"data": {"id": len(self.records)}}

    def get_chat_session(self, **kwargs):
        return {"data": {"property": {}}}

    retrieve_chat_session = get_chat_session

    def update_chat_session(self, **kwargs):
        return {"data": {}}

    def get_chat_session_contents(self, *args, **kwargs):
        return deepcopy(self.records) if args else {"data": deepcopy(self.records)}

    def resolve_access_token(self, username):
        return "test-token"

    def save_ai_response(self, session_code, result, *, turn_id):
        self.legacy_saves += 1
        self.records.append({"role": "assistant", "content": result["choices"][0]["delta"]["content"]})


def build_agent(storage, turn):
    factory = AgentInstanceFactory(
        agent_code="agent",
        session_code="mixed-session",
        username="tester",
        resource_manager=storage,
        _token=_FACTORY_TOKEN,
    )
    factory.get_agent_config = lambda _code: SimpleNamespace(command_agent_mapping={})
    base_args = factory._build_from_session()
    writer = AGUISessionWriter(session_code="mixed-session", client=SimpleNamespace(api=storage), turn_id=turn)
    return CrawCompletionAgent(backend=OpenClawBackend(transport="ws")).build(
        AgentBuildContext(
            agent_code="agent",
            agent_type=AgentType.CHAT,
            session_code="mixed-session",
            username="tester",
            resource_manager=storage,
            session_context_data=base_args["session_context_data"],
            event_handler=writer,
        )
    )


class Kernel:
    def __init__(self):
        self.history = []
        self.inputs = []
        self.requests = []
        self.on_chat = None

    def connect(self, session):
        session._ws = Wire(self)


class Wire:
    def __init__(self, kernel):
        self.kernel = kernel
        self.messages = []

    def settimeout(self, value):
        pass

    def close(self):
        pass

    def send(self, text):
        request = json.loads(text)
        self.kernel.requests.append(request)
        method, params = request["method"], request["params"]
        if method == "chat.history":
            self.messages.append(
                {"type": "res", "id": request["id"], "ok": True, "payload": {"messages": deepcopy(self.kernel.history)}}
            )
            return
        if method == "agent.wait":
            self.messages.append(
                {"type": "res", "id": request["id"], "ok": True, "payload": {"runId": params["runId"], "status": "ok"}}
            )
            return
        if method == "chat.abort":
            self.messages.append({"event": "chat", "payload": {"runId": params["runId"], "state": "aborted"}})
            return
        assert method == "chat.send"
        content, run_id = params["message"], params["idempotencyKey"]
        self.kernel.inputs.append(content)
        self.kernel.history.append({"role": "user", "content": content})
        if self.kernel.on_chat is not None:
            self.kernel.on_chat()

        def agent_event(stream, **data):
            self.messages.append({"event": "agent", "payload": {"runId": run_id, "stream": stream, "data": data}})

        if content == "tools":
            agent_event("assistant", delta="before")
            agent_event("tool", phase="start", toolCallId="call-1", name="search", args={"query": "hello"})
            agent_event("tool", phase="result", toolCallId="call-1", name="search", result="result")
            agent_event("assistant", delta="after")
            self.kernel.history.extend(
                [
                    {"role": "assistant", "content": "before"},
                    {"role": "toolResult", "content": "result"},
                    {"role": "assistant", "content": "after"},
                ]
            )
            agent_event("lifecycle", phase="end")
        elif content == "fail":
            # 官方模型错误 payload 仍可让 dispatch 正常完成并给 agent.wait ok。
            agent_event("lifecycle", phase="error", error="model failed")
        else:
            answer = "command response" if content == "/help" else "answer"
            self.kernel.history.append({"role": "assistant", "content": answer})
            self.messages.append(
                {
                    "event": "chat",
                    "payload": {
                        "runId": run_id,
                        "state": "final",
                        "message": {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                    },
                }
            )

    def recv(self):
        return json.dumps(self.messages.pop(0))


@pytest.fixture
def kernel(monkeypatch):
    instance = Kernel()
    monkeypatch.setattr(OpenClawWSSession, "connect", lambda session: instance.connect(session))
    # 仅替换消息队列；仍经过 execute、真实 WS 帧解析与真实 writer。
    monkeypatch.setattr(GeneratorStreamingHelper, "stream", lambda self, source, **kwargs: source)
    message_handler_factory.get().clear_cancel_signal("mixed-session")
    return instance


@pytest.mark.parametrize("streams", [(False, True, False), (True, False, True)])
def test_mixed_turns_keep_ws_history_and_persist_only_once(kernel, streams):
    storage = Storage()
    for number, (content, stream) in enumerate(zip(["tools", "/help", "next"], streams), 1):
        turn = f"turn-{number}"
        agent = build_agent(storage, turn)
        kwargs = ExecuteKwargs(input=content, turn_id=turn, stream=stream)
        output = AgentExecutor(storage).execute_with_save(agent, kwargs, "mixed-session", turn_id=turn)
        if stream:
            frames = list(output)
            assert not any('"RUN_ERROR"' in frame for frame in frames)
        else:
            assert output["choices"][0]["delta"]["content"] == (
                "beforeafter" if content == "tools" else "command response" if content == "/help" else "answer"
            )
            assert agent.sync_response_persisted is True
    assert kernel.inputs == ["tools", "/help", "next"]
    assert storage.legacy_saves == 0
    assert [row["content"] for row in storage.records if row["role"] == "assistant" and row["content"]] == [
        "before",
        "after",
        "command response",
        "answer",
    ]
    calls = [row for row in storage.records if (row.get("property", {}).get("builtin_property", {}).get("tool_calls"))]
    assert len(calls) == 1
    assert calls[0]["property"]["builtin_property"]["tool_calls"][0]["function"] == {
        "name": "search",
        "arguments": '{"query": "hello"}',
    }
    assert len([row for row in storage.records if row["role"] == "tool"]) == 1


@pytest.mark.parametrize("failure", ["model", "storage"])
def test_sync_error_does_not_claim_persistence_or_fall_back_to_duplicate_save(kernel, failure):
    storage = Storage()
    agent = build_agent(storage, "turn")
    agent.sync_response_persisted = True  # execute 必须清除上一轮状态。
    storage.fail = failure == "storage"
    kwargs = ExecuteKwargs(input="fail" if failure == "model" else "next", turn_id="turn", stream=False)
    with pytest.raises(OpenClawWSError if failure == "model" else RuntimeError):
        AgentExecutor(storage).execute_with_save(agent, kwargs, "mixed-session", turn_id="turn")
    assert agent.sync_response_persisted is False
    assert storage.legacy_saves == 0
    if failure == "model":
        assert any(row.get("status") == "error" for row in storage.records)
    else:
        assert agent.event_handler.write_error_count > 0


def test_http_nonstream_retains_host_save_path():
    storage = Storage()
    backend = SimpleNamespace(
        name="openclaw",
        transport="http",
        model="openclaw",
        chat_completions=lambda *args, **kwargs: {},
        message_text=lambda body: "HTTP answer",
    )
    agent = CrawCompletionAgent(backend=backend)
    agent.sync_response_persisted = True
    result = AgentExecutor(storage).execute_with_save(agent, ExecuteKwargs(stream=False), "session", turn_id="turn")
    assert result["choices"][0]["delta"]["content"] == "HTTP answer"
    assert storage.legacy_saves == 1
    assert agent.sync_response_persisted is False


@pytest.mark.parametrize("cancel_current", [False, True])
def test_sync_ws_resets_old_cancel_but_honors_current_cancel(kernel, cancel_current):
    storage = Storage()
    agent = build_agent(storage, "turn")
    assert GeneratorStreamingHelper.cancel("mixed-session")
    assert GeneratorStreamingHelper.is_cancelled("mixed-session")
    if cancel_current:
        kernel.on_chat = lambda: GeneratorStreamingHelper.cancel("mixed-session")
    try:
        result = AgentExecutor(storage).execute_with_save(
            agent, ExecuteKwargs(input="next", stream=False), "mixed-session", turn_id="turn"
        )
        assert result["choices"][0]["delta"]["content"] == ("" if cancel_current else "answer")
        assert any(request["method"] == "chat.abort" for request in kernel.requests) is cancel_current
        assert any(row.get("status") == "error" for row in storage.records) is cancel_current
    finally:
        message_handler_factory.get().clear_cancel_signal("mixed-session")
