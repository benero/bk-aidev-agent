# -*- coding: utf-8 -*-
"""官方chat展示窗口与最终run快照契约（仅替换网络和时钟）。"""

import json
from types import SimpleNamespace

import pytest
from aidev_agent.packages.craw import CrawCompletionAgent, OpenClawBackend
from aidev_agent.packages.craw import openclaw_ws as ws_module
from aidev_agent.packages.craw.mcp_egress import McpEgress
from aidev_agent.packages.craw.mcp_identity import McpIdentityLease
from aidev_agent.packages.craw.openclaw_ws import OpenClawHistoryError, OpenClawWSSession


def transcript(count, *, with_tools=False):
    messages = []
    for index in range(count):
        messages.append({"role": "user" if index % 2 == 0 else "assistant", "content": f"message-{index}"})
        if with_tools:
            messages.append({"role": "toolResult", "content": [{"type": "text", "text": "tool result"}]})
    for seq, item in enumerate(messages, 1):
        item["__openclaw"] = {"seq": seq, "id": f"entry-{seq}"}
    return messages


def visible(messages):
    return [
        {"role": item["role"], "content": item["content"]} for item in messages if item["role"] in {"user", "assistant"}
    ]


class HistoryWire:
    def __init__(self, records):
        self.records = records
        self.requests = []

    def send(self, text):
        self.requests.append(json.loads(text))

    def settimeout(self, _timeout):
        pass

    def recv(self):
        request = self.requests[-1]
        limit = request["params"].get("limit", 200)
        messages = self.records[-limit:]
        if request["method"] == "chat.history":
            # 官方每条128KiB预算占位；原始sessions.get没有这一展示变换。
            messages = [
                {
                    "role": item["role"],
                    "content": [{"type": "text", "text": "[chat.history omitted: message too large]"}],
                    "__openclaw": {"truncated": True, "reason": "oversized"},
                }
                if len(json.dumps(item)) > 128 * 1024
                else item
                for item in messages
            ]
            max_chars = request["params"].get("maxChars", 12000)
            messages = [
                {**item, "content": item["content"][:max_chars] + "\n...(truncated)..."}
                if isinstance(item["content"], str) and len(item["content"]) > max_chars
                else item
                for item in messages
            ]
        return json.dumps({"type": "res", "id": request["id"], "ok": True, "payload": {"messages": messages}})


def verify(records, expected):
    session = OpenClawWSSession("ws://localhost", "token")
    session._ws = HistoryWire(records)
    session.verify_history("session", expected)
    return session._ws.requests


@pytest.mark.parametrize("count,with_tools", [(202, False), (1202, False), (602, True)])
def test_bounded_history_accepts_long_healthy_transcript(count, with_tools):
    records = transcript(count, with_tools=with_tools)
    requests = verify(records, visible(records))
    assert requests[0]["params"]["limit"] == 1000
    assert requests[0]["params"]["maxChars"] == 500000
    if len(records) > 1000:
        assert requests[-1]["method"] == "sessions.get"
        assert requests[-1]["params"]["limit"] == 1000


@pytest.mark.parametrize("size", [13000, 150000, 600000])
def test_long_text_uses_raw_fallback_instead_of_comparing_display_placeholder(size):
    records = transcript(2)
    records[1]["content"] = "x" * size
    requests = verify(records, visible(records))
    assert [request["method"] for request in requests] == (
        ["chat.history"] if size == 13000 else ["chat.history", "sessions.get"]
    )


def test_tool_display_cropping_is_not_missing_conversation():
    records = transcript(2, with_tools=True)
    records[1]["content"] = [{"type": "text", "text": "x" * 150000}]
    requests = verify(records, visible(records))
    assert requests[-1]["method"] == "sessions.get"


def test_raw_fallback_recognizes_only_gateway_injected_timestamp():
    records = transcript(2)
    records[0]["content"] = "[Wed 2026-11-04 12:00 UTC] question"
    assert verify(records, [{"role": "user", "content": "question"}, {"role": "assistant", "content": "message-1"}])
    records[0]["content"] = "untrusted prefix question"
    with pytest.raises(OpenClawHistoryError):
        verify(records, [{"role": "user", "content": "question"}, {"role": "assistant", "content": "message-1"}])


@pytest.mark.parametrize("fault", ["missing_prefix", "misaligned_tail", "forged_short_seq", "missing_seq"])
def test_real_missing_or_misaligned_history_is_rejected(fault):
    original = transcript(1202 if fault in {"misaligned_tail", "missing_seq"} else 20)
    expected = visible(original)
    records = original
    if fault == "missing_prefix":
        records = original[2:]
        # 实际transcript丢了前缀，readSessionMessages重新从1编号。
        for seq, item in enumerate(records, 1):
            item["__openclaw"]["seq"] = seq
    elif fault == "forged_short_seq":
        records = original[2:]
    elif fault == "misaligned_tail":
        records[-1]["content"] = "wrong final message"
    else:
        for item in records:
            item.pop("__openclaw")
    with pytest.raises(OpenClawHistoryError):
        verify(records, expected)


class RunWire:
    def __init__(self, session, now, phases, statuses):
        self.session = session
        self.now = now
        self.messages = [
            {
                "event": "agent",
                "payload": {
                    "runId": "active",
                    "stream": "lifecycle",
                    "data": {"phase": phase, "error": "failed" if phase == "error" else None},
                },
            }
            for phase in phases
        ]
        self.statuses = list(statuses)
        self.sent = []
        self.timeout = 1

    def settimeout(self, value):
        self.timeout = value

    def send(self, text):
        request = json.loads(text)
        assert request["method"] == "agent.wait"
        assert request["params"] == {"runId": "active", "timeoutMs": 0}
        self.sent.append(self.now[0])
        status = self.statuses.pop(0) if self.statuses else "timeout"
        self.messages.append(
            {"type": "res", "id": request["id"], "ok": True, "payload": {"runId": "active", "status": status}}
        )

    def recv(self):
        if self.messages:
            self.now[0] += 0.05
            return json.dumps(self.messages.pop(0))
        self.now[0] += self.timeout
        raise TimeoutError("bounded read")


@pytest.mark.parametrize(
    "phases,statuses,kind,terminated",
    [
        (["error"], ["error"], "error", True),
        (["error"], ["ok"], "done", True),
        (["error", "start", "end"], ["timeout", "ok"], "done", True),
        (["error"], ["timeout"], "error", False),
        (["error"], ["unknown"], "error", False),
    ],
)
def test_attempt_terminal_requires_final_chat_snapshot(monkeypatch, phases, statuses, kind, terminated):
    now = [0.0]
    monkeypatch.setattr(ws_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    session = OpenClawWSSession("ws://localhost", "token", timeout=3)
    session._run_id = "active"
    session._ws = RunWire(session, now, phases, statuses)
    events = list(session.events())
    assert [event.kind for event in events] == [kind]
    assert session.terminated is terminated
    assert all(right >= left + 1 for left, right in zip(session._ws.sent, session._ws.sent[1:]))
    assert now[0] <= 3.2


def test_ticks_cannot_extend_absolute_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ws_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    session = OpenClawWSSession("ws://localhost", "token", timeout=3)
    session._run_id = "active"
    sent = []

    def recv():
        now[0] += 0.25
        return (
            {"event": "agent", "payload": {"runId": "active", "stream": "lifecycle", "data": {"phase": "error"}}}
            if now[0] == 0.25
            else {"event": "tick"}
        )

    monkeypatch.setattr(session, "_recv_json", recv)
    monkeypatch.setattr(session, "_send", lambda request: sent.append(now[0]))
    assert [event.kind for event in session.events()] == ["error"]
    assert session.terminated is False
    assert now[0] == 3
    assert len(sent) <= 3
    assert all(right >= left + 1 for left, right in zip(sent, sent[1:]))


@pytest.mark.parametrize("terminal_status,can_reuse", [("error", True), ("unknown", False)])
@pytest.mark.parametrize("terminal_source", ["agent", "chat"])
def test_confirmed_error_releases_lease_for_next_run(monkeypatch, terminal_status, can_reuse, terminal_source):
    class KernelWire:
        def __init__(self):
            self.messages = []

        def settimeout(self, _timeout):
            pass

        def close(self):
            pass

        def send(self, text):
            request = json.loads(text)
            if request["method"] == "chat.history":
                self.messages.append({"type": "res", "id": request["id"], "ok": True, "payload": {"messages": []}})
            elif request["method"] == "chat.send":
                payload = {"runId": request["params"]["idempotencyKey"]}
                if terminal_source == "chat":
                    payload.update(state="error", errorMessage="dispatch failed before agent start")
                else:
                    payload.update(stream="lifecycle", data={"phase": "error", "error": "model failed"})
                self.messages.append({"event": terminal_source, "payload": payload})
            else:
                self.messages.append(
                    {
                        "type": "res",
                        "id": request["id"],
                        "ok": True,
                        "payload": {
                            "runId": request["params"]["runId"],
                            "status": terminal_status,
                            "error": "model failed",
                        },
                    }
                )

        def recv(self):
            return json.dumps(self.messages.pop(0))

    monkeypatch.setattr(OpenClawWSSession, "connect", lambda session: setattr(session, "_ws", KernelWire()))
    slot = McpEgress(drain_seconds=0)
    for username in ("first-user", "next-user"):
        lease = McpIdentityLease(slot.acquire(username, timeout=0.1))
        assert lease.lease_id
        events = []
        agent = CrawCompletionAgent(
            backend=OpenClawBackend(transport="ws"),
            session_context_data=[{"role": "user", "content": "question"}],
            event_handler=events.append,
        )
        list(agent._run_stream(lease))
        assert any(event.type.value == "RUN_ERROR" for event in events)
        assert lease.quarantined is not can_reuse
        if not can_reuse:
            assert slot.quarantine(lease.lease_id)
            assert slot.acquire("next-user", timeout=0.01) == ""
            return
        assert slot.release(lease.lease_id)


@pytest.mark.parametrize(
    "state,status,kind", [("error", "error", "error"), ("final", "ok", "done"), ("aborted", "ok", "cancelled")]
)
def test_chat_terminal_without_agent_lifecycle_still_requires_confirmation(monkeypatch, state, status, kind):
    now = [0.0]
    monkeypatch.setattr(ws_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    session = OpenClawWSSession("ws://localhost", "token", timeout=3)
    session._run_id = "active"
    wire = RunWire(session, now, [], [status])
    wire.messages.append({"event": "chat", "payload": {"runId": "active", "state": state, "errorMessage": "failed"}})
    session._ws = wire
    assert [event.kind for event in session.events()] == [kind]
    assert session.terminated
    assert len(wire.sent) == 1


def test_abort_shortens_absolute_run_deadline(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(ws_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    session = OpenClawWSSession("ws://localhost", "token", timeout=300)
    session._run_id = "active"
    session._run_deadline = 300
    monkeypatch.setattr(session, "_send", lambda request: None)
    assert session.abort("session")
    assert session._run_deadline == 10


@pytest.mark.parametrize("foreign_state", ["error", "final", "aborted"])
def test_foreign_chat_terminal_never_triggers_confirmation(monkeypatch, foreign_state):
    now = [0.0]
    monkeypatch.setattr(ws_module, "time", SimpleNamespace(monotonic=lambda: now[0]))
    session = OpenClawWSSession("ws://localhost", "token", timeout=3)
    session._run_id = "active"
    wire = RunWire(session, now, [], ["ok"])
    wire.messages.extend(
        [
            {"event": "chat", "payload": {"runId": "foreign", "state": foreign_state, "errorMessage": "foreign error"}},
            {"event": "chat", "payload": {"runId": "active", "state": "final"}},
        ]
    )
    session._ws = wire
    assert [(event.kind, event.text) for event in session.events()] == [("done", "")]
    assert wire.sent == [0.1]  # 两个帧均已读取；外来终态未触发查询。
    assert session.terminated
