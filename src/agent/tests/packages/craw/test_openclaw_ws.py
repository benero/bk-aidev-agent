# -*- coding: utf-8 -*-
"""OpenClaw WebSocket 握手与生命周期终态。"""

import json

import pytest
import websocket
from aidev_agent.packages.craw.openclaw_ws import OpenClawWSError, OpenClawWSSession, _is_loopback_url


def test_unauthenticated_fallback_requires_loopback(monkeypatch):
    session = OpenClawWSSession("wss://example.com/", "token")
    handshakes = iter([[], ["operator.write"]])
    monkeypatch.setattr(session, "_handshake", lambda **_kwargs: next(handshakes))

    with pytest.raises(OpenClawWSError, match="loopback"):
        session.connect()


@pytest.mark.parametrize("url", ["ws://127.0.0.1:18789/", "ws://[::1]:18789/", "ws://localhost:18789/"])
def test_loopback_url_detection(url):
    assert _is_loopback_url(url) is True
    assert _is_loopback_url("wss://example.com/") is False


def test_handshake_uses_bounded_timeout(monkeypatch):
    class FakeWS:
        def __init__(self):
            self.messages = [
                json.dumps({"event": "connect.challenge"}),
                json.dumps({"type": "res", "ok": True, "payload": {"auth": {"scopes": ["operator.write"]}}}),
            ]
            self.timeouts = []

        def recv(self):
            return self.messages.pop(0)

        def send(self, _payload):
            return None

        def settimeout(self, value):
            self.timeouts.append(value)

    fake = FakeWS()
    captured = {}

    def create_connection(*_args, **kwargs):
        captured.update(kwargs)
        return fake

    monkeypatch.setattr(websocket, "create_connection", create_connection)
    session = OpenClawWSSession("ws://127.0.0.1:18789/", "token", timeout=300)

    assert session._handshake(with_auth=True) == ["operator.write"]
    assert captured["timeout"] == 20
    assert fake.timeouts[-1] == 300


def test_lifecycle_failure_is_not_reported_as_done():
    session = OpenClawWSSession("ws://127.0.0.1:18789/", "token")

    failed = session._translate({"stream": "lifecycle", "data": {"phase": "end", "stopReason": "error"}})
    done = session._translate({"stream": "lifecycle", "data": {"phase": "end", "stopReason": "stop"}})
    cancelled = session._translate({"stream": "lifecycle", "data": {"phase": "end", "stopReason": "aborted"}})

    assert failed.kind == "error"
    assert done.kind == "done"
    assert cancelled.kind == "cancelled"


def test_events_marks_confirmed_cancel_as_termination(monkeypatch):
    session = OpenClawWSSession("ws://localhost", "token")
    session._run_id = "active"
    messages = [
        {
            "event": "agent",
            "payload": {"runId": "active", "stream": "lifecycle", "data": {"phase": "end", "stopReason": "aborted"}},
        }
    ]

    def send(request):
        assert request["method"] == "agent.wait"
        assert not session.terminated
        messages.append(
            {"type": "res", "id": request["id"], "ok": True, "payload": {"runId": "active", "status": "ok"}}
        )

    monkeypatch.setattr(session, "_send", send)
    monkeypatch.setattr(session, "_recv_json", lambda: messages.pop(0))
    assert [event.kind for event in session.events()] == ["cancelled"]
    assert session.terminated is True


def test_cancel_sends_chat_abort_without_closing_transport():
    class FakeWS:
        def __init__(self):
            self.sent = []
            self.timeouts = []
            self.closed = False

        def send(self, payload):
            self.sent.append(json.loads(payload))

        def settimeout(self, value):
            self.timeouts.append(value)

        def close(self):
            self.closed = True

    session = OpenClawWSSession("ws://127.0.0.1:18789/", "token", timeout=300)
    session._ws = FakeWS()
    session.send_chat("session-1", "hello")

    assert session.cancel("session-1") is True
    assert session.cancel_requested is True
    assert session._ws.sent[-1]["method"] == "chat.abort"
    assert session._ws.sent[-1]["params"]["sessionKey"] == "session-1"
    assert session._ws.timeouts[-1] == 10
    assert session._ws.closed is False


@pytest.mark.parametrize("foreign_run", ["other", None])
def test_events_ignore_foreign_run_and_unrelated_response(monkeypatch, foreign_run):
    session = OpenClawWSSession("ws://localhost", "token")
    session._run_id = "active"
    session._chat_request_id = "request"
    messages = list(
        [
            {"type": "res", "id": "other-request", "ok": False},
            {"event": "agent", "payload": {"runId": foreign_run, "stream": "assistant", "data": {"delta": "secret"}}},
            {"event": "agent", "payload": {"runId": foreign_run, "stream": "lifecycle", "data": {"phase": "end"}}},
            {"event": "agent", "payload": {"runId": "active", "stream": "assistant", "data": {"delta": "answer"}}},
            {"event": "agent", "payload": {"runId": "active", "stream": "lifecycle", "data": {"phase": "end"}}},
        ]
    )

    def send(request):
        messages.append(
            {"type": "res", "id": request["id"], "ok": True, "payload": {"runId": "active", "status": "ok"}}
        )

    monkeypatch.setattr(session, "_send", send)
    monkeypatch.setattr(session, "_recv_json", lambda: messages.pop(0))
    assert [(event.kind, event.text) for event in session.events()] == [("text", "answer"), ("done", "")]
    assert session.terminated


@pytest.mark.parametrize(
    "actual,expected,ok",
    [
        ([], [], True),
        (
            [{"role": "user", "content": [{"type": "text", "text": "hello"}]}],
            [{"role": "user", "content": "hello"}],
            True,
        ),
        ([], [{"role": "user", "content": "lost"}], False),
        ([{"role": "assistant", "content": "foreign"}], [], False),
    ],
)
def test_history_must_match_before_sending(actual, expected, ok):
    class HistoryWS:
        def send(self, text):
            self.request = json.loads(text)

        def settimeout(self, value):
            pass

        def recv(self):
            return json.dumps({"type": "res", "id": self.request["id"], "ok": True, "payload": {"messages": actual}})

    session = OpenClawWSSession("ws://localhost", "token")
    session._ws = HistoryWS()
    if ok:
        session.verify_history("session-1", expected)
    else:
        with pytest.raises(OpenClawWSError, match="不支持安全恢复"):
            session.verify_history("session-1", expected)
    assert session._ws.request["method"] == ("chat.history" if ok else "sessions.get")


@pytest.mark.parametrize(
    "model,expected",
    [
        ("openclaw", "session-1"),
        ("openclaw/default", "session-1"),
        ("openclaw:ops", "agent:ops:session-1"),
        ("openclaw/OPS", "agent:ops:session-1"),
        ("agent:ops", "agent:ops:session-1"),
    ],
)
def test_model_routes_history_send_abort_to_same_agent(model, expected):
    class Wire:
        def __init__(self):
            self.sent = []

        def send(self, payload):
            self.sent.append(json.loads(payload))

        def recv(self):
            return json.dumps({"type": "res", "id": self.sent[-1]["id"], "ok": True, "payload": {"messages": []}})

        def settimeout(self, _timeout):
            pass

    session = OpenClawWSSession("ws://localhost", "token", model=model)
    session._ws = Wire()
    session.verify_history("session-1", [])
    session.send_chat("session-1", "hello")
    assert session.abort("session-1")
    requests = session._ws.sent
    assert [request["method"] for request in requests] == ["chat.history", "chat.send", "chat.abort"]
    assert all(request["params"]["sessionKey"] == expected for request in requests)
    assert all("agentId" not in request["params"] for request in requests)


@pytest.mark.parametrize(
    "model,key",
    [
        ("openclaw", "agent:ops:session-1"),
        ("openclaw:ops", "agent:other:session-1"),
        ("openclaw:ops", "agent:ops:"),
        ("invalid-model", "session-1"),
    ],
)
def test_ambiguous_agent_selection_is_rejected(model, key):
    session = OpenClawWSSession("ws://localhost", "token", model=model)
    with pytest.raises(OpenClawWSError):
        session.session_key(key)


def test_existing_matching_agent_key_is_not_nested():
    session = OpenClawWSSession("ws://localhost", "token", model="agent:ops")
    assert session.session_key("agent:ops:session-1") == "agent:ops:session-1"
