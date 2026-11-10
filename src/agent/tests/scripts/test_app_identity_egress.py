import io
import json
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

SCRIPT = Path(__file__).parents[4] / "template/builtin/{{cookiecutter.project_name}}/deploy/app-identity-egress.py"
PREFIX = "/openapi/aidev/gateway/llm/v1"


@pytest.fixture
def proxy(monkeypatch):
    spec = spec_from_file_location("app_identity_egress", SCRIPT)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setenv("BKPAAS_APP_ID", "test-app")
    monkeypatch.setenv("BKPAAS_APP_SECRET", "test-secret")
    monkeypatch.setenv("BKAI_AIDEV_APP_UPSTREAM_ORIGIN", "https://api.example.test/api/aidev/prod")
    monkeypatch.setenv("BKAI_AIDEV_APP_UPSTREAM_PATH_PREFIX", PREFIX)
    handler = module.Handler.__new__(module.Handler)
    handler.command = "GET"
    handler.headers = {"Authorization": "Bearer user", "X-Bkapi-Authorization": "untrusted"}
    handler.rfile = io.BytesIO()
    handler.wfile = io.BytesIO()
    statuses, requests = [], []
    handler.send_error = lambda code, message: statuses.append(code)
    handler.send_response = statuses.append
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    real_client = httpx.Client

    def upstream(request):
        requests.append(request)
        return httpx.Response(200, stream=httpx.ByteStream(b"ok"))

    monkeypatch.setattr(module.httpx, "Client", lambda **kw: real_client(transport=httpx.MockTransport(upstream), **kw))
    return handler, statuses, requests


@pytest.mark.parametrize(
    "suffix",
    [
        "/../../../../admin/delete",
        "/%2e%2e/admin",
        "/%252e%252e/admin",
        "/.%2e/admin",
        "/%2e%2e%2fadmin",
        "/%2e%2e%5cadmin",
        "/foo/../admin",
        "/foo\\..\\admin",
    ],
)
def test_proxy_rejects_traversal_before_attaching_identity(proxy, suffix):
    handler, statuses, requests = proxy
    handler.path = PREFIX + suffix

    handler._proxy()

    assert statuses == [403]
    assert requests == []


@pytest.mark.parametrize("path", ["https://evil.test" + PREFIX, "//evil.test" + PREFIX, "/admin", PREFIX + "-other"])
def test_proxy_rejects_outside_allowlist(proxy, path):
    handler, statuses, requests = proxy
    handler.path = path
    handler._proxy()
    assert statuses == [403]
    assert requests == []


def test_proxy_sends_identity_only_to_final_allowed_url(proxy):
    handler, statuses, requests = proxy
    handler.path = PREFIX + "/models?cursor=a%2Fb"
    handler._proxy()
    assert statuses == [200]
    assert len(requests) == 1
    assert str(requests[0].url) == "https://api.example.test/api/aidev/prod" + handler.path
    assert requests[0].headers["X-Bkapi-Authorization"] == '{"bk_app_code":"test-app","bk_app_secret":"test-secret"}'
    assert "Authorization" not in requests[0].headers
    assert handler.wfile.getvalue() == b"ok"


@pytest.mark.parametrize("tenant", [None, "", "tenant-a"])
@pytest.mark.parametrize("incoming", [None, "untrusted-tenant"])
def test_proxy_uses_trusted_application_tenant(proxy, monkeypatch, tenant, incoming):
    handler, statuses, requests = proxy
    if tenant is None:
        monkeypatch.delenv("BKPAAS_APP_TENANT_ID", raising=False)
    else:
        monkeypatch.setenv("BKPAAS_APP_TENANT_ID", tenant)
    if incoming is not None:
        handler.headers["x-bk-tenant-id"] = incoming
    handler.path = PREFIX + "/models"
    handler._proxy()
    assert statuses == [200]
    assert requests[0].headers["X-Bk-Tenant-Id"] == (tenant or "default")


@pytest.mark.parametrize("finish_reason,tool_call", [(None, False), (None, True), ("stop", False)])
def test_chat_done_completes_without_waiting_for_http_eof(proxy, finish_reason, tool_call):
    handler, _, _ = proxy
    delta = {"tool_calls": [{"id": "call-1"}]} if tool_call else {"content": "answer"}
    event = {"id": "test", "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}]}

    def lines():
        yield "data: " + json.dumps(event)
        yield ""
        yield "data: [DONE]"
        raise AssertionError("must not wait for HTTP EOF after protocol completion")

    handler._relay_chat_stream(SimpleNamespace(iter_lines=lines))
    output = handler.wfile.getvalue()
    assert output.endswith(b"data: [DONE]\n\n")
    assert output.count(b"data: [DONE]") == 1
    expected_reason = "tool_calls" if tool_call else "stop"
    assert output.count(f'"finish_reason": "{expected_reason}"'.encode()) == 1
