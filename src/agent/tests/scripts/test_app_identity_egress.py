import io
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
