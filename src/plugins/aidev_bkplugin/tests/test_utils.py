"""用户token绑定与可选依赖缺失的模块导入兼容。"""

import subprocess
import sys
from types import SimpleNamespace

import pytest
from aidev_agent.packages.craw.mcp_identity import bind_user_access_token, get_bound_user_access_token
from aidev_bkplugin import utils


@pytest.fixture(autouse=True)
def clear_bound_token():
    bind_user_access_token("")
    yield
    bind_user_access_token("")


def test_issued_token_is_normalized_and_bound(monkeypatch):
    request = object()

    def issue(received):
        assert received is request
        return SimpleNamespace(access_token='Bearer "user-token"')

    monkeypatch.setattr(utils, "bkoauth", SimpleNamespace(get_access_token=issue))
    assert utils.set_user_access_token(request) == "user-token"
    assert get_bound_user_access_token() == "user-token"


def test_failed_exchange_clears_previous_user_binding(monkeypatch):
    bind_user_access_token("previous-user")

    def fail(_request):
        raise RuntimeError("unavailable")

    monkeypatch.setattr(utils, "bkoauth", SimpleNamespace(get_access_token=fail))
    assert utils.set_user_access_token(object()) == ""
    assert get_bound_user_access_token() == ""


def test_empty_request_or_missing_oauth_clears_binding(monkeypatch):
    bind_user_access_token("previous-user")
    assert utils.set_user_access_token(None) == ""
    assert get_bound_user_access_token() == ""
    bind_user_access_token("previous-user")
    monkeypatch.setattr(utils, "bkoauth", None)
    assert utils.set_user_access_token(object()) == ""
    assert get_bound_user_access_token() == ""


@pytest.mark.parametrize("sdk_available", [False, True])
def test_module_import_without_optional_dependencies(sdk_available):
    script = """
import sys
sys.modules['bkoauth'] = None
"""
    if not sdk_available:
        script += "sys.modules['aidev_agent'] = None\n"
    script += """
from aidev_bkplugin.utils import set_user_access_token
assert set_user_access_token(None) == ''
assert set_user_access_token(object()) == ''
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
