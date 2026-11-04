import json
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from aidev_agent.packages.craw.mcp_egress import rewrite_openclaw_config_file

TEMPLATE_DEPLOY = Path(__file__).parents[4] / "template" / "builtin" / "{{cookiecutter.project_name}}" / "deploy"
SCRIPT = TEMPLATE_DEPLOY / "apply-agent-config.py"
SUPERVISOR = TEMPLATE_DEPLOY / "craw-supervisor.sh"


@pytest.fixture
def module(monkeypatch):
    django = ModuleType("django")
    django_conf = ModuleType("django.conf")
    django_conf.settings = SimpleNamespace(configured=True)
    monkeypatch.setitem(sys.modules, "django", django)
    monkeypatch.setitem(sys.modules, "django.conf", django_conf)

    spec = spec_from_file_location("apply_agent_config", SCRIPT)
    loaded = module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_supervisor_uses_one_config_path_and_fails_closed_on_rewrite():
    script = SUPERVISOR.read_text(encoding="utf-8")

    assert 'OPENCLAW_STATE_DIR="${OPENCLAW_STATE_DIR:-${HOME}/.openclaw}"' in script
    assert 'OPENCLAW_CONFIG_PATH="${OPENCLAW_CONFIG_PATH:-${OPENCLAW_STATE_DIR}/openclaw.json}"' in script
    assert '--port "${BKAI_MCP_EGRESS_PORT}" --config "" &' in script
    assert 'if [ ! -f "${OPENCLAW_CONFIG_PATH}" ]; then' in script
    assert "FATAL: MCP egress rewrite failed" in script
    assert "WARN: MCP egress rewrite failed" not in script


def test_main_rejects_missing_injected_runtime_settings(module, monkeypatch):
    monkeypatch.setenv("BKAI_AGENT", "demo-agent")
    monkeypatch.delenv("AIDEV_GATEWAY_NAME", raising=False)
    monkeypatch.delenv("BK_APIGW_STAGE", raising=False)

    assert module.main() == 4


def test_optional_skill_failure_does_not_abort(module, monkeypatch, tmp_path):
    def fail(*args, **kwargs):
        raise ValueError("bad archive")

    monkeypatch.setattr(module, "_install_skill", fail)
    status_path = tmp_path / "status.json"

    assert module.materialize_skills(object(), [{"id": "1", "skill_code": "demo"}], tmp_path, status_path) == []
    status = json.loads(status_path.read_text())
    assert status["skills_root"] == str(tmp_path)
    assert status["related_count"] == 1
    assert status["installed"] == []
    assert status["failures"] == [{"skill": "demo", "error": "ValueError"}]


def test_openclaw_2026_8_migration_preserves_supported_ui_settings(module):
    config = {"ui": {"assistant": {"name": "legacy"}, "prefs": {"theme": "claw"}}}

    module._migrate_openclaw_2026_8(config)

    assert config == {"ui": {"prefs": {"theme": "claw"}}}


def test_openclaw_2026_8_migration_removes_empty_ui(module):
    config = {"ui": {"assistant": {"name": "legacy"}}}

    module._migrate_openclaw_2026_8(config)

    assert "ui" not in config


@pytest.mark.parametrize(
    "current_servers", [{}, {"retained": {"url": "https://mcp.example.test/new", "transport": "streamable_http"}}]
)
def test_main_replaces_removed_platform_mcp_and_routes(module, monkeypatch, tmp_path, current_servers):
    config_path, routes_path = tmp_path / "openclaw.json", tmp_path / "routes.json"
    config_path.write_text(
        json.dumps({"mcp": {"servers": {"removed": {"url": "http://127.0.0.1:18787/egress/shared/removed/"}}}})
    )
    routes_path.write_text(json.dumps({"removed": "https://mcp.example.test/removed"}))
    for name, value in {
        "BKAI_AGENT": "demo",
        "AIDEV_GATEWAY_NAME": "aidev",
        "BK_APIGW_STAGE": "prod",
        "OPENCLAW_STATE_DIR": str(tmp_path),
        "OPENCLAW_CONFIG_PATH": str(config_path),
        "BKAI_MCP_EGRESS_ROUTES": str(routes_path),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("BKAI_CRAW_SKILLS_ONLY", raising=False)
    config = SimpleNamespace(chat_model="model", mcp_server_config=current_servers)
    monkeypatch.setattr(module, "AgentResourceManager", lambda: SimpleNamespace(get_agent_config=lambda code: config))
    monkeypatch.setattr(module, "render_soul", lambda config: "soul")
    assert module.main() == 0
    applied = json.loads(config_path.read_text())["mcp"]["servers"]
    assert set(applied) == set(current_servers)
    if current_servers:
        assert applied["retained"]["transport"] == "streamable-http"
    rewrite_openclaw_config_file(str(config_path), "http://127.0.0.1:18787", egress_key="test")
    routes = json.loads(routes_path.read_text())
    assert "removed" not in routes
    assert list(routes.values()) == [item["url"] for item in current_servers.values()]
