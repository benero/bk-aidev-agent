import io
import json
import shutil
import subprocess
import sys
import zipfile
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

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
@pytest.mark.parametrize("container", ["mcp", "mcpServers", "both"])
def test_main_replaces_removed_platform_mcp_and_routes(module, monkeypatch, tmp_path, current_servers, container):
    config_path, routes_path = tmp_path / "openclaw.json", tmp_path / "routes.json"
    old_servers = {"removed": {"url": "http://127.0.0.1:18787/egress/shared/removed/"}}
    old_config = {}
    if container in {"mcp", "both"}:
        old_config["mcp"] = {"servers": old_servers}
    if container in {"mcpServers", "both"}:
        old_config["mcpServers"] = old_servers
    config_path.write_text(json.dumps(old_config))
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
    applied_config = json.loads(config_path.read_text())
    assert "mcpServers" not in applied_config
    applied = applied_config["mcp"]["servers"]
    assert set(applied) == set(current_servers)
    if current_servers:
        assert applied["retained"]["transport"] == "streamable-http"
    rewrite_openclaw_config_file(str(config_path), "http://127.0.0.1:18787", egress_key="test")
    routes = json.loads(routes_path.read_text())
    assert "removed" not in routes
    assert list(routes.values()) == [item["url"] for item in current_servers.values()]


@pytest.mark.parametrize("config_version", [None, "", "1.0.1"])
@pytest.mark.parametrize("skills_only", [False, True])
def test_startup_reads_deployment_snapshot_for_config_and_skills(
    module, monkeypatch, tmp_path, config_version, skills_only
):
    if config_version is None:
        monkeypatch.delenv("BKAI_CRAW_AGENT_CONFIG_VERSION", raising=False)
    else:
        monkeypatch.setenv("BKAI_CRAW_AGENT_CONFIG_VERSION", config_version)
    for name, value in {
        "BKAI_AGENT": "demo",
        "AIDEV_GATEWAY_NAME": "aidev",
        "BK_APIGW_STAGE": "prod",
        "OPENCLAW_STATE_DIR": str(tmp_path),
        "OPENCLAW_CONFIG_PATH": str(tmp_path / "openclaw.json"),
        "OPENCLAW_WORKSPACE_DIR": str(tmp_path / "workspace"),
        "BKAI_CRAW_SKILLS_ONLY": "1" if skills_only else "0",
    }.items():
        monkeypatch.setenv(name, value)
    snapshot = SimpleNamespace(
        chat_model="snapshot-model",
        role_prompt="snapshot prompt",
        mcp_server_config={"snapshot-mcp": {"url": "https://snapshot.example.com/mcp"}},
        related_skills=[{"id": "skill", "version": "0.3.0"}],
    )
    manager = Mock()
    manager.get_agent_config.return_value = snapshot
    monkeypatch.setattr(module, "AgentResourceManager", lambda: manager)
    monkeypatch.setattr(module, "render_soul", lambda config: config.role_prompt)
    skills = Mock(return_value=["skill"])
    monkeypatch.setattr(module, "materialize_skills", skills)

    assert module.main() == 0

    if config_version:
        manager.get_agent_config.assert_called_once_with("demo", version="1.0.1")
    else:
        manager.get_agent_config.assert_called_once_with("demo")
    if skills_only:
        assert skills.call_args.args[1] == snapshot.related_skills
    else:
        assert (tmp_path / "workspace" / "SOUL.md").read_text() == "snapshot prompt"
        applied = json.loads((tmp_path / "openclaw.json").read_text())
        assert applied["mcp"]["servers"] == snapshot.mcp_server_config
        assert applied["agents"]["defaults"]["model"] == "bkaidev/snapshot-model"


@pytest.mark.parametrize("config_version", [None, "1.0.1"])
@pytest.mark.parametrize("callee_agent_code", [None, "demo", "other-agent"])
def test_skill_download_uses_deployment_version_for_relationship_scope(
    module, monkeypatch, tmp_path, config_version, callee_agent_code
):
    monkeypatch.setenv("BKAI_AGENT", "demo")
    if config_version:
        monkeypatch.setenv("BKAI_CRAW_AGENT_CONFIG_VERSION", config_version)
    else:
        monkeypatch.delenv("BKAI_CRAW_AGENT_CONFIG_VERSION", raising=False)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("SKILL.md", "snapshot skill")
    stream = MagicMock()
    stream.__enter__.return_value.iter_bytes.return_value = [payload.getvalue()]
    monkeypatch.setattr(module.httpx, "stream", Mock(return_value=stream))
    manager = Mock()
    manager.retrieve_skill_download.return_value = {"url": "https://skills.example.com/snapshot.zip"}

    assert (
        module._install_skill(
            manager,
            {"id": "skill", "skill_code": "demo", "version": "0.3.0", "callee_agent_code": callee_agent_code},
            tmp_path,
        )
        == "demo"
    )

    expected = {"version": "0.3.0", "callee_agent_code": callee_agent_code}
    if config_version and callee_agent_code in (None, "demo"):
        expected["params"] = {"agent_version": "1.0.1"}
    manager.retrieve_skill_download.assert_called_once_with("skill", **expected)
    assert (tmp_path / "demo" / "SKILL.md").read_text() == "snapshot skill"


@pytest.mark.parametrize("runtime,config_version", [("craw", "1.0.1"), ("craw", None), ("builtin", "1.0.1")])
def test_generated_project_pins_only_craw_snapshot_without_changing_sdk_version(tmp_path, runtime, config_version):
    cookiecutter = shutil.which("cookiecutter")
    if cookiecutter is None:
        pytest.skip("cookiecutter CLI is required for full template generation")
    template_root = TEMPLATE_DEPLOY.parents[1]
    command = [
        cookiecutter,
        str(template_root),
        "--no-input",
        "--default-config",
        "--output-dir",
        str(tmp_path),
        "project_name=version-bound-agent",
        f"agent_runtime={runtime}",
    ]
    if config_version is not None:
        command.append(f"agent_config_version={config_version}")
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
    generated = tmp_path / "version-bound-agent"
    assert (generated / "VERSION").read_text() == (TEMPLATE_DEPLOY.parent / "VERSION").read_text()
    if runtime == "craw":
        assert f'BKAI_CRAW_AGENT_CONFIG_VERSION="{config_version or ""}"' in (generated / "Dockerfile").read_text()
        assert "BKAI_CRAW_AGENT_CONFIG_VERSION" in (generated / "deploy" / "apply-agent-config.py").read_text()
    else:
        assert not (generated / "Dockerfile").exists()
        assert not (generated / "deploy").exists()
