import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from contextlib import suppress
from pathlib import Path

import pytest

SUPERVISOR = Path(__file__).parents[4] / "template/builtin/{{cookiecutter.project_name}}/deploy/craw-supervisor.sh"
WORKER = """
import os
import socket
import sys
import time
from pathlib import Path

root = Path(os.environ["TEST_ROOT"])
role = Path(sys.argv[0]).name
if role == "python":
    if sys.argv[1] == "-c":
        os.execv(sys.executable, [sys.executable, *sys.argv[1:]])
    if "--rewrite-only" in sys.argv:
        (root / "rewritten").touch()
        raise SystemExit(0)
    if sys.argv[1].endswith("apply-agent-config.py"):
        if os.environ.get("BKAI_CRAW_SKILLS_ONLY") == "1" and os.environ.get("HOLD_SKILLS") == "1":
            (root / "skill.pid").write_text(str(os.getpid()))
            (root / "skill.ready").touch()
            time.sleep(60)
            raise SystemExit(0)
        Path(os.environ["OPENCLAW_CONFIG_PATH"]).write_text("{}")
        raise SystemExit(0)
    if sys.argv[1] == "bin/manage.py":
        raise SystemExit(0)
    role = "identity" if sys.argv[1].endswith("app-identity-egress.py") else "mcp"
if role == "curl":
    raise SystemExit(0 if (root / "kernel.ready").exists() else 1)
(root / (role + ".pid")).write_text(str(os.getpid()))
if os.environ.get("FAIL_ROLE") == role and os.environ["FAIL_STAGE"] == "startup":
    raise SystemExit(2)
if role in {"identity", "mcp"} and not (os.environ.get("FAIL_ROLE") == role and os.environ["FAIL_STAGE"] == "unready"):
    port_name = "BKAI_APP_IDENTITY_EGRESS_PORT" if role == "identity" else "BKAI_MCP_EGRESS_PORT"
    listener = socket.socket()
    listener.bind(("127.0.0.1", int(os.environ[port_name])))
    listener.listen()
(root / (role + ".ready")).touch()
while True:
    if os.environ.get("FAIL_ROLE") == role and (root / "gunicorn.ready").exists():
        raise SystemExit(2)
    time.sleep(0.02)
"""


def _free_ports():
    with socket.socket() as first, socket.socket() as second:
        first.bind(("127.0.0.1", 0))
        second.bind(("127.0.0.1", 0))
        return first.getsockname()[1], second.getsockname()[1]


@pytest.fixture
def runtime(tmp_path):
    app = tmp_path / "app"
    (app / ".venv/bin").mkdir(parents=True)
    for executable in [app / ".venv/bin/python", tmp_path / "kernel", tmp_path / "gunicorn", tmp_path / "curl"]:
        executable.write_text(f"#!{sys.executable}\n" + textwrap.dedent(WORKER))
        executable.chmod(0o755)
    script = tmp_path / "supervisor.sh"
    script.write_text(SUPERVISOR.read_text().replace("/app", str(app)))
    identity_port, mcp_port = _free_ports()
    env = {
        **os.environ,
        "TEST_ROOT": str(tmp_path),
        "PATH": f"{tmp_path}:{os.environ['PATH']}",
        "BKAI_CRAW_ENTRYPOINT": str(tmp_path / "kernel"),
        "BKAI_AGENT": "test",
        "AIDEV_GATEWAY_NAME": "test",
        "BK_APIGW_STAGE": "prod",
        "BKAI_AIDEV_APP_UPSTREAM_ORIGIN": "https://example.test",
        "BKAI_MCP_EGRESS_ENABLED": "1",
        "BKAI_APP_IDENTITY_EGRESS_PORT": str(identity_port),
        "BKAI_MCP_EGRESS_PORT": str(mcp_port),
        "OPENCLAW_CONFIG_PATH": str(tmp_path / "openclaw.json"),
        "BKAI_OPENCLAW_READY_ATTEMPTS": "30",
        "BKAI_OPENCLAW_READY_INTERVAL": "0.02",
    }
    return tmp_path, script, env


@pytest.mark.parametrize(
    "role,stage",
    [
        ("identity", "startup"),
        ("mcp", "startup"),
        ("kernel", "startup"),
        ("identity", "unready"),
        ("mcp", "unready"),
        ("identity", "running"),
        ("mcp", "running"),
        ("kernel", "running"),
        ("gunicorn", "running"),
    ],
)
def test_supervisor_fails_closed_and_stops_all_children(runtime, role, stage):
    root, script, env = runtime
    env.update(FAIL_ROLE=role, FAIL_STAGE=stage)
    proc = subprocess.Popen(
        ["bash", str(script), str(root / "gunicorn")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=8)
        assert proc.returncode == 1, output
        assert "FATAL" in output
        if stage != "running":
            assert not (root / "gunicorn.ready").exists(), output
        else:
            assert (root / "rewritten").exists(), output
            assert (root / "gunicorn.ready").exists(), output
            assert "application identity egress listener is ready" in output
            assert "MCP egress listener is ready" in output
        for pid_file in root.glob("*.pid"):
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)


@pytest.mark.parametrize("shutdown", ["term", "required_exit"])
def test_supervisor_stops_slow_optional_skill_sync(runtime, shutdown):
    root, script, env = runtime
    env["HOLD_SKILLS"] = "1"
    proc = subprocess.Popen(
        ["bash", str(script), str(root / "gunicorn")],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 5
        while not ((root / "gunicorn.ready").exists() and (root / "skill.ready").exists()):
            assert proc.poll() is None
            assert time.monotonic() < deadline
            time.sleep(0.02)
        if shutdown == "term":
            proc.terminate()
        else:
            os.kill(int((root / "mcp.pid").read_text()), signal.SIGTERM)
        output, _ = proc.communicate(timeout=3)
        assert proc.returncode == (143 if shutdown == "term" else 1), output
        for pid_file in root.glob("*.pid"):
            with pytest.raises(ProcessLookupError):
                os.kill(int(pid_file.read_text()), 0)
    finally:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=3)
