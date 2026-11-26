"""Run inside one user's sandbox. Read the prompt from stdin, emit AG-UI envelopes."""

import argparse
import asyncio
import json
import os
import secrets
import shutil
import sys
import tempfile
import uuid
from dataclasses import replace
from pathlib import Path

import httpx

from aidev_agent_adapter.adapter import Adapter
from aidev_agent_adapter.cli import load_settings
from aidev_agent_adapter.config_sync import ConfigSynchronizer, private_directory
from aidev_agent_adapter.platform import PlatformClient
from aidev_agent_adapter.runner import serve_runner


def install_user_config(settings, snapshot, user_root):
    # Persist configuration separately from SQLite. Credentials are resolved per sandbox start.
    config = json.loads((snapshot.path / "openclaw.json").read_text())
    for index, server in enumerate(config.get("mcp", {}).get("servers", {}).values()):
        for position, (key, value) in enumerate(server.get("headers", {}).items()):
            variable = f"AIDEV_MCP_{index}_HEADER_{position}"
            settings.engine_env[variable] = value
            server["headers"][key] = "${" + variable + "}"
    target = user_root / "openclaw.json"
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=user_root, delete=False) as tmp:
        stage = Path(tmp.name)
        try:
            json.dump(config, tmp, ensure_ascii=False, indent=2)
            tmp.flush()
            os.fsync(tmp.fileno())
            os.replace(stage, target)
        finally:
            stage.unlink(missing_ok=True)
    return replace(snapshot, engine_config=target)


async def wait_for_gateway(gateway: asyncio.subprocess.Process) -> None:
    # Leave time for Runner registration within Channel's 120-second startup window.
    last_status = "not probed"
    async with httpx.AsyncClient(trust_env=False, timeout=2) as client:
        try:
            async with asyncio.timeout(100):
                while gateway.returncode is None:
                    try:
                        response = await client.get("http://127.0.0.1:18789/readyz")
                        if response.status_code == 200:
                            return
                        last_status = f"HTTP {response.status_code}"
                    except httpx.RequestError as exc:
                        last_status = type(exc).__name__
                    await asyncio.sleep(1)
                raise RuntimeError(f"OpenClaw Gateway exited with status {gateway.returncode}")
        except TimeoutError as exc:
            raise RuntimeError(f"OpenClaw Gateway readiness timed out; last probe: {last_status}") from exc


async def run(config: Path, prompt: str | None) -> None:
    if sys.platform != "linux" or Path.cwd().resolve() not in {Path("/opt/adapter"), Path("/app/workspace")}:
        raise RuntimeError("use the OpenClaw sandbox image entrypoint")
    settings = load_settings(config)
    if settings.engine != "openclaw" or (prompt is not None and not prompt.strip()):
        raise ValueError("expected an OpenClaw profile and a nonempty prompt")
    binary = shutil.which("openclaw")
    if not binary:
        raise RuntimeError("install openclaw in the sandbox first")
    settings.native_config["gateway"] = {"mode": "local", "bind": "loopback", "port": 18789}
    client = PlatformClient(settings)
    if prompt is None and "AIDEV_PREVIEW_CONFIG" in os.environ:
        client.preview_config = json.loads(os.environ.pop("AIDEV_PREVIEW_CONFIG"))
    try:
        snapshot = await ConfigSynchronizer(settings, client).sync()
    finally:
        await client.close()

    if settings.mounted_storage:
        snapshot = install_user_config(settings, snapshot, Path("/app/.storage/user"))

    # Both processes share the same user-private state, token and immutable config.
    settings.engine_env.update(
        {
            "PATH": os.environ["PATH"],
            "HOME": str(private_directory(settings.root / "engine-home")),
            "OPENCLAW_STATE_DIR": str(
                private_directory(
                    Path("/app/.storage/user/state") if settings.mounted_storage else settings.root / "openclaw-state"
                )
            ),
            "OPENCLAW_GATEWAY_TOKEN": secrets.token_urlsafe(32),
        }
    )
    for name in ("OPENCLAW_MODEL_KEY", "OPENCLAW_APP_AUTH"):
        if name in os.environ:
            settings.engine_env[name] = os.environ[name]
    env = {**settings.engine_env, **snapshot.environment}
    settings.command = [binary, "acp"]
    adapter = Adapter(settings, snapshot)
    gateway = None
    try:
        gateway = await asyncio.create_subprocess_exec(
            binary,
            "gateway",
            "run",
            "--bind",
            "loopback",
            "--port",
            "18789",
            "--auth",
            "token",
            env=env,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await wait_for_gateway(gateway)
        if prompt is None:
            await serve_runner(
                adapter,
                os.environ["CHANNEL_RUNNER_URL"],
                Path(os.environ["CHANNEL_RUNNER_TICKET_FILE"]),
                os.environ["SANDBOX_ID"],
            )
            return
        run_id = uuid.uuid4().hex
        await adapter.submit(run_id, uuid.uuid4().hex, prompt)
        async for item in adapter.events(run_id):
            print(json.dumps(item["event"], ensure_ascii=False), flush=True)
            if item["event"]["type"] == "RUN_ERROR":
                raise RuntimeError("OpenClaw run failed")
    finally:
        try:
            await adapter.close()
        finally:
            if gateway and gateway.returncode is None:
                gateway.terminate()
                try:
                    await asyncio.wait_for(gateway.wait(), 10)
                except TimeoutError:
                    gateway.kill()
                    await gateway.wait()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--runner", action="store_true")
    args = parser.parse_args()
    if args.config:
        asyncio.run(run(args.config, None if args.runner else sys.stdin.read()))
    elif args.runner:
        # PaaS supplies bootstrap data as environment variables. Keep credentials off the user PV.
        with tempfile.TemporaryDirectory(prefix="aidev-bootstrap-") as directory:
            profile = Path(directory) / "profile.json"
            profile.write_text(os.environ.pop("AIDEV_ADAPTER_CONFIG"))
            ticket = Path(directory) / "runner.ticket"
            ticket.write_text(os.environ.pop("CHANNEL_RUNNER_TICKET"))
            ticket.chmod(0o600)
            os.environ["CHANNEL_RUNNER_TICKET_FILE"] = str(ticket)
            asyncio.run(run(profile, None))
    else:
        parser.error("--config is required for a single prompt")
