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
from pathlib import Path

from aidev_agent_adapter.adapter import Adapter
from aidev_agent_adapter.cli import load_settings
from aidev_agent_adapter.config_sync import ConfigSynchronizer, private_directory
from aidev_agent_adapter.platform import PlatformClient
from aidev_agent_adapter.runner import serve_runner


async def run(config: Path, prompt: str | None) -> None:
    if sys.platform != "linux" or Path.cwd().resolve() != Path("/opt/adapter"):
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

    # Both processes share the same user-private state, token and immutable config.
    settings.engine_env.update(
        {
            "PATH": os.environ["PATH"],
            "HOME": str(private_directory(settings.root / "engine-home")),
            "OPENCLAW_STATE_DIR": str(private_directory(settings.root / "openclaw-state")),
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
        async with asyncio.timeout(60):
            while gateway.returncode is None:
                probe = await asyncio.create_subprocess_exec(
                    binary,
                    "gateway",
                    "health",
                    env=env,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                try:
                    if await probe.wait() == 0:
                        break
                finally:
                    if probe.returncode is None:
                        probe.kill()
                        await probe.wait()
                await asyncio.sleep(1)
            else:
                raise RuntimeError("OpenClaw Gateway exited before becoming ready")
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
