"""Run inside one user's sandbox. Read the prompt from stdin, emit AG-UI envelopes."""

import argparse
import asyncio
import json
import os
import secrets
import shutil
import sys
import uuid
from pathlib import Path

from aidev_agent_adapter.adapter import Adapter
from aidev_agent_adapter.cli import load_settings
from aidev_agent_adapter.config_sync import ConfigSynchronizer, private_directory
from aidev_agent_adapter.platform import PlatformClient


async def run(config: Path, prompt: str) -> None:
    if sys.platform != "linux" or Path(__file__).resolve() != Path("/opt/adapter/demo.py"):
        raise RuntimeError("use the OpenClaw sandbox image entrypoint")
    settings = load_settings(config)
    if settings.engine != "openclaw" or not prompt.strip():
        raise ValueError("expected an OpenClaw profile and a nonempty prompt")
    binary = shutil.which("openclaw")
    if not binary:
        raise RuntimeError("install openclaw in the sandbox first")
    settings.native_config["gateway"] = {"mode": "local", "bind": "loopback", "port": 18789}
    client = PlatformClient(settings)
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
    if "OPENCLAW_MODEL_KEY" in os.environ:
        settings.engine_env["OPENCLAW_MODEL_KEY"] = os.environ["OPENCLAW_MODEL_KEY"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    asyncio.run(run(args.config, sys.stdin.read()))
