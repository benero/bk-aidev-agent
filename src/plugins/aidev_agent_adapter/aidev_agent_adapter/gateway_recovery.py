"""Recover an unchanged OpenClaw owner only after Channel confirms its sandbox exited."""

import asyncio
import json
import re

# OpenClaw 2026.11.6 prints the same owner description for sidecars and the SQLite lease.
_OWNER = re.compile(
    r"(?:/app/\.storage/user/state/[^:\n]+\.lock|SQLite Gateway owner lease): "
    r"holder [1-9][0-9]* at (aidev-[1-9][0-9]*)/pid-ns [0-9]+, last renewed "
)


async def _recover_lock(binary, env, *args):
    process = await asyncio.create_subprocess_exec(
        binary,
        "gateway",
        "recover-lock",
        "--json",
        *args,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        async with asyncio.timeout(20):
            output, _ = await process.communicate()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    if process.returncode:
        raise RuntimeError(f"OpenClaw lock recovery command exited with status {process.returncode}")
    return json.loads(output)


async def recover_confirmed_owner(binary, env, exited_hosts):
    if not exited_hosts:
        return
    inspection = await _recover_lock(binary, env)
    if inspection.get("present") is False:
        return
    holders = inspection.get("holders")
    fingerprint = inspection.get("confirmation")
    if not isinstance(holders, list) or not holders or not isinstance(fingerprint, str):
        raise RuntimeError("Incomplete OpenClaw lock inspection")
    hosts = set()
    for holder in holders:
        match = _OWNER.match(holder) if isinstance(holder, str) else None
        if not match:
            raise RuntimeError("Unknown OpenClaw lock holder")
        hosts.add(match[1])
    if len(hosts) != 1 or not hosts.issubset(exited_hosts):
        raise RuntimeError("Channel has not confirmed this OpenClaw owner exited")
    if not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise RuntimeError("Invalid OpenClaw lock fingerprint")
    await _recover_lock(binary, env, "--confirm", fingerprint)
