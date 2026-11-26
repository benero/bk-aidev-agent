from types import SimpleNamespace

import httpx
import pytest
from aidev_agent_adapter.openclaw import wait_for_gateway


async def test_register_only_after_gateway_is_ready(monkeypatch):
    responses = iter([503, 200])
    probes = []

    async def get(client, url):
        probes.append(url)
        return httpx.Response(next(responses))

    monkeypatch.setattr(httpx.AsyncClient, "get", get)
    await wait_for_gateway(SimpleNamespace(returncode=None))
    assert probes == ["http://127.0.0.1:18789/readyz"] * 2


async def test_gateway_exit_is_reported_before_registration():
    with pytest.raises(RuntimeError, match="status 7"):
        await wait_for_gateway(SimpleNamespace(returncode=7))
