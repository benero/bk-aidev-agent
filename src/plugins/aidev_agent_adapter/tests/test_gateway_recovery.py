from unittest.mock import AsyncMock

import pytest
from aidev_agent_adapter import gateway_recovery

OWNER = "/app/.storage/user/state/tmp/openclaw-1000/gateway.state.lock: holder 19 at aidev-13/pid-ns 42, last renewed 500 s ago."
FINGERPRINT = "a" * 64


@pytest.mark.parametrize(
    "hosts,holders,allowed",
    [
        ([], [OWNER], False),
        (["aidev-13"], [OWNER], True),
        (["aidev-13"], [OWNER, OWNER.replace("aidev-13", "aidev-14")], False),
        (["aidev-13"], ["unknown"], False),
        (["aidev-14"], [OWNER], False),
    ],
)
async def test_only_confirmed_unchanged_owner_is_recovered(monkeypatch, hosts, holders, allowed):
    cli = AsyncMock(return_value={"present": True, "holders": holders, "confirmation": FINGERPRINT})
    monkeypatch.setattr(gateway_recovery, "_recover_lock", cli)
    if hosts and not allowed:
        with pytest.raises(RuntimeError):
            await gateway_recovery.recover_confirmed_owner("openclaw", {}, hosts)
    else:
        await gateway_recovery.recover_confirmed_owner("openclaw", {}, hosts)
    assert cli.await_count == (2 if allowed else int(bool(hosts)))
    if allowed:
        assert cli.await_args.args[-2:] == ("--confirm", FINGERPRINT)


async def test_fingerprint_changed_during_confirmation_stops_startup(monkeypatch):
    inspection = {"present": True, "holders": [OWNER], "confirmation": FINGERPRINT}
    cli = AsyncMock(side_effect=[inspection, RuntimeError("fingerprint changed")])
    monkeypatch.setattr(gateway_recovery, "_recover_lock", cli)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        await gateway_recovery.recover_confirmed_owner("openclaw", {}, ["aidev-13"])
    assert cli.await_count == 2
