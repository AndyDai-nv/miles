import asyncio

import pytest

from miles.backends.dynamo_utils.engine_state import EngineObservation
from miles.backends.dynamo_utils.sync_gate import SyncRolloutGate

pytestmark = pytest.mark.asyncio


def _gate(*, timeout=1):
    engines = [EngineObservation(11, None), EngineObservation(22, None)]
    drained = []

    async def observe():
        return tuple(engines)

    async def drain():
        drained.append(True)

    return SyncRolloutGate(observe=observe, drain=drain, timeout=timeout), engines, drained


async def _publish(gate, engines, version):
    await gate.begin_update()
    engines[:] = [EngineObservation(item.instance_id, version) for item in engines]
    await gate.finish_update(version=version)


async def test_initial_and_two_consecutive_updates():
    gate, engines, drained = _gate()
    with pytest.raises(RuntimeError, match="initial"):
        await gate.acquire(version="1")
    for version in ("1", "2"):
        await _publish(gate, engines, version)
        ticket = await gate.acquire(version=version)
        await gate.release(ticket=ticket, success=True)
    assert len(drained) == 2


async def test_update_closes_new_dispatch_and_waits_for_successful_batch():
    gate, engines, drained = _gate()
    await _publish(gate, engines, "1")
    ticket = await gate.acquire(version="1")
    update = asyncio.create_task(gate.begin_update())
    await asyncio.sleep(0)
    assert not update.done()
    with pytest.raises(RuntimeError, match="updating"):
        await gate.acquire(version="1")
    await gate.release(ticket=ticket, success=True)
    await update
    assert len(drained) == 2


@pytest.mark.parametrize("failure", ["restart", "version", "unreachable", "cancel", "missing"])
async def test_failed_update_never_reopens(failure):
    gate, engines, _ = _gate()
    await _publish(gate, engines, "1")
    await gate.begin_update()
    engines[:] = [EngineObservation(11, "2"), EngineObservation(22, "2")]
    if failure == "restart":
        engines[1] = EngineObservation(33, "2")
    elif failure == "version":
        engines[1] = EngineObservation(22, "1")
    elif failure in ("unreachable", "cancel"):

        async def fail():
            raise asyncio.CancelledError() if failure == "cancel" else OSError("connection lost")

        gate._observe = fail
    with pytest.raises((RuntimeError, ValueError, OSError, asyncio.CancelledError)):
        await gate.finish_update(version="" if failure == "missing" else "2")
    with pytest.raises(RuntimeError, match="failed"):
        await gate.acquire(version="2")
    with pytest.raises(RuntimeError, match="failed"):
        await gate.begin_update()


async def test_uncertain_request_does_not_count_as_drain():
    gate, engines, drained = _gate()
    await _publish(gate, engines, "1")
    ticket = await gate.acquire(version="1")
    with pytest.raises(RuntimeError, match="in flight"):
        await gate.release(ticket=ticket, success=False)
    with pytest.raises(RuntimeError, match="failed"):
        await gate.begin_update()
    assert len(drained) == 1


async def test_lost_batch_or_controller_response_times_out_closed():
    gate, engines, _ = _gate(timeout=0.02)
    await _publish(gate, engines, "1")
    await gate.acquire(version="1")
    with pytest.raises(TimeoutError):
        await gate.begin_update()
    with pytest.raises(RuntimeError, match="failed"):
        await gate.acquire(version="1")


async def test_restart_after_generation_discards_batch():
    gate, engines, _ = _gate()
    await _publish(gate, engines, "1")
    ticket = await gate.acquire(version="1")
    engines[0] = EngineObservation(99, "1")
    with pytest.raises(RuntimeError, match="restarted"):
        await gate.release(ticket=ticket, success=True)


async def test_old_ticket_cannot_complete_new_batch():
    gate, engines, _ = _gate()
    await _publish(gate, engines, "1")
    old = await gate.acquire(version="1")
    await gate.release(ticket=old, success=True)
    await gate.acquire(version="1")
    with pytest.raises(ValueError, match="ticket"):
        await gate.release(ticket=old, success=True)
    with pytest.raises(RuntimeError, match="failed"):
        await gate.begin_update()


async def test_failed_drain_prevents_weight_transfer():
    gate, _, _ = _gate()

    async def fail():
        raise TimeoutError("scheduler still busy")

    gate._drain = fail
    with pytest.raises(TimeoutError):
        await gate.begin_update()
    with pytest.raises(RuntimeError, match="failed"):
        await gate.finish_update(version="1")


@pytest.mark.parametrize("operation", ["finish", "acquire", "release"])
async def test_removal_during_observation_cannot_reopen_gate(operation):
    gate, engines, _ = _gate()
    await _publish(gate, engines, "1")
    if operation == "finish":
        await gate.begin_update()
    elif operation == "release":
        ticket = await gate.acquire(version="1")

    async def observe():
        gate.fail()
        return tuple(engines)

    gate._observe = observe
    with pytest.raises(RuntimeError, match="failed"):
        if operation == "finish":
            await gate.finish_update(version="1")
        elif operation == "acquire":
            await gate.acquire(version="1")
        else:
            await gate.release(ticket=ticket, success=True)
    with pytest.raises(RuntimeError, match="failed"):
        await gate.acquire(version="1")
