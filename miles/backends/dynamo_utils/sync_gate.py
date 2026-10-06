import asyncio
import math
import uuid
from collections.abc import Awaitable, Callable, Sequence

from miles.backends.dynamo_utils.engine_state import EngineObservation


class SyncRolloutGate:
    """Single-owner synchronous batches. Unknown remote outcomes require restarting the run."""

    def __init__(
        self,
        *,
        observe: Callable[[], Awaitable[Sequence[EngineObservation]]],
        drain: Callable[[], Awaitable[None]],
        timeout: float,
    ):
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("gate timeout must be finite and positive")
        self._observe = observe
        self._drain = drain
        self._timeout = timeout
        self._lock = asyncio.Lock()
        self._idle = asyncio.Event()
        self._idle.set()
        self._identities: tuple[int, ...] | None = None
        self._version: str | None = None
        self._state = "initial"
        self._ticket: str | None = None

    async def begin_update(self) -> None:
        async with self._lock:
            self._require("initial", "ready")
            self._state = "updating"
        try:
            async with asyncio.timeout(self._timeout):
                await self._idle.wait()
                async with self._lock:
                    self._require("updating")
                    await self._verify(expected=self._version)
                    # Exclusive request ownership plus a scheduler-confirmed flush,
                    # not HTTP connection closure, establishes the update boundary.
                    await self._drain()
                    self._require("updating")
        except BaseException:
            self.fail()
            raise

    async def finish_update(self, *, version: str) -> None:
        async with self._lock:
            try:
                self._require("updating")
                if not isinstance(version, str) or not version.strip():
                    raise ValueError("trainer must publish a nonempty weight version")
                async with asyncio.timeout(self._timeout):
                    await self._verify(expected=version)
                self._require("updating")
                self._version = version
                self._state = "ready"
            except BaseException:
                self.fail()
                raise

    async def acquire(self, *, version: str) -> str:
        async with self._lock:
            self._require("ready")
            if self._ticket is not None:
                raise RuntimeError("concurrent Dynamo rollout/evaluation batches are unsupported")
            try:
                if version != self._version:
                    raise ValueError("rollout version does not match the trainer publication")
                async with asyncio.timeout(self._timeout):
                    await self._verify(expected=version)
                self._require("ready")
            except BaseException:
                self.fail()
                raise
            self._ticket = uuid.uuid4().hex
            self._idle.clear()
            return self._ticket

    async def release(self, *, ticket: str, success: bool) -> None:
        async with self._lock:
            try:
                if self._ticket is None or ticket != self._ticket:
                    raise ValueError("unknown rollout ticket")
                self._require("ready", "updating")
                if not success:
                    raise RuntimeError("rollout did not finish cleanly; remote work may still be in flight")
                async with asyncio.timeout(self._timeout):
                    await self._verify(expected=self._version)
                self._require("ready", "updating")
            except BaseException:
                self.fail()
                raise
            finally:
                self._ticket = None
                self._idle.set()

    def fail(self) -> None:
        self._state = "failed"

    def _require(self, *states: str) -> None:
        if self._state not in states:
            raise RuntimeError(f"Dynamo rollout gate is {self._state}; expected {states}")

    async def _verify(self, *, expected: str | None) -> None:
        observations = await self._observe()
        identities = tuple(item.instance_id for item in observations)
        if not identities or any(identity <= 0 for identity in identities):
            raise ValueError("missing engine identities")
        if self._identities is not None and identities != self._identities:
            raise RuntimeError("engine restarted or fleet changed; restart the training run")
        if expected is not None and any(item.weight_version != expected for item in observations):
            raise ValueError("not every engine has the trainer's expected weight version")
        self._identities = identities
