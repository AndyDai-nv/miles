import asyncio
from dataclasses import dataclass

import httpx

from miles.backends.dynamo_utils.config import Address
from miles.backends.sglang_utils.sglang_api_client import SGLangApiClient, WorkerType
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.workers.worker_provider.base import BaseWorkerProvider


@dataclass
class DynamoServingRegistry:
    provider: BaseWorkerProvider
    worker_name: str
    api_key: str | None
    timeout_seconds: float
    _expected_version: str | None = None
    _session: str | None = None

    async def _endpoint(self) -> str:
        address = (await self.provider.get_addrs(worker_name=self.worker_name))["dynamo_system"]
        return Address(host=address.host.strip("[]"), port=address.port).url

    async def _post(self, endpoint: str, action: str, payload: dict) -> dict:
        while True:
            try:
                response = await GeneralHttpClientProvider.client().post(
                    f"{endpoint}/engine/admission/{action}", json=payload, timeout=self.timeout_seconds
                )
                response.raise_for_status()
                break
            except httpx.HTTPStatusError as error:
                if error.response.status_code not in (408, 429, 500, 502, 503, 504):
                    raise
            except httpx.TransportError:
                pass
            # Retrying a mutation preserves its session/revision; it cannot grant a new incarnation.
            await asyncio.sleep(0.1)
        result = response.json()
        if result.get("status") != "ok" or result.get("shutting_down"):
            raise RuntimeError(f"sidecar admission {action} failed: {result}")
        if self._session is None:
            self._session = result["session"]
        elif self._session != result["session"]:
            raise RuntimeError("sidecar restarted; the cell must be reconciled before readmission")
        return result

    async def _change_admission(self, *, admit: bool, expected_version: str | None = None) -> None:
        async with asyncio.timeout(self.timeout_seconds):
            endpoint = await self._endpoint()
            status = await self._post(endpoint, "status", {})
            if admit:
                while (
                    not status.get("engine_ready")
                    or status.get("observed") is None
                    or status["observed"]["weight_version"] != expected_version
                ):
                    await asyncio.sleep(0.1)
                    status = await self._post(endpoint, "status", {})
                observed = status["observed"]
                if status["admitted"] and status["published"] is True and status["expected"] == observed:
                    return
            command = {"session": status["session"], "revision": status["revision"] + 1}
            if admit:
                command["expected"] = {**observed, "weight_version": expected_version}
            await self._post(endpoint, "admit" if admit else "withdraw", command)
            while True:
                status = await self._post(endpoint, "status", {})
                if status["session"] != command["session"] or status["revision"] != command["revision"]:
                    raise RuntimeError("sidecar restarted or admission command was superseded")
                if status.get("last_error"):
                    raise RuntimeError(f"sidecar admission failed: {status['last_error']}")
                if status["applied_revision"] == command["revision"] and status["published"] is admit:
                    if admit and (not status["admitted"] or status["expected"] != command["expected"]):
                        raise RuntimeError("sidecar admission was revoked")
                    return
                await asyncio.sleep(0.1)

    async def register(self, *, worker_url: str, worker_type: WorkerType, bootstrap_port: int | None) -> None:
        if worker_type != WorkerType.REGULAR:
            raise ValueError("Dynamo registry currently requires aggregated SGLang engines")
        # Read-only/debug models have no trainer commit; training sets this in finish_update.
        version = self._expected_version
        if version is None:
            async with asyncio.timeout(self.timeout_seconds):
                version = await SGLangApiClient(server_url=worker_url, api_key=self.api_key).get_weight_version()
        if not isinstance(version, str) or not version:
            raise ValueError("SGLang must publish a non-empty weight version before admission")
        await self._change_admission(admit=True, expected_version=version)

    async def unregister(self, *, worker_url: str) -> None:
        await self._change_admission(admit=False)

    async def prepare_update(self, *, worker_url: str) -> None:
        await self.unregister(worker_url=worker_url)

    async def finish_update(self, *, worker_url: str, weight_version: str | None) -> None:
        if weight_version is None:
            raise ValueError("Dynamo admission requires the trainer's completed weight version")
        self._expected_version = weight_version
        await self.register(worker_url=worker_url, worker_type=WorkerType.REGULAR, bootstrap_port=None)

    async def check_health(self, *, worker_url: str, serving: bool) -> None:
        async with asyncio.timeout(self.timeout_seconds):
            status = await self._post(await self._endpoint(), "status", {})
        if not status["engine_ready"]:
            raise RuntimeError("sidecar lost its healthy engine observation")
        if serving and (not status["admitted"] or status["published"] is not True):
            raise RuntimeError("sidecar serving grant was revoked")
