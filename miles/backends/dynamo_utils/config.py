import ipaddress
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from miles.utils.pydantic_utils import FrozenStrictBaseModel

Name = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]
Revision = Annotated[str, Field(pattern=r"^[0-9a-f]{40}$")]


class Address(FrozenStrictBaseModel):
    host: str
    port: Annotated[int, Field(strict=True, ge=1, le=65535)]

    @field_validator("host")
    @classmethod
    def _validate_host(cls, host: str) -> str:
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            if (
                not host
                or len(host) > 253
                or any(
                    not re.fullmatch(r"[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?", label)
                    for label in host.rstrip(".").split(".")
                )
            ):
                raise ValueError("host must be a bare IP address or DNS name") from None
        else:
            if "%" in host:
                raise ValueError("scoped IP addresses are not supported")
            return address.compressed
        return host.lower().rstrip(".")

    @property
    def url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


class EngineBinding(FrozenStrictBaseModel):
    name: Name
    http: Address
    grpc: Address
    sidecar: Address
    tensor_parallel_size: Annotated[int, Field(strict=True, gt=0)]

    @model_validator(mode="after")
    def _validate_addresses(self) -> Self:
        # Native /generate derives its HTTP host from the gRPC endpoint.
        if self.http.host != self.grpc.host:
            raise ValueError("SGLang HTTP and gRPC must use the same reachable host")
        for address in (self.http, self.grpc, self.sidecar):
            if address.host in ("0.0.0.0", "::"):
                raise ValueError("engine bindings require reachable hosts, not wildcard listeners")
        if len({self.http, self.grpc, self.sidecar}) != 3:
            raise ValueError("HTTP, gRPC and sidecar system addresses must be distinct")
        return self


class EtcdDiscovery(FrozenStrictBaseModel):
    backend: Literal["etcd"] = "etcd"
    endpoints: Annotated[tuple[Address, ...], Field(min_length=1)]


class FileDiscovery(FrozenStrictBaseModel):
    backend: Literal["file"] = "file"
    root: Path

    @field_validator("root")
    @classmethod
    def _absolute_root(cls, root: Path) -> Path:
        if not root.is_absolute():
            raise ValueError("file discovery requires an absolute shared directory")
        return root


class SourcePins(FrozenStrictBaseModel):
    dynamo: Revision
    sglang: Revision
    sglang_branch: Literal["sglang-miles"] = "sglang-miles"


class DynamoConfig(FrozenStrictBaseModel):
    namespace: Name
    model_path: Annotated[str, Field(min_length=1)]
    discovery: Annotated[EtcdDiscovery | FileDiscovery, Field(discriminator="backend")]
    engines: Annotated[tuple[EngineBinding, ...], Field(min_length=1)]
    pins: SourcePins

    @model_validator(mode="after")
    def _validate_fleet(self) -> Self:
        if not self.model_path.strip():
            raise ValueError("model_path must not be blank")
        if len({engine.name for engine in self.engines}) != len(self.engines):
            raise ValueError("engine names must be unique")
        addresses = [address for engine in self.engines for address in (engine.http, engine.grpc, engine.sidecar)]
        if len(set(addresses)) != len(addresses):
            raise ValueError("engine addresses must not overlap")
        if len({engine.tensor_parallel_size for engine in self.engines}) != 1:
            raise ValueError("the static external provider requires a uniform tensor parallel size")
        return self


def load_dynamo_config(path: Path) -> DynamoConfig:
    return DynamoConfig.model_validate_json(path.read_text())


def runtime_env(config: DynamoConfig, *, inherited_env: Mapping[str, str]) -> dict[str, str]:
    # The resolved run owns Dynamo settings; inherited flags must not change its topology or routing.
    env = {key: value for key, value in inherited_env.items() if not key.startswith("DYN_")}
    env.pop("ETCD_ENDPOINTS", None)
    env.update(
        DYN_NAMESPACE=config.namespace,
        DYN_DISCOVERY_BACKEND=config.discovery.backend,
        DYN_REQUEST_PLANE="tcp",
        DYN_RESPONSE_PLANE="tcp",
        DYN_EVENT_PLANE="zmq",
    )
    match config.discovery:
        case EtcdDiscovery(endpoints=endpoints):
            env["ETCD_ENDPOINTS"] = ",".join(address.url for address in endpoints)
        case FileDiscovery(root=root):
            env["DYN_FILE_KV"] = str(root)
    return env
