"""Encode Miles rollout-control operations for Dynamo's engine API.

Dynamo does not mirror every SGLang HTTP endpoint.  Its SGLang worker exposes
two control mechanisms instead:

* first-class ``/engine/<route>`` handlers for the operations Dynamo owns; and
* an RL-only ``/engine/call_tokenizer_manager`` escape hatch for the remaining
  SGLang tokenizer-manager operations.

This module captures that routing contract without importing Dynamo, SGLang,
HTTP clients, or rollout worker types.  A later API-client change can therefore
reuse it while the Miles rollout refactor is still in flight.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

CALL_TOKENIZER_MANAGER_ROUTE = "call_tokenizer_manager"

# These routes are registered unconditionally by Dynamo's SGLang worker.  They
# must stay direct: some add Dynamo lifecycle behavior around the underlying
# tokenizer-manager call (for example discovery unregister/register on memory
# release/resume).
DIRECT_ENGINE_ROUTES = frozenset(
    {
        "release_memory_occupation",
        "resume_memory_occupation",
        "stop_profile",
        "update_weight_version",
        "update_weights_from_disk",
        "update_weights_from_distributed",
        "update_weights_from_ipc",
        "update_weights_from_tensor",
    }
)


@dataclass(frozen=True)
class DynamoControlRequest:
    """One JSON request to Dynamo's system/control server."""

    route: str
    body: dict[str, Any]

    @property
    def path(self) -> str:
        """Return the HTTP path exposed by Dynamo's engine API."""
        return f"/engine/{self.route}"


@dataclass(frozen=True)
class _TokenizerManagerOperation:
    method: str
    io_struct_name: str | None


# Operation names match SGLangApiClient's public method names.  The class names
# are from the sglang-miles io_struct contract; Dynamo resolves this tagged JSON
# representation into the corresponding request object at the worker.
_TOKENIZER_MANAGER_OPERATIONS = {
    "begin_weight_update": _TokenizerManagerOperation(
        method="begin_weight_update",
        io_struct_name="BeginWeightUpdateReqInput",
    ),
    "check_weights": _TokenizerManagerOperation(
        method="check_weights",
        io_struct_name="CheckWeightsReqInput",
    ),
    "continue_generation": _TokenizerManagerOperation(
        method="continue_generation",
        io_struct_name="ContinueGenerationReqInput",
    ),
    "destroy_weights_update_group": _TokenizerManagerOperation(
        method="destroy_weights_update_group",
        io_struct_name="DestroyWeightsUpdateGroupReqInput",
    ),
    "end_weight_update": _TokenizerManagerOperation(
        method="end_weight_update",
        io_struct_name="EndWeightUpdateReqInput",
    ),
    # Unlike the other passthrough operations, TokenizerManager.flush_cache
    # accepts timeout_s directly rather than an io_struct request object.
    "flush_cache": _TokenizerManagerOperation(
        method="flush_cache",
        io_struct_name=None,
    ),
    "init_weights_update_group": _TokenizerManagerOperation(
        method="init_weights_update_group",
        io_struct_name="InitWeightsUpdateGroupReqInput",
    ),
    "pause_generation": _TokenizerManagerOperation(
        method="pause_generation",
        io_struct_name="PauseGenerationReqInput",
    ),
    "pull_weights": _TokenizerManagerOperation(
        method="pull_weights",
        io_struct_name="PullWeightsReqInput",
    ),
    # Current sglang-miles accepts one ProfileReq object, while Dynamo's
    # first-class start_profile handler still expands the body as **kwargs.
    # Use the typed RL path until that upstream route follows the new API.
    "start_profile": _TokenizerManagerOperation(
        method="start_profile",
        io_struct_name="ProfileReq",
    ),
}

SUPPORTED_CONTROL_OPERATIONS = DIRECT_ENGINE_ROUTES | frozenset(_TOKENIZER_MANAGER_OPERATIONS)


def typed_io_struct(class_name: str, fields: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Encode a request object understood by Dynamo's RL passthrough.

    Dynamo recognizes a one-key mapping named ``io_struct.<ClassName>`` and
    constructs that class from ``sglang.srt.managers.io_struct`` using the
    nested mapping as keyword arguments.
    """
    if not class_name or not class_name.isidentifier():
        raise ValueError(f"invalid SGLang io_struct class name: {class_name!r}")
    return {f"io_struct.{class_name}": _copy_mapping(fields, name="fields")}


def tokenizer_manager_request(
    method: str,
    *,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> DynamoControlRequest:
    """Build a low-level ``call_tokenizer_manager`` request."""
    if not method or not method.isidentifier():
        raise ValueError(f"invalid tokenizer-manager method name: {method!r}")
    if isinstance(args, (str, bytes)) or not isinstance(args, Sequence):
        raise TypeError("args must be a non-string sequence")

    return DynamoControlRequest(
        route=CALL_TOKENIZER_MANAGER_ROUTE,
        body={
            "method": method,
            "args": list(args),
            "kwargs": _copy_mapping(kwargs, name="kwargs"),
        },
    )


def build_dynamo_control_request(
    operation: str,
    payload: Mapping[str, Any] | None = None,
) -> DynamoControlRequest:
    """Translate one Miles/SGLang control operation to Dynamo's wire shape.

    ``payload`` follows the corresponding SGLang HTTP request schema.  This
    function only selects the transport and wraps the payload; it deliberately
    leaves value construction (for example dtype normalization) to the future
    API client that owns that public method signature.
    """
    body = _copy_mapping(payload, name="payload")

    if operation in DIRECT_ENGINE_ROUTES:
        return DynamoControlRequest(route=operation, body=body)

    spec = _TOKENIZER_MANAGER_OPERATIONS.get(operation)
    if spec is None:
        supported = ", ".join(sorted(SUPPORTED_CONTROL_OPERATIONS))
        raise ValueError(f"unsupported Dynamo control operation {operation!r}; supported operations: {supported}")

    if spec.io_struct_name is None:
        return tokenizer_manager_request(spec.method, kwargs=body)

    return tokenizer_manager_request(
        spec.method,
        args=[typed_io_struct(spec.io_struct_name, body)],
    )


def _copy_mapping(value: Mapping[str, Any] | None, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping or None")
    return dict(value)
