"""External process entrypoints and an isolated-fleet native streaming smoke check."""

import argparse
import asyncio
import json
import os
import shlex
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

from miles.backends.dynamo_utils.config import DynamoConfig, load_dynamo_config
from miles.backends.dynamo_utils.engine_contract import configure_engine_args
from miles.backends.dynamo_utils.engine_state import observe_fleet
from miles.backends.dynamo_utils.generate_transport import stream_generate
from miles.backends.dynamo_utils.sidecar_args import sidecar_launch
from miles.backends.dynamo_utils.stream_sample import sample_from_stream
from miles.utils.types import Sample


def engine_launch(config: DynamoConfig, *, name: str, overrides: dict) -> list[str]:
    from miles.backends.sglang_utils.server_args_utils import server_args_to_argv

    binding = next(binding for binding in config.engines if binding.name == name)
    resolved = configure_engine_args(config, binding=binding, overrides=overrides)
    return [sys.executable, "-m", "sglang.launch_server", *server_args_to_argv(resolved)]


async def smoke(
    config: DynamoConfig,
    *,
    frontend_url: str,
    weight_version: str,
    input_ids: list[int],
    n_samples: int,
    max_new_tokens: int,
) -> list[Sample]:
    """Run only as the sole client of a quiescent fleet, never beside a training job."""
    if not config.exclusive_run or not weight_version.strip():
        raise ValueError("smoke requires an isolated fleet and an explicit expected weight version")
    if not input_ids or any(type(token) is not int or token < 0 for token in input_ids):
        raise ValueError("input_ids must be a nonempty list of nonnegative token IDs")
    if n_samples <= 0 or max_new_tokens <= 0:
        raise ValueError("sample count and output budget must be positive")
    before = await observe_fleet(config, timeout=config.control_timeout_seconds)
    if any(item.weight_version != weight_version for item in before):
        raise ValueError("engines do not have the expected weight version")
    async with httpx.AsyncClient() as client:

        async def one(index):
            async with stream_generate(
                client,
                url=f"{frontend_url.rstrip('/')}/generate",
                payload={
                    "input_ids": input_ids,
                    "return_logprob": True,
                    "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 1.0},
                },
                timeout_seconds=config.request_timeout_seconds,
            ) as events:
                return await sample_from_stream(
                    Sample(index=index, group_index=0, tokens=input_ids.copy()),
                    events,
                    expected_weight_version=weight_version,
                    max_new_tokens=max_new_tokens,
                )

        async with asyncio.TaskGroup() as group:
            tasks = [group.create_task(one(index)) for index in range(n_samples)]
    after = await observe_fleet(config, timeout=config.control_timeout_seconds)
    if before != after:
        raise RuntimeError("engine incarnation or version changed during smoke generation")
    return [task.result() for task in tasks]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("engine", "sidecar"):
        command = commands.add_parser(name)
        command.add_argument("--name", required=True)
        command.add_argument("--dry-run", action="store_true", help="print argv only; do not launch")
        if name == "engine":
            command.add_argument("--overrides", type=Path, help="JSON SGLang ServerArgs overrides")
        else:
            command.add_argument("--executable", default="dynamo-sglang-sidecar")
    commands.add_parser("check", help="read native engine snapshots without sending generation requests")
    check = commands.add_parser("smoke", help="generate on an isolated idle fleet; never run beside training")
    check.add_argument("--frontend-url", required=True)
    check.add_argument("--weight-version", required=True)
    check.add_argument("--input-ids", required=True, type=json.loads)
    check.add_argument("--n-samples", default=4, type=int)
    check.add_argument("--max-new-tokens", default=32, type=int)
    args = parser.parse_args(argv)
    config = load_dynamo_config(args.config)
    if args.command == "check":
        states = asyncio.run(observe_fleet(config, timeout=config.control_timeout_seconds))
        print(json.dumps({binding.name: asdict(state) for binding, state in zip(config.engines, states, strict=True)}))
        return
    if args.command == "smoke":
        samples = asyncio.run(
            smoke(
                config,
                frontend_url=args.frontend_url,
                weight_version=args.weight_version,
                input_ids=args.input_ids,
                n_samples=args.n_samples,
                max_new_tokens=args.max_new_tokens,
            )
        )
        print(json.dumps({"samples": len(samples), "response_tokens": [sample.response_length for sample in samples]}))
        return
    if args.name not in {binding.name for binding in config.engines}:
        parser.error(f"unknown engine binding: {args.name}")
    if args.command == "engine":
        overrides = json.loads(args.overrides.read_text()) if args.overrides else {}
        argv, env = engine_launch(config, name=args.name, overrides=overrides), dict(os.environ)
    else:
        binding = next(binding for binding in config.engines if binding.name == args.name)
        argv, env = sidecar_launch(config, binding=binding, inherited_env=os.environ, executable=args.executable)
    if args.dry_run:
        print(shlex.join(argv))
    else:
        os.execvpe(argv[0], argv, env)


if __name__ == "__main__":
    main()
