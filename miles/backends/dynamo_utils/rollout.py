import asyncio
from contextlib import suppress

from miles.backends.dynamo_utils.generate import GenerationContext, generate
from miles.rollout.base_types import (
    BaseRolloutFn,
    RolloutFnConstructorInput,
    RolloutFnTrainInput,
    RolloutFnTrainOutput,
)
from miles.rollout.filter_hub.common_filters import apply_preput_filters
from miles.rollout.inference_rollout.inference_rollout_common import GenerateState, generate_and_rm_group
from miles.utils.types import Sample


class DynamoRolloutFn(BaseRolloutFn):
    def __init__(self, input: RolloutFnConstructorInput):
        super().__init__(input)
        self._controller = input.inference_controller
        if self._controller is None:
            raise ValueError("Dynamo rollout requires the inference controller handle")
        self._state = GenerateState(input.args)
        self._state.generate_function = generate
        self._state.dynamo_context = None
        self._lock = asyncio.Lock()

    async def __call__(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        if input.evaluation or input.weight_version is None or input.trainer_model_id is not None:
            raise ValueError("Dynamo rollout requires a versioned, single-policy training batch")
        async with self._lock:
            return await self._run_batch(input)

    async def _run_batch(self, input: RolloutFnTrainInput) -> RolloutFnTrainOutput:
        args = self._state.args
        timeout = args.dynamo_config.control_timeout_seconds
        version = str(input.weight_version)
        async with asyncio.timeout(timeout):
            ticket = await self._controller.dynamo_acquire_rollout(version=version)
        tasks = []
        finished = False
        try:
            host = args.sglang_router_ip
            host = f"[{host}]" if ":" in host and not host.startswith("[") else host
            self._state.dynamo_context = GenerationContext(
                url=f"http://{host}:{args.sglang_router_port}/generate",
                weight_version=version,
                timeout_seconds=args.dynamo_config.request_timeout_seconds,
            )
            groups = self.constructor_input.data_source.get_samples(args.rollout_batch_size)
            if len(groups) != args.rollout_batch_size or any(
                len(group) != args.n_samples_per_prompt for group in groups
            ):
                raise ValueError("data source returned an incomplete GRPO batch")
            if any(sample.generate_function_path for group in groups for sample in group):
                raise ValueError("per-sample generate overrides would bypass the Dynamo version contract")
            if any(sample.status is not Sample.Status.PENDING for group in groups for sample in group):
                raise ValueError("Dynamo batches require fresh samples, not previously generated responses")
            tasks = [
                asyncio.create_task(generate_and_rm_group(self._state, group, self._state.sampling_params.copy()))
                for group in groups
            ]
            samples = await asyncio.gather(*tasks)
            for group in samples:
                if not (result := apply_preput_filters(args, None, group)).keep:
                    raise ValueError(f"Dynamo batch is not trainable: {result.reason}")
            async with asyncio.timeout(timeout):
                await self._controller.dynamo_release_rollout(ticket=ticket, success=True)
            finished = True
            return RolloutFnTrainOutput(samples=samples)
        finally:
            if not finished:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                # Failure is terminal: closing streams does not establish engine drain.
                with suppress(Exception):
                    async with asyncio.timeout(timeout):
                        await self._controller.dynamo_release_rollout(ticket=ticket, success=False)
            self._state.dynamo_context = None
