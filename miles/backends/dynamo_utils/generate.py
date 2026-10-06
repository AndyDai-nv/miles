from dataclasses import dataclass, replace

from miles.backends.dynamo_utils.generate_transport import stream_generate
from miles.backends.dynamo_utils.stream_sample import sample_from_stream
from miles.rollout.base_types import GenerateFnInput, GenerateFnOutput
from miles.rollout.generate_utils.generate_endpoint_utils import (
    compute_prompt_ids_from_sample,
    compute_request_payload,
)
from miles.utils.http_utils import GeneralHttpClientProvider
from miles.utils.types import Sample


@dataclass(frozen=True)
class GenerationContext:
    url: str
    weight_version: str
    timeout_seconds: float


async def generate(input: GenerateFnInput) -> GenerateFnOutput:
    context = input.state.dynamo_context
    if not isinstance(context, GenerationContext):
        raise RuntimeError("Dynamo generation requires an active, versioned rollout batch")
    sample = input.sample
    if sample.multimodal_inputs or sample.multimodal_train_inputs or sample.status is not Sample.Status.PENDING:
        raise ValueError("Dynamo generation currently accepts fresh single-turn text samples")
    if sample.response or sample.response_length or sample.rollout_log_probs or sample.weight_versions:
        raise ValueError("Dynamo generation cannot resume a partial sample")
    prompt_ids = compute_prompt_ids_from_sample(input.state, sample)
    if sample.tokens and sample.tokens != prompt_ids:
        raise ValueError("pretokenized sample does not match the prompt")
    payload, halt_status = compute_request_payload(
        input.args, input_ids=prompt_ids, sampling_params=input.sampling_params, evaluation=input.evaluation
    )
    if payload is None:
        return GenerateFnOutput(samples=replace(sample, tokens=prompt_ids, status=halt_status))
    for field in (
        "return_sampling_mask",
        "return_routed_experts",
        "return_indexer_topk",
        "top_logprobs_num",
        "lora_path",
    ):
        if payload.get(field):
            raise ValueError(f"Dynamo stream mapping does not support {field}")
    prepared = replace(sample, tokens=prompt_ids)
    async with stream_generate(
        GeneralHttpClientProvider.client(),
        url=context.url,
        payload=payload,
        timeout_seconds=context.timeout_seconds,
    ) as events:
        result = await sample_from_stream(
            prepared,
            events,
            expected_weight_version=context.weight_version,
            max_new_tokens=payload["sampling_params"]["max_new_tokens"],
        )
    return GenerateFnOutput(samples=result)
