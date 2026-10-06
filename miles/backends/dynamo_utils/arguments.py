from pathlib import Path

from miles.backends.dynamo_utils.config import DynamoConfig, load_dynamo_config
from miles.utils.environ import use_legacy_rollout_v1
from miles.utils.lora.utils import lora_rollout_enabled

ROLLOUT_PATH = "miles.backends.dynamo_utils.rollout.DynamoRolloutFn"
PROVIDER_PATH = "miles.backends.dynamo_utils.launch.external_engine_provider"


def configure_dynamo(args) -> None:
    """Resolve the opt-in synchronous text backend; refuse unsupported control paths."""
    value = args.dynamo_config
    config = value if isinstance(value, DynamoConfig) else load_dynamo_config(Path(value))
    if not config.exclusive_run:
        raise ValueError(
            "Dynamo requires exclusive_run=true: isolate generation and weight-control access to this run"
        )
    if use_legacy_rollout_v1():
        raise ValueError("Dynamo requires the class-based rollout API")
    # These modes bypass this synchronous gate or require data the stream mapper does not carry.
    for name in (
        "fully_async",
        "colocate",
        "partial_rollout",
        "multi_lora",
        "use_session_server",
        "use_opd",
        "recompute_logprobs_via_prefill",
        "use_sampling_support_replay",
        "use_rollout_routing_replay",
        "use_rollout_indexer_replay",
        "rollout_top_logprobs_num",
        "offload_rollout",
        "use_fault_tolerance",
        "ft_components",
        "megatron_config",
        "debug_train_only",
        "debug_rollout_only",
        "debug_skip_weight_update",
        "load_debug_rollout_data",
        "sglang_config",
        "sglang_api_key",
        "rollout_external_router_pd",
        "use_miles_router",
        "eval_num_gpus",
        "dynamic_sampling_filter_path",
        "rollout_sample_filter_path",
        "rollout_all_samples_process_path",
        "custom_generate_function_path",
    ):
        if getattr(args, name, None):
            raise ValueError(f"Dynamo synchronous text rollout does not support --{name.replace('_', '-')}")
    if lora_rollout_enabled(args):
        raise ValueError("Dynamo stream mapping does not support LoRA")
    if args.eval_interval is not None or args.eval_function_path is not None:
        raise ValueError("Dynamo synchronous rollout does not yet implement evaluation")
    if args.deploy_component != "all" or args.entry != "train":
        raise ValueError("Dynamo currently requires the all-in-one synchronous training driver")
    if args.update_weight_transfer_mode == "disk-delta":
        raise ValueError("Dynamo engine identity checks do not yet support disk-delta model-path changes")
    if args.hf_checkpoint != config.model_path:
        raise ValueError("Dynamo model_path must match --hf-checkpoint")
    if args.rollout_submission_granularity not in (None, "group"):
        raise ValueError("Dynamo uses complete synchronous prompt groups")
    if args.over_sampling_batch_size not in (None, args.rollout_batch_size):
        raise ValueError("Dynamo currently requires over_sampling_batch_size == rollout_batch_size")
    for name, expected in (
        ("rollout_function_path", ROLLOUT_PATH),
        ("custom_inference_engine_provider_path", PROVIDER_PATH),
    ):
        if getattr(args, name) not in (None, expected):
            raise ValueError(f"{name} conflicts with --dynamo-config")
        setattr(args, name, expected)
    total_gpus = sum(binding.tensor_parallel_size for binding in config.engines)
    if args.rollout_num_gpus not in (None, total_gpus):
        raise ValueError("rollout GPU count does not match Dynamo bindings")
    if args.rollout_num_gpus_per_engine != config.engines[0].tensor_parallel_size:
        raise ValueError("rollout TP size does not match Dynamo bindings")
    # The manifest supplies omitted layout; explicitly conflicting layout above is rejected.
    args.rollout_num_gpus = total_gpus
    args.dynamo_config = config
