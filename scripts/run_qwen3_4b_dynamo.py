"""Qwen3-4B GRPO: four trainer GPUs and two Miles-managed TP2 engines.

Requires converted Megatron weights, DAPO data, and the pinned SGLang-miles build.
Dynamo additionally requires its admission build and private discovery services.

Args:
  --router: dynamo or cache_aware; both run the same training recipe.
  --dynamo-config: Dynamo configuration; used only with --router dynamo.
  --model-dir / --data-dir / --output-dir: Mounted checkpoint, dataset and output roots.
  --cuda-visible-devices: Full eight-GPU slice for training and inference.
  --num-gpus-per-node: Ray capacity (default: 8).
  --num-rollout: Number of complete synchronous GRPO batches (default: 20).

  python scripts/run_qwen3_4b_dynamo.py --dynamo-config /config/dynamo.json
"""

import os
import shlex
from dataclasses import dataclass, field
from typing import Literal

import typer

from miles.utils.external_utils import command_utils as U


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = field(default_factory=lambda: U.create_run_id())
    dynamo_config: str = "examples/dynamo_sidecar/dynamo.json"
    router: Literal["dynamo", "cache_aware"] = "dynamo"
    model_dir: str = "/root/models"
    data_dir: str = "/root/datasets"
    megatron_path: str = "/root/Megatron-LM"
    cuda_visible_devices: str = "0,1,2,3,4,5,6,7"
    num_gpus_per_node: int = 8
    num_rollout: int = 20
    extra_args: str = ""


def execute(args: ScriptArgs):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    checkpoint = (
        f"--hf-checkpoint {shlex.quote(args.model_dir + '/Qwen3-4B')} "
        f"--ref-load {shlex.quote(args.model_dir + '/Qwen3-4B_torch_dist')} "
        f"--load {shlex.quote(args.model_dir + '/Qwen3-4B_torch_dist')} "
        "--no-load-optim --no-load-rng "
    )
    rollout = (
        (
            f"--dynamo-config {shlex.quote(args.dynamo_config)} "
            if args.router == "dynamo"
            else "--sglang-router-policy cache_aware "
        )
        + f"--prompt-data {shlex.quote(args.data_dir + '/dapo-math-17k/dapo-math-17k.jsonl')} "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type deepscaler --rollout-batch-size 4 --n-samples-per-prompt 4 "
        "--global-batch-size 16 --rollout-max-response-len 1024 --rollout-temperature 1 "
        f"--num-rollout {args.num_rollout} --rollout-num-gpus 4 --rollout-num-gpus-per-engine 2 "
    )
    performance = (
        "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 2048 "
    )
    algorithm = "--advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 "
    optimizer = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 "
    )
    misc = (
        "--attention-dropout 0 --hidden-dropout 0 --attention-backend flash "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 4 --seed 42 "
        "--sglang-page-size 16 --sglang-mem-fraction-static 0.7 "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        f"--save-debug-rollout-data {shlex.quote(args.output_dir + '/rollout_{rollout_id}.pt')} "
    )
    args.create_backend().execute_train(
        train_args=checkpoint
        + rollout
        + performance
        + algorithm
        + optimizer
        + misc
        + U.get_default_wandb_args(__file__, run_id=args.run_id)
        + " "
        + args.extra_args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type="qwen3-4B",
        megatron_path=args.megatron_path,
    )


@U.dataclass_cli
def main(args: ScriptArgs):
    execute(args)


if __name__ == "__main__":
    typer.run(main)
