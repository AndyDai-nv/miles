# Dynamo sidecar quick start

Draft: synchronous single-policy text only; no eval/async/LoRA/PD/recovery.
Image builds and the 8-GPU training run have not been validated.

## Prerequisites

- Use the revisions in `dynamo.json`: SGLang from `sglang-miles`, including its
  matching native gRPC extension (`SGLANG_BUILD_RUST_EXTS=grpc` when installing).
- Build/install Dynamo's `lib/bindings/python`, root Python package and
  `dynamo-sglang-sidecar` executable. Do not install the `[sglang]` extra: it
  replaces SGLang-miles. Install Miles with `pip install -e '.[dynamo]'`.
- Provide converted Qwen3-4B weights, DAPO data and private etcd on port 2379.
  Edit the model path/addresses/namespace in `dynamo.json` and engine overrides
  in `engine-0.json` / `engine-1.json`. KV event ports are 5557 and 5558.

## Launch

Run each process below in its own container with a private shared network.
The example uses same-host loopback; use routable addresses across hosts.
Expose GPUs 0–3 to the trainer, 4–5 to engine-0, and 6–7 to engine-1, with the
same model mount path. Keep external engines outside the trainer PID namespace:
the standard Miles launcher cleans previous processes.

```bash
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json engine --name engine-0 --overrides examples/dynamo_sidecar/engine-0.json
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json engine --name engine-1 --overrides examples/dynamo_sidecar/engine-1.json
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json sidecar --name engine-0
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json sidecar --name engine-1
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json check
python scripts/run_qwen3_4b_dynamo.py --dynamo-config examples/dynamo_sidecar/dynamo.json --num-rollout 2
```

Miles starts the frontend/router. `engine` and `sidecar` accept `--dry-run`.
Use `deployment smoke --help` for transport diagnostics on a separate idle fleet,
never alongside training; smoke neither initializes weights nor starts a frontend.

`exclusive_run=true` does not enforce isolation: prevent other generation/control
clients and unrelated workers in the namespace. An uncertain request or engine
restart requires restarting the run after draining the old fleet. Miles does not
stop externally owned processes or remove their discovery records.
