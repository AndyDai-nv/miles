# Synchronous Miles + standalone Dynamo sidecar

Draft integration: single policy, text-only GRPO, no async/eval/LoRA/PD or recovery.
The 8-GPU training recipe below still needs GPU acceptance testing. CPU tests and
source-level parser/protobuf checks do not establish NCCL or end-to-end correctness.

## Build prerequisites

Use the exact source revisions in `dynamo.json`. `pins` records provenance; it does
not install packages or prove which binary an operator deployed.

- Build the normal Miles image with `SGLANG_BRANCH=sglang-miles` and
  `SGLANG_COMMIT=14a1fa7d93e87355375b2c3b6fa17255a4b8aa07`.
- Rebuild its native gRPC extension from that same SGLang checkout, rather than
  assuming the base image's Rust extension implements the matching protocol:
  `SGLANG_BUILD_RUST_EXTS=grpc pip install --no-deps -e /sgl-workspace/sglang/python`.
  This requires Rust/Cargo, protoc, and the build dependencies from that checkout.
- At Dynamo commit `269939c6cc6eab0de90c35361ffa1e8bab73968c`, build/install
  `lib/bindings/python` and the root Python package into the Miles environment.
  Follow that checkout's build prerequisites; install **without the `[sglang]`
  extra**, which would replace SGLang-miles with Dynamo's published SGLang pin.
  Build `cargo build --release --locked -p dynamo-sglang-sidecar` and put the
  resulting executable on the sidecar container's PATH.
- Install this Miles checkout with `pip install -e '.[dynamo]'`. Verify
  `python -m dynamo.frontend --help`, `dynamo-sglang-sidecar --help`, and the
  native `--grpc-port` SGLang option inside the final environments.

No `python -m dynamo.sglang`, embedded `--sidecar`, or fork-only admission API is used.
The complete image build is not part of the CPU-tested acceptance claim.

## 4 trainer GPUs + two TP2 engines

Use separate trainer, engine and sidecar **PID namespaces/containers**, with a
private shared network. On a single host, the example uses host networking and
loopback addresses. Across hosts replace every address with a routable IP, and
use an identical mounted model path. Trainer Ray sees only GPUs 0–3; engine
containers see GPUs 4–5 and 6–7 respectively. Never put external engine processes
in the trainer container: the standard Miles launcher cleans previous processes.

Provide private etcd on port 2379. Edit `dynamo.json`: unique namespace per run,
model path, addresses, and any `frontend.extra_args/env` or `sidecar.extra_args/env`.
Edit each engine JSON with native SGLang overrides. KV event ports 5557 and 5558
must be reachable; the example publishes events with page size 16 for KV routing.

Run each command in its respective container (GPU selection happens at container
creation; do not select physical GPU IDs again inside a remapped container):

```bash
# Engine 0 / engine 1, each with its own two visible GPUs
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json engine --name engine-0 --overrides examples/dynamo_sidecar/engine-0.json
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json engine --name engine-1 --overrides examples/dynamo_sidecar/engine-1.json

# Independent CPU sidecar processes; each connects to one engine's native gRPC
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json sidecar --name engine-0
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json sidecar --name engine-1

# Read-only engine preflight, after startup
python -m miles.backends.dynamo_utils.deployment --config examples/dynamo_sidecar/dynamo.json check

# Trainer container: converted Qwen3-4B weights and DAPO data already mounted
python scripts/run_qwen3_4b_dynamo.py --dynamo-config examples/dynamo_sidecar/dynamo.json --num-rollout 2
```

`engine` and `sidecar` support `--dry-run` (argv only, no environment secrets).
Miles launches the frontend/router itself as a CPU command worker. Sidecars own
discovery; Miles controls weights directly through existing SGLang HTTP clients.
The gate permits the first rollout only after initial weight publication, and
checks both engine incarnations and versions before/after each complete batch.

`exclusive_run=true` is an operator assertion, **not a network ACL**. Prevent other
clients from generating or updating weights, and keep unrelated workers out of
the namespace. A failed/uncertain request or engine restart terminates this run;
restart the complete run after confirming the old fleet is drained. Dispose does
not kill externally owned engines/sidecars or withdraw their discovery records.

## Acceptance

The recipe runs two GRPO batches and writes `rollout_0.pt` / `rollout_1.pt` under
`--output-dir`. Check that both train steps finish, each group has four responses,
response tokens match the logprob count, and every response carries the trainer's
published version. Confirm the second weight update reaches **both** engines.
Then repeat with a killed engine/sidecar or stale engine version: no affected
batch may enter training and no subsequent rollout may be dispatched.

For transport-only diagnosis on a **separate, idle fleet**, `deployment smoke`
accepts `--frontend-url`, `--weight-version`, `--input-ids '[1,2]'`, and
`--n-samples 4`. Supply valid model token IDs and an already-published known
version; the command does not initialize weights or start a frontend. It checks
SSE completion, token/logprob/version mapping and before/after engine identity,
not rewards, training, or whether every router replica served a request. Never
run it concurrently with Miles training; it would bypass the controller gate.
