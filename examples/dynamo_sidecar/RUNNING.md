# Miles-managed Dynamo sidecar

Draft: single-node GRPO and recovery tested; authenticated launch remains blocked.

Build SGLang from `sgl-project/sglang` **sglang-miles**, at the revision in
`dynamo.json`, with its native gRPC extension (`SGLANG_BUILD_RUST_EXTS=grpc`).
This SGLang revision requires PyTorch 2.14.1; use matching CUDA 13 training
extensions from Miles' `cu130-torch214-x86_64` wheels release on x86-64.
Match FlashInfer's Python, cubin and JIT-cache versions (0.7.1). For this
Qwen3 recipe, retain Miles' Transformers 5.12.1 and tokenizers 0.22.2:
Transformers 5.19 conflicts with Miles' HF aliases. This overrides SGLang's
declared package pins and is not validation for other model families.
Build Dynamo's Python bindings and root package from `AndyDai-nv/dynamo` at
the manifest revision: this includes the still-unmerged admission PRs and
authenticated native HTTP forwarding. Do not install Dynamo's `[sglang]`
extra over the SGLang-miles build. Pins record required builds; Miles does
not verify the installed binaries' Git revisions.

Provide converted Qwen3-4B weights, DAPO data, etcd and NATS. Set a unique
namespace and reachable service addresses in `dynamo.json`. Use a trusted
private network: the sidecar's admission listener is unauthenticated. Miles
and the sidecar forward `--sglang-api-key` to engine HTTP, but the pinned
SGLang rejects API keys with native gRPC because that listener has no auth.
Authenticated sidecar launch therefore remains blocked on SGLang gRPC auth;
do not bypass its validation or expose these listeners to untrusted clients.

```bash
python scripts/run_qwen3_4b_dynamo.py --router dynamo --output-dir /data/parity/dynamo
python scripts/run_qwen3_4b_dynamo.py --router cache_aware --output-dir /data/parity/cache-aware
```

Run sequentially on the same eight GPUs with identical checkpoints, data and
seed. Miles allocates four training GPUs and two TP2 engines, starts the frontend,
and launches each sidecar via SGLang `--sidecar dynamo.sglang.sidecar`.
No external engine list, standalone sidecar process or special rollout function
is needed. Add `--extra-args '--use-fault-tolerance'` for a separate recovery run.
The [GPU validation](https://github.com/radixark/miles/pull/3890#issuecomment-6228599619)
records tested revisions, wall-clock timing, per-rollout rewards and in-flight
failure recovery. Record startup separately when comparing other configurations.

Frontend/sidecar native options go in their `extra_args` or `env`; namespace,
transport, authentication and allocated listeners remain Miles-owned. The first
revision supports one aggregated model with TP engines and SGLang DP=1; snapshot
evaluation, batched prefill scoring, fully-async colocation and SMG routing-key
policies are rejected.

Admission withdraws discovery membership before the native weight-update session
and verifies the trainer version before readmission. It is not a router-cache
convergence barrier or per-request execution fence: native pause/update/resume
still owns execution safety. Lost sidecar grants feed the native FT health checker.
