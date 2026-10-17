# Miles-managed Dynamo sidecar

Draft: CPU-tested integration, not a measured GPU parity result.

Build SGLang from `sgl-project/sglang` **sglang-miles**, at the revision in
`dynamo.json`, with its native gRPC extension (`SGLANG_BUILD_RUST_EXTS=grpc`).
Build Dynamo's Python bindings and root package from `AndyDai-nv/dynamo` at
the manifest revision: this includes the still-unmerged admission PRs and
authenticated native HTTP forwarding. Do not install Dynamo's `[sglang]`
extra over the SGLang-miles build. Pins record required builds; Miles does
not verify the installed binaries' Git revisions.

Provide converted Qwen3-4B weights, DAPO data, etcd and NATS. Set a unique
namespace and reachable service addresses in `dynamo.json`. Use a trusted
private network: the sidecar's admission listener is not authenticated by
`--sglang-api-key` (that key protects engine HTTP).

```bash
python scripts/run_qwen3_4b_dynamo.py --router dynamo --output-dir /data/parity/dynamo
python scripts/run_qwen3_4b_dynamo.py --router cache_aware --output-dir /data/parity/cache-aware
```

Run sequentially on the same eight GPUs with identical checkpoints, data and
seed. Miles allocates four training GPUs and two TP2 engines, starts the frontend,
and launches each sidecar via SGLang `--sidecar dynamo.sglang.sidecar`.
No external engine list, standalone sidecar process or special rollout function
is needed. Add `--extra-args '--use-fault-tolerance'` for a separate recovery run.
Compare end-to-end training wall-clock and per-rollout rewards from both runs;
report startup separately and record the exact image/commit/settings. No parity
numbers are available for this revision yet.

Frontend/sidecar native options go in their `extra_args` or `env`; namespace,
transport, authentication and allocated listeners remain Miles-owned. The first
revision supports one aggregated model with TP engines and SGLang DP=1; snapshot
evaluation, batched prefill scoring, fully-async colocation and SMG routing-key
policies are rejected.

Admission withdraws discovery membership before the native weight-update session
and verifies the trainer version before readmission. It is not a router-cache
convergence barrier or per-request execution fence: native pause/update/resume
still owns execution safety. Lost sidecar grants feed the native FT health checker.
