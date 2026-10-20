# Building the Miles and Dynamo sidecar experiment environment

This guide describes how to build the environment used for the Qwen3-4B GSM8K
comparison between Dynamo's KV router and Miles' default `cache_aware` router.
Both runs used the same software, with four training GPUs and two TP2 inference
engines on one eight-A100-80GB node.

Start from the public Miles image below. No private registry, cluster account,
or internal build artifact is required. The commands are intended to be adapted
to your own image build; they are not a Dockerfile.

The October 19, 2026 experiment used an intermediate image plus runtime package
updates. The sequence below combines those steps into a build from the public
base. The resulting environment should be checked before training: this direct
public-base rebuild has not itself been run end to end.

## Source revisions

Use these revisions, not the current tips of the branches.

| Component | Source | Commit |
| --- | --- | --- |
| Miles | [AndyDai-nv/miles](https://github.com/AndyDai-nv/miles/commit/a81072ab106d9fd5dd35666a70d6ca194ce77de6) | `a81072ab106d9fd5dd35666a70d6ca194ce77de6` |
| SGLang | [AndyDai-nv/sglang](https://github.com/AndyDai-nv/sglang/commit/318ffdf2e65fbbbc55d30b78be63fb6347f6bc4c) | `318ffdf2e65fbbbc55d30b78be63fb6347f6bc4c` |
| Dynamo | [AndyDai-nv/dynamo](https://github.com/AndyDai-nv/dynamo/commit/57086811d93986509eda5af37d23cc63e3126aae) | `57086811d93986509eda5af37d23cc63e3126aae` |
| Megatron-LM | Inherited from the base at `/root/Megatron-LM` | `fd15ee20a4f03b03680529baf4b8f7eeed64df1d` |

The SGLang revision is based on **sglang-miles**, with native gRPC authentication
changes. It is not stock SGLang main. The Dynamo revision includes the sidecar
admission and authenticated-forwarding changes used by this integration.
Miles also records these two dependency revisions in
[`examples/dynamo_sidecar/dynamo.json`](https://github.com/AndyDai-nv/miles/blob/a81072ab106d9fd5dd35666a70d6ca194ce77de6/examples/dynamo_sidecar/dynamo.json).

Build all three Dynamo pieces from the same commit: the sidecar executable,
the Python runtime bindings, and the Python package containing the frontend.
Miles launches SGLang natively and attaches `dynamo.sglang.sidecar`; it does
**not** run a `python -m dynamo.sglang` worker.

## 1. Start from the public Miles base

Use this Docker Hub image, selecting `linux/amd64`:

```text
docker.io/radixark/miles@sha256:ca11221f8226700b309c11e296876adf8a78be13419555dd4ab149ce27fa906a
```

That is the multi-platform image digest. Its AMD64 manifest is
`sha256:688efeaab16c0e39f56d9a3de5473ec1d68e2a28d7a920b94c278c1196cd4ecf`.
Do not substitute a moving `latest` or `dev` tag: the training extensions and
Megatron checkout are part of the starting environment.

For an interactive build session:

```bash
docker run --name miles-sidecar-build --platform linux/amd64 -it \
  --entrypoint bash \
  docker.io/radixark/miles@sha256:ca11221f8226700b309c11e296876adf8a78be13419555dd4ab149ce27fa906a
```

Run the remaining build commands **inside the container as root**, using its
existing Python environment. Keep the `/work` source directories in the final
image because Miles and SGLang will be installed in editable mode. You do not
need GPUs until validation. The wheel set below is for Python 3.12 and x86-64,
not ARM64.

```bash
python --version
git -C /root/Megatron-LM rev-parse HEAD
```

Expect Python 3.12 and the Megatron revision in the table above.

## 2. Install the build tools

The intermediate experiment image added Clang and the Rust toolchain. Include
these steps when starting from the public base:

```bash
apt-get update
apt-get install -y --no-install-recommends \
  build-essential clang cmake libclang-dev pkg-config libssl-dev \
  protobuf-compiler libprotobuf-dev perl patchelf

export PATH=/root/.cargo/bin:$PATH
rustup toolchain install 1.96.1 --profile minimal
export RUSTUP_TOOLCHAIN=1.96.1
export CARGO_BUILD_JOBS=16
export CARGO_PROFILE_RELEASE_LTO=false
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16
export CARGO_TARGET_DIR=/build-cache/dynamo

python -m pip install setuptools-rust==1.13.0 setuptools-scm==10.3.4
```

Lower `CARGO_BUILD_JOBS` if compilation runs out of host memory. Persist these
environment settings across build steps if your builder starts a new shell for
each step.

## 3. Check out the sources

Use fresh directories so that an inherited checkout or local patch cannot
silently become part of the build:

```bash
mkdir -p /work

git init /work/miles
git -C /work/miles fetch --depth=1 https://github.com/AndyDai-nv/miles.git \
  a81072ab106d9fd5dd35666a70d6ca194ce77de6
git -C /work/miles checkout --detach FETCH_HEAD

git init /work/sglang
git -C /work/sglang fetch --depth=1 https://github.com/AndyDai-nv/sglang.git \
  318ffdf2e65fbbbc55d30b78be63fb6347f6bc4c
git -C /work/sglang checkout --detach FETCH_HEAD

git init /work/dynamo
git -C /work/dynamo fetch --depth=1 https://github.com/AndyDai-nv/dynamo.git \
  57086811d93986509eda5af37d23cc63e3126aae
git -C /work/dynamo checkout --detach FETCH_HEAD
```

Set the source paths for both the build and the final runtime environment:

```bash
export PYTHONPATH=/work/miles:/work/sglang/python:/root/Megatron-LM
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

## 4. Align PyTorch and the training wheels

Install PyTorch from the CUDA 13 index before replacing its compiled extensions:

```bash
python -m pip install torch==2.14.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

The experiment also had `torchvision==0.29.1+cu130` and
`torchaudio==2.11.0+cu130`. They are not exercised by this text-only recipe.
If matching the recorded auxiliary packages, install them without allowing
their dependency resolution to replace Torch:

```bash
python -m pip install --no-deps torchvision==0.29.1 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu130
```

Download the twelve wheels listed in the [wheel manifest below](#wheel-manifest)
from the public
[Miles CUDA 13 / Torch 2.14 release](https://github.com/radixark/miles-wheels/releases/tag/cu130-torch214-x86_64).
Put only those files in `/work/miles-wheels`, check their SHA256 hashes, then
install them together:

```bash
mkdir -p /work/miles-wheels
# Download the listed release assets into this directory before continuing.
sha256sum /work/miles-wheels/*.whl
python -m pip install --force-reinstall --no-deps /work/miles-wheels/*.whl
```

This replaces the base image's Torch-dependent extensions, including Apex,
FlashAttention, and Transformer Engine. Keeping the older binaries after
upgrading Torch can produce undefined-symbol errors. Do not substitute generic
PyPI builds for these Miles wheels.

## 5. Build SGLang with both Rust extensions

```bash
SGLANG_BUILD_RUST_EXTS=grpc,mem_cache \
  CARGO_TARGET_DIR=/build-cache/sglang \
  python -m pip install --no-build-isolation -e /work/sglang/python
```

`grpc` builds SGLang's native gRPC server, which the Dynamo sidecar uses.
`mem_cache` builds the Rust TreeCore implementation used by this revision's
default cache. Building only `grpc` is not sufficient for this configuration.

SGLang's installation resolves its Python dependencies. Afterwards, restore
the versions used in the Qwen3 experiment:

```bash
python -m pip install --no-deps sglang-kernel==0.4.9 flashinfer-python==0.7.1
python -m pip install --no-deps flashinfer-cubin==0.7.1 \
  --index-url https://flashinfer.ai/whl
python -m pip install --no-deps flashinfer-jit-cache==0.7.1 \
  --index-url https://flashinfer.ai/whl/cu130
python -m pip install --no-deps transformers==5.12.1 tokenizers==0.22.2
python -m pip uninstall -y flash-attn-4
```

Keep all three FlashInfer packages on the same release. The installed JIT-cache
version will include the CUDA suffix, `0.7.1+cu130`.

The Transformers override is intentional: 5.19 conflicted with Miles' HF aliases
in this Qwen3 setup. Removing `flash-attn-4` is also intentional: the tested
FA4/CuTe combination failed during A100 training. The trainer used
FlashAttention 2.7.4.post1 with Transformer Engine 2.17.0+miles instead. These
choices describe the tested model and hardware, not a general dependency policy
for every SGLang model.

## 6. Build Dynamo and install Miles

Install the runtime packages that were present in the intermediate image,
using the versions recorded after the experiment's updates:

```bash
python -m pip install \
  uvloop==0.23.0 msgspec==0.22.0 redis==8.1.0 zstandard==0.25.0 \
  nvtx==0.2.16 grpcio==1.84.0 protobuf==6.33.6

cd /work/dynamo
cargo build --release --locked -p dynamo-sglang-sidecar
install -m 0755 /build-cache/dynamo/release/dynamo-sglang-sidecar \
  /usr/local/bin/dynamo-sglang-sidecar
python -m pip install --no-deps /work/dynamo/lib/bindings/python
python -m pip install --no-deps /work/dynamo

python -m pip install --no-deps -e /work/miles
python -m pip install wandb==0.30.0
cd /work/miles
```

Do not install Dynamo's `[sglang]` extra: SGLang has already been built from the
required sglang-miles revision. The `--no-deps` installations keep Dynamo and
Miles from replacing the dependency set assembled above.

### W&B shutdown patch

The experiment also applied the following change to
`miles/utils/workers/ray_worker_manager.py`. It is **not in the Miles commit
listed above**. Include it when reproducing the W&B-enabled run: it lets Ray
workers flush tracking before shutdown. It does not change the routing or
training algorithm.

Apply this diff in `/work/miles` before saving your final image:

```diff
diff --git a/miles/utils/workers/ray_worker_manager.py b/miles/utils/workers/ray_worker_manager.py
--- a/miles/utils/workers/ray_worker_manager.py
+++ b/miles/utils/workers/ray_worker_manager.py
@@ -464,6 +464,12 @@ class _CommandActorManager(_BaseActorManager[CommandWorkerSpec]):

 @dataclass
 class _ServeActorRayCommManager(_BaseActorManager[ServeWorkerSpec]):
+    async def _shutdown_gracefully(self) -> None:
+        try:
+            await asyncio.wait_for(self.actor_handle.finish_tracking.remote(), timeout=_SHUTDOWN_TIMEOUT)
+        except Exception as e:
+            logger.warning(f"Tracking flush before shutdown failed ({self=}, {e})")
+
     def _compute_remote_options(self) -> dict:
         groups = self.spec.concurrency_groups
         return {} if groups is None else dict(concurrency_groups=groups)
@@ -548,6 +554,11 @@ def bootstrapped_worker_class(worker_class_path: str) -> type:
         ) -> None:
             super().__init__(**ctor_kwargs(_ctor_context(context)))

+        def finish_tracking(self) -> None:
+            from miles.utils.tracking_utils.tracking import finish_tracking
+
+            finish_tracking()
+
     BootstrappedWorker.__name__ = worker_class.__name__
     BootstrappedWorker.__qualname__ = worker_class.__qualname__
     BootstrappedWorker.__module__ = worker_class.__module__
```

Keep W&B credentials out of the image. Mount a `.netrc` file or use your runtime's
secret mechanism. If a parent process calls `wandb.Api()` for an authentication
check, remove `WANDB_SERVICE` from the environment passed to independent training
processes. Otherwise their writers can attach to the wrong W&B service; the
shutdown patch alone does not prevent this.

## 7. Validate and record the build

These checks require no GPU and should pass before publishing the image:

```bash
python -c 'import dynamo._core; from sglang.srt.rust_extensions import _grpc; from sglang.srt.mem_cache.rust_tree_core.extension import RustUnifiedTreeCoreBinding'
python -m dynamo.frontend --help
dynamo-sglang-sidecar --help
```

Then start the finished image on a GPU host and check the training imports:

```bash
python -c 'import torch, sgl_kernel, transformer_engine.pytorch, flash_attn, apex; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name())'
```

The experiment used driver `615.71.09`. A different host needs a driver capable
of running the CUDA 13 stack. Import checks alone do not validate attention
kernels or rollout. Before the full comparison, exercise a packed BF16
attention forward/backward pass and a short rollout plus weight-update cycle.
The original attention check used 16 query heads, 4 KV heads, head dimension
128, and packed lengths `[64, 128]` and `[504, 1600, 4096]`, checking finite
outputs and gradients.

Save the installed package list, source revisions, local patch, and binary hash
with your build record:

```bash
python -m pip freeze > /work/pip-freeze.txt
git -C /work/miles rev-parse HEAD
git -C /work/sglang rev-parse HEAD
git -C /work/dynamo rev-parse HEAD
git -C /work/miles diff
sha256sum /usr/local/bin/dynamo-sglang-sidecar
```

The expected core package versions are Torch `2.14.1+cu130`, SGLang
`0.0.0.dev1+g318ffdf2e`, SGLang Router `0.3.2`, Ray `2.58.0`, and
`ai-dynamo` / `ai-dynamo-runtime` `1.6.0`, alongside the explicit pins above.
SGLang's Git revision matters more than its development-version label.

The base, source revisions, and wheel contents are pinned, but apt packages and
some transitive Python dependencies are not fully locked. Rebuilding can still
produce differences in those dependencies and in the image output. The
model-specific dependency overrides can also leave `pip check` conflicts; do
not remove the pins merely to silence them. Record any remaining conflicts and
validate the actual training path. Once checked, publish one image and use its
immutable digest for both router runs.

## What belongs outside the image

Model weights, datasets, converted Megatron checkpoints, outputs, and credentials
are runtime inputs. The GSM8K comparison used:

- `Qwen/Qwen3-4B` at `1cfa9a7208912126459214e8b04321603b3df60c`.
- `zhuzilin/gsm8k` at `0cbd9f31d91ac21a7613dcbc7fef992adac459ae`, using
  `train.parquet`, input field `messages`, and target field `label`.
- etcd `3.5.21` and NATS `2.11.10`, running as separate services for Dynamo.

Configure reachable service addresses and a unique Dynamo namespace at launch.
Keep the frontend and sidecar admission listeners on a trusted private network.
The [Miles launcher](https://github.com/AndyDai-nv/miles/blob/a81072ab106d9fd5dd35666a70d6ca194ce77de6/scripts/run_qwen3_4b_dynamo.py)
defaults to DAPO and 20 iterations; the comparison overrides the dataset fields
above and uses `--num-rollout 30`. The comparison ran the routers sequentially,
with four prompts and four responses per prompt per iteration. An image build by itself
does not prepare the converted checkpoint or reproduce those launch settings.

The pinned SGLang revision predates the later admin-key changes. This experiment
did not test admin-key support or fault recovery. Updating the dependencies for
those features requires a new validation run.

## Wheel manifest

These are the twelve files installed from the
[public Miles wheels release](https://github.com/radixark/miles-wheels/releases/tag/cu130-torch214-x86_64).
The hashes match the experiment's installed-package record. Other release assets,
such as source metadata and gateway tarballs, are not needed for this step.

| Filename | SHA256 |
| --- | --- |
| `apex-0.1-cp312-cp312-linux_x86_64.whl` | `a39d91e30aa97d24fc357aa0ee01d5e7065cd2d02daa655fa109e5949e90e855` |
| `causal_conv1d-1.6.1-cp312-cp312-linux_x86_64.whl` | `4bc49e666cdc565e653916fb32ae3c7b367930b305870504ad60c5d423082a24` |
| `fake_int4_quant_cuda-0.0.0-cp312-cp312-linux_x86_64.whl` | `2c64afb60263e848596e8232a41a7ce060f3cdd767ac6ca2576bfb2dbb0df39d` |
| `fast_hadamard_transform-1.1.0-cp312-cp312-linux_x86_64.whl` | `40667deef8a41b507ae357f9f9855fe31338f711549db5d99ab18b0113e5b115` |
| `flash_attn-2.7.4.post1-cp312-cp312-linux_x86_64.whl` | `7b24ef33122b9dbe8167d613f433344dfa2c32956859920429b8ebfd4130910f` |
| `flash_attn_3-3.0.0-cp310-abi3-linux_x86_64.whl` | `6b19fd7c3ced3e70b3a75b6d84d194bb3fcf18fcd609b88cd047d85d467a9498` |
| `flash_mla-1.0.0+b7643bd-cp312-cp312-linux_x86_64.whl` | `745a950c3378d61a83c892d14edf13ec48318c79a5749bd3bf5ac31052b1347a` |
| `mamba_ssm-2.3.1-cp312-cp312-linux_x86_64.whl` | `86cc0901d5d75ed5216b88e05c9d80ad8e308c1c3804de4a189c0e9946b6c556` |
| `sglang_router-0.3.2-cp38-abi3-manylinux_2_39_x86_64.whl` | `f86479783fedf9a3d50e53204be93568fead799c1b9883ee752fc8d5174d4aff` |
| `transformer_engine-2.17.0+miles-py3-none-any.whl` | `468ebc317039ce673d0cace07158e15a33801a64e7a856a4791160e69ab10c7f` |
| `transformer_engine_cu13-2.17.0+miles-py3-none-manylinux_2_28_x86_64.whl` | `43aa1f20da8141bb56c31432334ae0af809bc987c3513ff435eb0de2a43ef2fd` |
| `transformer_engine_torch-2.17.0+miles-cp312-cp312-linux_x86_64.whl` | `3bcd366f254a705c1ddfbff3173bca949eb84506976d7f83dfa9359c90e1bcac` |
