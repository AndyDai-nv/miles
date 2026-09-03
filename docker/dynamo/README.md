# Dynamo compatibility image

This image layers Dynamo onto the normal Miles image while preserving one hard
runtime invariant: the installed SGLang must come from a pinned commit on the
`sglang-miles` branch. Dynamo's optional dependency must never replace it with
a stock SGLang wheel.

Build the two Dynamo wheels from a clean, exact checkout:

```bash
python tools/dynamo/build_wheels.py \
  --dynamo-repo /path/to/dynamo \
  --dynamo-commit <full-40-character-sha> \
  --output-dir /tmp/dynamo-wheels
```

Then build the compatibility image:

```bash
docker buildx build \
  --file docker/dynamo/Dockerfile \
  --build-context dynamo_wheels=/tmp/dynamo-wheels \
  --build-arg MILES_IMAGE=radixark/miles:dev \
  --build-arg DYNAMO_COMMIT=<full-40-character-sha> \
  --build-arg SGLANG_BRANCH=sglang-miles \
  --build-arg SGLANG_COMMIT=<full-40-character-sha> \
  --tag miles:dynamo-contract .
```

The build fails when:

- the Dynamo wheel manifest does not match `DYNAMO_COMMIT`;
- `SGLANG_COMMIT` is not an ancestor of `sglang-miles`;
- Dynamo's frontend or worker argument groups do not parse the required flags;
- the installed SGLang lacks Miles' weight-update session methods or request
  types; or
- Dynamo lacks a required control handler or its RL passthrough.

The output image records both commits and the SGLang branch in OCI labels and
environment variables. A branch name alone is deliberately insufficient.

This is a compatibility/build artifact, not the rollout integration itself. It
does not launch etcd, a Dynamo frontend, or a Dynamo SGLang worker, and it does
not modify Miles' existing SGLang execution path.
