# syntax=docker/dockerfile:1.7

# Build with the normal Miles image as the base and a wheel directory produced
# by tools/dynamo/build_wheels.py as the named `dynamo_wheels` context.
ARG MILES_IMAGE=radixark/miles:dev
FROM ${MILES_IMAGE}

ARG DYNAMO_COMMIT
ARG SGLANG_REPO=https://github.com/sgl-project/sglang.git
ARG SGLANG_BRANCH=sglang-miles
ARG SGLANG_COMMIT
ARG SGLANG_SOURCE_DIR=/sgl-workspace/sglang

# Dynamo declares a stock SGLang dependency. Installing both wheels with
# --no-deps is intentional: the final SGLang installation below is the only
# SGLang implementation allowed in this image.
COPY --from=dynamo_wheels / /tmp/dynamo-wheelhouse/
RUN test -n "${DYNAMO_COMMIT}" \
 && test -f /tmp/dynamo-wheelhouse/dynamo-build-manifest.json \
 && python -m pip install --no-deps \
      /tmp/dynamo-wheelhouse/ai_dynamo_runtime-*.whl \
      /tmp/dynamo-wheelhouse/ai_dynamo-*.whl

# A movable branch name is not a reproducible build. Require a full commit and
# prove that commit belongs to sglang-miles before installing it last.
RUN test "${SGLANG_BRANCH}" = "sglang-miles" \
 && test -n "${SGLANG_COMMIT}" \
 && git -C "${SGLANG_SOURCE_DIR}" remote set-url origin "${SGLANG_REPO}" \
 && git -C "${SGLANG_SOURCE_DIR}" fetch origin \
      "${SGLANG_BRANCH}:refs/remotes/origin/${SGLANG_BRANCH}" \
 && git -C "${SGLANG_SOURCE_DIR}" fetch origin "${SGLANG_COMMIT}" \
 && git -C "${SGLANG_SOURCE_DIR}" merge-base --is-ancestor \
      "${SGLANG_COMMIT}" "refs/remotes/origin/${SGLANG_BRANCH}" \
 && git -C "${SGLANG_SOURCE_DIR}" checkout --detach "${SGLANG_COMMIT}" \
 && python -m pip install --no-deps -e "${SGLANG_SOURCE_DIR}/python[all]"

COPY tools/dynamo /opt/miles-dynamo-contract/tools/dynamo
ENV PYTHONPATH=/opt/miles-dynamo-contract

# This checks object-level contracts, not version strings alone. A stock
# SGLang wheel fails here because it lacks Miles' weight-update session API.
RUN python -m tools.dynamo.check_contract \
      --wheel-manifest /tmp/dynamo-wheelhouse/dynamo-build-manifest.json \
      --expected-dynamo-commit "${DYNAMO_COMMIT}" \
      --sglang-source-dir "${SGLANG_SOURCE_DIR}" \
      --expected-sglang-commit "${SGLANG_COMMIT}" \
 && rm -rf /tmp/dynamo-wheelhouse

LABEL org.opencontainers.image.miles.dynamo.commit="${DYNAMO_COMMIT}" \
      org.opencontainers.image.miles.sglang.branch="${SGLANG_BRANCH}" \
      org.opencontainers.image.miles.sglang.commit="${SGLANG_COMMIT}"

ENV MILES_DYNAMO_COMMIT=${DYNAMO_COMMIT} \
    MILES_SGLANG_BRANCH=${SGLANG_BRANCH} \
    MILES_SGLANG_COMMIT=${SGLANG_COMMIT}
