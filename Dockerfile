# syntax=docker/dockerfile:1
#
# A RunPod pod image with Unsloth + the pinned FlashAttention-2 / causal-conv1d wheels already
# installed, so a rented GPU never spends ~9 min rebuilding the same environment before it can
# start training.
#
# Derived from the stock RunPod PyTorch image ON PURPOSE, not from something slimmer: RunPod
# machines cache images per machine, and this is the image their PyTorch template pulls, so a
# warm machine only has to fetch the layers this file adds. That property is fragile — see the
# `provenance: false` / `sbom: false` note in .github/workflows/build.yml — and it is the whole
# reason a bigger base beats a smaller one here.
ARG BASE_IMAGE=runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404
FROM ${BASE_IMAGE}

# The pin set. Defaults live in pins.json and are passed in by the workflow; overriding them on
# the command line is how you build a one-off against a different torch.
ARG PYTHON_VERSION
ARG TORCH_SPEC
ARG TORCH_BACKEND
ARG CAUSAL_CONV1D_WHEEL
ARG FLASH_ATTN_WHEEL
# Provenance, recorded in the fingerprint so a pod can say what it is running.
ARG IMAGE_REF=unknown
ARG RUN_URL=unknown
ARG BUILT_AT=unknown

# DO NOT add ENTRYPOINT, CMD, USER or WORKDIR here, and DO NOT put /opt/venv/bin on PATH.
# The base image's `ENTRYPOINT ["/opt/nvidia/nvidia_entrypoint.sh"]` + `CMD ["/start.sh"]` are what
# bring up sshd and turn the pod's PUBLIC_KEY env var into an authorized_keys entry. Override
# either and the pod still boots, still bills, and simply never accepts SSH — which surfaces
# minutes later as an orchestrator timeout that points nowhere near this file. The venv is always
# addressed by absolute path (/opt/venv/bin/python) for the same reason.
COPY pins.json /tmp/pins.json
COPY build_env.sh /tmp/build_env.sh
RUN bash /tmp/build_env.sh && rm -f /tmp/build_env.sh /tmp/pins.json

LABEL org.opencontainers.image.source="https://github.com/olsh/unsloth-runpod" \
      org.opencontainers.image.description="RunPod pod image with Unsloth, pinned FA2 + causal-conv1d wheels and a prebuilt llama.cpp" \
      org.opencontainers.image.licenses="MIT"
