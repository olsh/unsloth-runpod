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
#
# cu1300, because the venv is cu130 (torch 2.14 + Unsloth's cu13 FA2/causal-conv1d wheels, all built
# against CUDA 13.0). This tag's one load-bearing property is its `NVIDIA_REQUIRE_CUDA=cuda>=13.0`,
# which the container runtime enforces at start: a 12.x host refuses the container instead of
# booting it and failing at the first CUDA call. Do NOT move to a cu1320 base for "newer": the venv
# brings its own CUDA runtime via pip, so a 13.2 base buys nothing and its `cuda>=13.2` requirement
# would refuse every 13.0/13.1 host (on 2026-10-07 that was the 3090, A6000, A40, L40S and 5090).
# 1.4.0 is the newest stable runpod/pytorch line (1.4.1 is still an rc). The baked torch (2.13.0)
# is discarded by build_env.sh either way; what this tag actually decides is the CUDA requirement,
# Ubuntu/glibc, and the base's uv + sshd setup (same entrypoint and /start.sh as 1.1.0, checked).
# Ubuntu 24.04 is the newest runpod/pytorch offers — there is no ubuntu2604 tag.
#
# NOTE this is NOT part of the pin set: BASE_IMAGE never enters pins.json, so it is absent from the
# fingerprint and a base change is invisible to the consumer's verify_baked_env(). The image TAG's
# date is the only evidence of which base an image carries. The workflow's DEFAULT_BASE is what a
# bare dispatch uses, so this default and that one must move together.
ARG BASE_IMAGE=runpod/pytorch:1.4.0-cu1300-torch2130-ubuntu2404
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
