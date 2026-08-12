#!/usr/bin/env bash
# Everything the image bakes, in ONE layer. Kept out of the Dockerfile so it stays readable, and
# because it is the ONLY definition of this environment: the consumer's from-scratch install path
# was deleted with issue #67, deliberately, so there is nothing to diff against and nothing to keep
# in step. What this file installs is what a pod gets.
set -euxo pipefail
export DEBIAN_FRONTEND=noninteractive

# uv writes to the cache before linking into the venv. The base image points UV_CACHE_DIR (and
# PIP_CACHE_DIR, HF_HOME) at /workspace — which is a REAL directory at build time and a mounted pod
# volume at run time, so anything written there is baked into the image and then immediately
# shadowed, invisible to every pod. Disable the cache outright: this is a single-shot build, there
# is no second resolve for it to accelerate.
export UV_NO_CACHE=1
export UV_LINK_MODE=copy
# uv does NOT byte-compile by default, unlike pip. Without this, the ~200 MB of torch/transformers/
# unsloth source is compiled on FIRST IMPORT — i.e. on a billed GPU, inside the run's deps_import
# stage, every single run. Paying it once here is the point of the exercise.
export UV_COMPILE_BYTECODE=1

VENV=/opt/venv
PY="$VENV/bin/python"

# Build tools stay in the image even though llama.cpp is prebaked below: the source-build fallback
# has to keep working if a future Unsloth stops shipping a prebuilt bundle, and having them here
# removes an apt-mirror outage from the failure surface of every training run. jq is here for a
# different reason — it is what reads the package lists out of pins.json below.
apt-get update -qq
apt-get install -y -qq build-essential cmake git curl jq
rm -rf /var/lib/apt/lists/*

# The RunPod base images ship uv at /bin/uv; the installer is the fallback if that ever changes.
# (Ubuntu 24.04's system python is PEP-668 "externally managed", so `pip install uv` is blocked.)
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi

uv venv "$VENV" --python "${PYTHON_VERSION}"

# The package set is READ FROM pins.json, not spelled here. That array rides into the fingerprint
# the consumer compares against, so it has always LOOKED like a contract — but the list used to be
# hardcoded below, which made it inert: adding `unsloth==X` to pins.json would have changed the tag
# hash and forced a rebuild while the image kept installing an unpinned unsloth, i.e. a fingerprint
# asserting a pin that nothing enforced. Now the array is the thing that installs.
# Today it is `unsloth` (which pulls unsloth_zoo, transformers, peft, trl, hf_hub transitively) and
# `flash-linear-attention` — the fast linear-attention kernels for Qwen3.5's GatedDeltaNet layers.
mapfile -t PACKAGES < <(jq -r '.packages[]' /tmp/pins.json)
[ "${#PACKAGES[@]}" -gt 0 ] || { echo "ERROR: pins.json declares no packages"; exit 1; }

# ONE resolve for torch and every package, never a sequence of them. flash-linear-attention used to
# be a second `uv pip install` on its own line, which gave it its own resolve and with it its own
# chance to move torch out from under the ABI-pinned wheels below; resolving jointly turns that into
# a build-time conflict instead of a silent runtime downgrade. The drift guard further down still
# checks after the fact — this just stops it being the only thing that would notice.
# Nothing here is best-effort: `set -e` means a failure fails the BUILD. That is deliberate for
# flash-linear-attention in particular, which was best-effort back when it ran on a billed pod
# (where a transient failure must not waste the rental). In CI there is no such pressure, and an
# image quietly lacking the kernel would slow down every run made with it.
# --torch-backend is EXPLICIT, never `auto`: `auto` detects CUDA from the *driver*, and a Docker
# build has no GPU, so `auto` here silently resolves a CPU-only torch that then fails on the pod at
# the worst possible moment.
uv pip install --python "$PY" "${PACKAGES[@]}" "torch==${TORCH_SPEC}" --torch-backend="${TORCH_BACKEND}"

# Both accelerator wheels are prebuilt for one exact torch+CUDA ABI, and their URLs are VERSION-
# pinned, so they stay downloadable long after they stop matching — a stale pin installs silently
# and then goes dark at import (Unsloth's banner flips to "Your Flash Attention 2 installation
# seems to be broken. Using Xformers instead"). Assert the ABI the URL encodes against the torch
# that actually resolved above, and fail the BUILD rather than ship an incoherent image.
GOT=$("$PY" -c "import torch;print('cu'+(torch.version.cuda or '0').split('.')[0]+'torch'+'.'.join(torch.__version__.split('.')[:2]))")
for WHEEL in "${CAUSAL_CONV1D_WHEEL}" "${FLASH_ATTN_WHEEL}"; do
  WANT=$(echo "$WHEEL" | grep -oE 'cu[0-9]+torch[0-9.]+')
  if [ "$WANT" != "$GOT" ]; then
    echo "ERROR: wheel targets $WANT but the resolved runtime is $GOT — refresh the pin in pins.json"
    echo "       wheel: $WHEEL"
    exit 1
  fi
  # --only-binary :all: guarantees this never falls back to a source build (flash-attn from sdist
  # is a 1–3 h CUDA compile).
  uv pip install --python "$PY" --only-binary :all: "$WHEEL"
done

# What Unsloth's GGUF converter shells out to install at export time if it isn't already present.
# Read from pins.json for the same reason `packages` is, and with the same consequence: a name added
# to `converter_deps` now changes the tag hash AND reaches the install, instead of only the former.
mapfile -t CONVERTER_DEPS < <(jq -r '.converter_deps[]' /tmp/pins.json)
[ "${#CONVERTER_DEPS[@]}" -gt 0 ] || { echo "ERROR: pins.json declares no converter_deps"; exit 1; }
uv pip install --python "$PY" "${CONVERTER_DEPS[@]}"

# llama.cpp, installed through Unsloth's OWN installer rather than by hand, because Unsloth decides
# at export time whether an existing install counts — and its test is specific (a working
# llama-quantize + a converter script at the folder root, plus either a src/ggml/common checkout or
# an UNSLOTH_PREBUILT_INFO.json marker). Anything hand-rolled risks being re-downloaded anyway.
# gpu_support=False matches what save_pretrained_gguf itself passes; llama-quantize is CPU-only, so
# the BUILD needs no GPU. Importing unsloth_zoo does, though — twice over. Its __init__ evaluates
# device_type.get_device_type() at module scope ("Unsloth cannot find any torch accelerator? You
# need a GPU.") and then refuses to be imported outside of unsloth ("Please install Unsloth via
# `pip install unsloth`!", an UNSLOTH_IS_PRESENT check). Both live under one `if not _SKIP_GPU_INIT`
# block, and UNSLOTH_ZOO_DISABLE_GPU_INIT=1 is upstream's own opt-in for precisely this shape of
# caller — a "download-only helper child" that skips the heavy torch/transformers/device init it
# never uses. UNSLOTH_ALLOW_CPU=1 rides along to cover any device probe reached lazily rather than
# at import. Both are scoped to these two commands, never exported: globally set, a later step could
# quietly believe it had a GPU.
# This also hydrates the bundled converter, which stops export fetching convert_hf_to_gguf.py.
ZOO_ENV=(UNSLOTH_ZOO_DISABLE_GPU_INIT=1 UNSLOTH_ALLOW_CPU=1)
env "${ZOO_ENV[@]}" "$PY" -c "from unsloth_zoo.llama_cpp import install_llama_cpp; install_llama_cpp(gpu_support=False, print_output=True)"
# Fail the build if Unsloth can't detect what Unsloth just installed — otherwise the prebake is
# dead weight that every pod silently redoes, and nobody finds out.
env "${ZOO_ENV[@]}" "$PY" -c "from unsloth_zoo.llama_cpp import check_llama_cpp; print('llama.cpp detected:', check_llama_cpp())"

# The fingerprint. `pins` is the CONTRACT: the consumer compares it byte-for-byte against its own
# pins.json and refuses to train on a mismatch, which is what makes "the image silently trains on
# an old unsloth" impossible rather than merely unlikely. `resolved` is EVIDENCE — the versions that
# actually landed — printed into the run log so a drift regression is diagnosable after the fact.
IMAGE_REF="${IMAGE_REF}" RUN_URL="${RUN_URL}" BUILT_AT="${BUILT_AT}" "$PY" - <<'PYEOF'
import json, os
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import torch


def ver(name):
    try:
        return version(name)
    except PackageNotFoundError:
        return None


llama_info = Path("/root/.unsloth/llama.cpp/UNSLOTH_PREBUILT_INFO.json")
llama_tag = None
if llama_info.is_file():
    llama_tag = json.loads(llama_info.read_text()).get("tag")

marker = {
    "pins": json.loads(Path("/tmp/pins.json").read_text(encoding="utf-8")),
    "built_at": os.environ["BUILT_AT"],
    "image": os.environ["IMAGE_REF"],
    "run_url": os.environ["RUN_URL"],
    "resolved": {
        "python": ".".join(map(str, __import__("sys").version_info[:3])),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "unsloth": ver("unsloth"),
        "unsloth_zoo": ver("unsloth_zoo"),
        "flash_attn": ver("flash_attn"),
        "causal_conv1d": ver("causal_conv1d"),
        "flash_linear_attention": ver("flash-linear-attention"),
        "llama_cpp_tag": llama_tag,
    },
}
Path("/opt/venv/.venv-pins.json").write_text(json.dumps(marker, indent=2), encoding="utf-8")
print(json.dumps(marker, indent=2))
PYEOF

# Anything under /workspace is shadowed by the pod volume at run time; anything under /root/.cache
# is dead weight. Neither can be reclaimed by a later layer, hence same-layer cleanup.
rm -rf /root/.cache/* /workspace/.cache /workspace/.venv 2>/dev/null || true
