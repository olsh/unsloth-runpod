# unsloth-runpod

A [RunPod](https://runpod.io) pod image with [Unsloth](https://github.com/unslothai/unsloth), the
pinned prebuilt **FlashAttention-2** and **causal-conv1d** wheels, and a **prebuilt llama.cpp**
already installed — so a rented GPU starts training instead of spending ~9 minutes rebuilding the
same Python environment from PyPI, every single run.

```
docker.io/olsh/unsloth-runpod:<YYYY-MM-DD>-<pins-hash>
```

Public image. Nothing project-specific is baked in — no training code, no
data, no model weights — so it is usable by anything that wants this stack on a pod.

## What's in it

| | |
|---|---|
| Base | `runpod/pytorch:1.1.0-cu1281-torch280-ubuntu2404` (CUDA 12.8) — a workflow input, **not** part of the pin set, so it is absent from the fingerprint |
| venv | `/opt/venv` — **not** `/workspace`, which the pod volume mounts over |
| Packages | `pins.json`'s `packages` array (`unsloth`, `flash-linear-attention`) resolved in ONE `uv pip install` together with the pinned `torch`, then the two prebuilt CUDA wheels and `converter_deps` |
| llama.cpp | prebuilt at `/root/.unsloth/llama.cpp`, installed via Unsloth's own installer so Unsloth detects it |
| Fingerprint | `/opt/venv/.venv-pins.json` |
| Bytecode | precompiled (`UV_COMPILE_BYTECODE=1`) — `uv` skips this by default, which otherwise costs minutes on first import, on the clock |

Everything is declared in [`pins.json`](pins.json) and installed by
[`build_env.sh`](build_env.sh).

## Using it

Point a pod at the image and use the absolute interpreter path:

```bash
/opt/venv/bin/python your_script.py
```

Do **not** activate it via `PATH` in a start script: the base image's entrypoint (`/start.sh`) is
what brings up sshd and installs the pod's `PUBLIC_KEY`, and it resolves `python` off `PATH`.

### The fingerprint

`/opt/venv/.venv-pins.json` holds two things:

```jsonc
{
  "pins":     { /* verbatim pins.json — the CONTRACT */ },
  "resolved": { /* the versions that actually landed — EVIDENCE */ },
  "built_at": "...", "image": "...", "run_url": "..."
}
```

A consumer that keeps its own copy of the pin set should read `pins` before training and **abort on
any difference**, rather than train in an environment it can't name. That check is the entire
answer to "a stale prebuilt image silently trains on an old unsloth" — without it, the image is a
liability. `resolved` is for diagnosing drift afterwards; log it.

## Building

Rebuilds run on `workflow_dispatch` (inputs override `pins.json`) and weekly, so the floating
`unsloth` / `unsloth_zoo` don't go stale:

```bash
gh workflow run build.yml --repo olsh/unsloth-runpod \
  -f torch_spec='2.10.*' \
  -f flash_attn_wheel='https://github.com/Dao-AILab/flash-attention/releases/download/…'
```

An overridden input is written back into the `pins.json` that gets baked, so the image's
fingerprint always describes the image that was actually built.

Locally:

```bash
docker buildx build . \
  --build-arg PYTHON_VERSION=3.13 \
  --build-arg TORCH_SPEC='2.10.*' \
  --build-arg TORCH_BACKEND=cu128 \
  --build-arg CAUSAL_CONV1D_WHEEL=… \
  --build-arg FLASH_ATTN_WHEEL=… \
  --provenance=false --sbom=false \
  -t docker.io/olsh/unsloth-runpod:dev
```

Tags are immutable (`<date>-<sha8 of pins.json>`); there is deliberately **no `:latest`**, because
a moving tag turns "which environment did that run train in?" into an unanswerable question.

### Why Docker Hub, and why the puller must authenticate

This built to `ghcr.io` until 2026-08-30. It moved because ghcr refuses pulls of an image this size
often enough to break automated deployment, answering `429 Too Many Requests` /
`toomanyrequests: retry-after: …` partway through a transfer. GitHub publishes **no** ghcr.io pull
rate limit, so there is no budget to raise and no documented credential that raises one — and
authenticating does not help: the refusals reproduce for an authenticated puller, including a
GitHub-hosted Actions runner using its own `GITHUB_TOKEN`.

Docker Hub's limits are published, and the decisive column is the **scope**, not the size:

| user type | pull rate limit / 6 h |
|---|---|
| Business / Team / Pro (authenticated) | unlimited |
| **Personal (authenticated)** | **200** |
| unauthenticated | **100 per IPv4 address or IPv6 /64 subnet** |

Anonymous is rationed *per address*, so a pod on a shared datacenter IP is rationed by strangers.
Authenticated, the budget is per namespace and entirely yours — and Docker's own pull-attribution
rules key on the **authenticated identity**, not on repository visibility, so a *public* image
pulled with credentials still draws on the account budget. That is why this repo is public and the
consumer still logs in: **a public image pulled anonymously buys nothing**, because it lands right
back on the shared per-IP pool.

## Gotchas worth knowing before you edit this

- **A pull the pod cannot make does not look like an auth error.** The pod is created, *bills*,
  and simply never accepts SSH — so a consumer's "wait_for_ssh timed out" points nowhere near the
  registry. Check the credential and the image's visibility first when that happens. Keeping the
  Docker Hub repository **public** is deliberate: it is not what makes the image pullable (the
  consumer authenticates either way, see above), it is that a Personal account gets exactly **one**
  private repository, and a public image keeps an anonymous `docker pull` available for smoke tests.
- **`import unsloth_zoo` needs a GPU, twice over.** Its `__init__` evaluates
  `device_type.get_device_type()` at module scope *and* refuses to import outside of `unsloth` (an
  `UNSLOTH_IS_PRESENT` check). Both are under one `if not _SKIP_GPU_INIT` block, so
  `UNSLOTH_ZOO_DISABLE_GPU_INIT=1` — upstream's opt-in for a "download-only helper child" — clears
  them; `UNSLOTH_ALLOW_CPU=1` alone clears only the first. Scoped to the llama.cpp commands, never
  exported.
- **`--torch-backend=auto` cannot work at build time.** It detects CUDA from the driver, and a
  Docker build has no GPU, so it resolves a CPU-only torch that fails much later. The backend is
  always explicit here.
- **The wheels' ABI is asserted at build time.** Both wheel URLs are version-pinned, so they stay
  downloadable long after they stop matching the resolved torch — a stale pin installs *silently*
  and then goes dark at import. `build_env.sh` compares the `cuNNtorchN.M` encoded in each URL
  against the torch that actually resolved, and fails the build.
- **`/workspace` is shadowed at run time.** It is a real directory during the build and a mounted
  volume on the pod, so anything written there during the build is invisible forever. That includes
  `UV_CACHE_DIR` and `HF_HOME`, which the base image points at `/workspace` — hence `UV_NO_CACHE=1`.
- **Baking model weights is out of scope.** RunPod advises against it, and a base model downloads
  in ~1 min against an image that would grow by tens of GB.

## Licence

MIT.
