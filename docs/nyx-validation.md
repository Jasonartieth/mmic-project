# Nyx render validation

Verified on `A843-2x4090` (`15003-2x-4090`) on 2026-10-22 with an NVIDIA
GeForce RTX 4090 and driver 580.178.04.

The CUDA extra installs upstream Genesis 1.4.4, Nyx plugin 0.1.10, Nyx runtime
0.1.8, and PyTorch 2.14.1+cu130. Genesis is the PyPI release, with no local fork.

```sh
cd ~/diskstation/mmic-project
uv sync --locked --extra cuda

CUDA_VISIBLE_DEVICES=1 \
TMPDIR="$HOME/diskstation/uv-cache/tmp" \
XDG_CACHE_HOME="$PWD/outputs/cache" \
uv run --locked --extra cuda python src/scripts/smoke/nyx_smoke.py --output outputs/nyx-smoke-gpu1
```

The command completed with exit code 0 and no Nyx denoiser or cleanup warnings.
It rendered a falling box with FastPathTracer and read two CUDA RGB frames,
after physics steps 1 and 61. The image shape was `(1, 240, 320, 3)` with dtype
`uint8`, on logical `cuda:0` (physical GPU 1 under the visibility mask).

| Check | Result |
| --- | --- |
| Final box center height | 0.079892233 m; expected 0.08 m |
| Settled image mean pixel value | 92.9486 / 255 |
| Settled image pixel standard deviation | 31.5311 |
| Pixels changing by more than 8 in any channel | 4.3294% |
| Overall validation | Passed |

`initial.png`, `settled.png`, and `result.json` are saved under
`outputs/nyx-smoke-gpu1/`. The PNGs show the box above the plane and then resting
on it.

Nyx selects its Vulkan rendering device independently of CUDA's visibility mask.
On this host it selected physical GPU 1. Exposing only physical GPU 0 caused
device-pairing failure. Leaving both GPUs visible rendered successfully but
produced denoiser and cleanup warnings; exposing physical GPU 1 aligned the
backends and removed those warnings.

The smoke check uses the pinned plugin's camera light dictionaries. The plugin
converts these into native light assets and handles the coordinate transform.
