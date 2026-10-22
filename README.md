# mmic-project

Genesis simulation and sim-to-real robotics experiments.

The Python 3.12 dependency setup follows `../srt2/pyproject.toml`: PyTorch,
Quadrants 1.3.3, TorchRL 0.14, TensorBoard, and mutually exclusive GPU extras.
Genesis comes from the upstream `genesis-world` PyPI release, without a local
fork or submodule. See the [Genesis installation guide](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/installation.html).

## Setup

Install [uv](https://docs.astral.sh/uv/), then select the extra for the machine:

```sh
# AMD / Linux (ROCm 7.2.4 wheels, as in SRT2)
uv sync --extra rocm

# NVIDIA (CUDA 13.0 wheels, as in SRT2)
uv sync --extra cuda

# NVIDIA with Nyx path tracing
uv sync --extra cuda --extra nyx

# Apple Silicon (PyTorch MPS / Genesis Metal)
uv sync --extra metal
```

Use one extra at a time and keep the same extra on subsequent `uv run` commands.
GPU drivers must support the selected wheel build. `uv sync` without an extra
uses the default PyPI PyTorch build; this may include CUDA dependencies on Linux.
`uv.lock` is committed so machines share the same resolution. Add `--locked` to
`uv sync` and `uv run` to require that resolution without updating it.

## Smoke check

```sh
# Headless CPU physics check using the installed ROCm environment
uv run --extra rocm mmic-smoke

# Check the machine's GPU backend
uv run --extra rocm mmic-smoke --backend gpu
```

Substitute `cuda` or `metal` when using those environments. The check drops a
small box onto a plane for 100 steps and verifies its resting height. The first
run compiles simulation kernels. It does not load robot assets or control hardware.

## Nyx rendering

Nyx is an optional NVIDIA renderer, installed alongside the CUDA extra:

```sh
uv sync --extra cuda --extra nyx
uv run --extra cuda --extra nyx mmic-nyx-smoke
```

The headless check renders a falling red box with Nyx's FastPathTracer, verifies
CUDA RGB tensors, image brightness/contrast, frame changes, and final contact
height, and writes `initial.png`, `settled.png`, and `result.json` under
`outputs/nyx-smoke/`. It uses procedural geometry and a directional light, so no
external assets are needed. Nyx requires an NVIDIA GPU and driver 575 or newer;
see the [Nyx installation guide](https://genesis-embodied-ai.github.io/genesis-nyx/latest/user_guide/installation.html).
The Nyx extra conflicts with ROCm and Metal.

## Layout

- `src/mmic_project/`: project Python package; simulation and sim-to-real code goes here.
- `pyproject.toml`: dependencies, GPU indexes, and the `mmic-smoke` command.
- `outputs/`, `runs/`, `run-logs/`, `checkpoints/`: ignored experiment artifacts.

Robot tasks, hardware adapters, and training algorithms can be added as the first
experiment is defined.
