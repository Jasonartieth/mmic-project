# Native depth sensor validation

Validated `gs.sensors.DepthCamera` on `A843-2x4090` (`15003-2x-4090`) on
2026-10-23 using upstream Genesis 1.4.4, Quadrants 1.3.3, and PyTorch
2.14.1+cu130. Physical GPU 1 was exposed as logical `cuda:0`.

The sensor uses Genesis's batched geometric raycaster. It is independent of
Nyx RGB rendering, and follows the public
[raycaster sensor API](https://genesis-world.readthedocs.io/en/latest/user_guide/sensing/raycaster.html).
This test validates ideal geometry sensing; it does not model hardware depth
noise, material-dependent failures, or RGB/depth registration.

```sh
cd ~/diskstation/mmic-project
CUDA_VISIBLE_DEVICES=1 \
TMPDIR="$HOME/diskstation/uv-cache/tmp" \
XDG_CACHE_HOME="$PWD/outputs/cache" \
MPLCONFIGDIR="$PWD/outputs/cache/matplotlib" \
uv run --locked --extra cuda python -u -m mmic_project.depth_smoke
```

The world-fixed sensor points along +X from `(0, 0, 1)` m, with a 90-degree
horizontal FOV, 320×240 pixels, and 6 m maximum range. Four environments place
a foreground box at different distances, with a larger background box behind
it. The foreground then moves 0.5 m farther away. Independent NumPy pinhole
rays and analytic box intersections provide the expected distance at every
pixel; no-hit pixels are checked separately.

| Check | Result |
| --- | --- |
| Native image tensor | `(4, 240, 320)`, CUDA `float32` |
| Images checked | 8: four environments at two time steps |
| Maximum hit-distance error | 0.0000012265 m (about 1.23 µm) |
| Acceptance tolerance | 0.0002 m |
| Initial center ranges | 1.9000186, 2.1500211, 2.4000235, 2.6500261 m |
| Center range changes | 0.5000048–0.5000050 m, matching the pinhole geometry |
| No-hit pixels | 57,036 per image, all exactly 6 m |
| Depth-only mode | `read().points is None` |
| Validation | All eight images and the motion check passed |

Native `read_image()` values are **along-ray ranges**, rather than optical-axis
depth. Multiplying valid ranges by each ray's +X component gives camera-axis
depth; the diagnostic saves that derived array separately. The central pixel
is slightly off-axis at this even image resolution, explaining the small
difference between the 0.5 m translation and its along-ray range change.

Artifacts are stored under `outputs/depth-camera/` on the server and copied to
the same directory locally:

- `depth.npz`: native `range_m` (float32), derived `axial_depth_m`, expected
  ranges, ray directions, and frame labels. Frame and environment are the first
  two dimensions. Native misses retain 6 m; derived axial misses are NaN.
- `depth-preview.png`: environment 0 before and after movement, with absolute
  error against geometry. Gray pixels are misses.
- `result.json`: per-image checks, camera calibration, scene dimensions,
  package versions, and artifact SHA-256 hashes.

The first compilation emitted upstream Quadrants BVH integer-precision warnings
and an AST deprecation warning. The measured images passed the numerical checks.
