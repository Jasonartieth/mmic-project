"""Render and validate a moving rigid body with the Nyx CUDA sensor."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import distribution, version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/nyx-smoke"))
    args = parser.parse_args()

    print("Loading Genesis and Nyx...", flush=True)
    import genesis as gs
    import gs_nyx.nyx_py_renderer as renderer
    import torch
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions
    from PIL import Image

    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("Nyx requires an NVIDIA CUDA device")
    if distribution("genesis-world").read_text("direct_url.json"):
        raise RuntimeError("This check requires the upstream Genesis registry release")

    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Initializing CUDA on {torch.cuda.get_device_name()}...", flush=True)
    gs.init(backend=gs.cuda, seed=0, logging_level="warning")
    try:
        world = gs.Scene(sim_options=gs.options.SimOptions(dt=0.01), show_viewer=False)
        world.add_entity(gs.morphs.Plane(plane_size=(5.0, 5.0)))
        box = world.add_entity(
            gs.morphs.Box(pos=(0.0, 0.0, 0.45), size=(0.16, 0.16, 0.16)),
            surface=gs.surfaces.Rough(color=(0.85, 0.12, 0.05, 1.0)),
        )
        camera = world.add_sensor(NyxCameraOptions(
            res=(320, 240),
            pos=(1.1, -1.1, 0.9),
            lookat=(0.0, 0.0, 0.2),
            fov=45.0,
            spp=16,
            render_mode=renderer.ERenderMode.FastPathTracer,
            lights=[{
                "type": "directional",
                "dir": (-0.5, -0.3, -1.0),
                "color": (1.0, 1.0, 1.0),
                "intensity": 5.0,
            }],
        ))
        print("Building the physics scene and Nyx renderer...", flush=True)
        world.build(n_envs=1)
        frames = []
        for step in range(61):
            world.step()
            if step not in (0, 60):
                continue
            pixels = camera.read().rgb
            if pixels.shape != (1, 240, 320, 3) or pixels.dtype != torch.uint8 or not pixels.is_cuda:
                raise RuntimeError(f"Unexpected Nyx output: {pixels.shape}, {pixels.dtype}, {pixels.device}")
            rgb_device = str(pixels.device)
            frames.append(pixels[0].cpu().clone())
            print(f"Rendered Nyx frame after step {step + 1}.", flush=True)
        for label, pixels in zip(("initial", "settled"), frames, strict=True):
            Image.fromarray(pixels.numpy()).save(args.output / f"{label}.png")

        before, after = (pixels.float() for pixels in frames)
        brightness = float(after.mean())
        contrast = float(after.std())
        changed_fraction = float(((after - before).abs().amax(dim=-1) > 8).float().mean())
        height = float(box.get_pos()[0, 2])
        passed = brightness > 1.0 and contrast > 2.0 and changed_fraction > 0.005 and 0.06 < height < 0.10
        result = {
            "status": "passed" if passed else "failed",
            "hostname": platform.node(),
            "gpu": torch.cuda.get_device_name(),
            "versions": {name: version(name) for name in ("genesis-world", "gs-nyx-plugin", "gs-nyx", "torch")},
            "cuda_runtime": torch.version.cuda,
            "genesis_file": gs.__file__,
            "renderer": "Nyx FastPathTracer",
            "rgb_shape": [1, 240, 320, 3],
            "rgb_device": rgb_device,
            "steps": 61,
            "final_box_z": height,
            "mean_pixel_value": brightness,
            "pixel_std": contrast,
            "changed_pixel_fraction": changed_fraction,
            "images": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                       for name in ("initial.png", "settled.png")},
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not passed:
            raise RuntimeError("Nyx image or physics validation failed; inspect result.json and PNGs")
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
