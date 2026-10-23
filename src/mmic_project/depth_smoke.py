"""Validate Genesis's native CUDA DepthCamera against analytic box intersections."""

import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/depth-camera"))
    args = parser.parse_args()

    print("Loading Genesis for the native depth sensor...", flush=True)
    import genesis as gs
    import numpy as np
    import torch

    if not torch.cuda.is_available() or torch.version.hip:
        raise RuntimeError("This server validation requires NVIDIA CUDA")
    args.output.mkdir(parents=True, exist_ok=True)
    width, height, batch = 320, 240, 4
    max_range, tolerance = 6.0, 2e-4
    origin = np.array([0.0, 0.0, 1.0])
    front_size = np.array([0.2, 1.0, 1.0])
    back_center = np.array([4.0, 0.0, 1.0])
    back_size = np.array([0.1, 4.0, 3.0])

    # Independent pinhole rays and slab intersections provide the ground truth.
    fx = fy = width / 2.0  # 90-degree horizontal FOV, square pixels
    uu, vv = np.meshgrid(np.arange(width) + 0.5, np.arange(height) + 0.5)
    rays = np.stack([np.ones_like(uu), -(uu - width / 2) / fx,
                     -(vv - height / 2) / fy], axis=-1)
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)

    def box_ranges(center, size):
        near = (center - size / 2 - origin) / rays
        far = (center + size / 2 - origin) / rays
        enter = np.minimum(near, far).max(axis=-1)
        leave = np.maximum(near, far).min(axis=-1)
        return np.where((leave >= np.maximum(enter, 0)) & (enter >= 0), enter, np.inf)

    print(f"Initializing native CUDA raycasting on {torch.cuda.get_device_name()}...", flush=True)
    gs.init(backend=gs.cuda, seed=0, logging_level="warning")
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
            show_viewer=False,
        )
        front = scene.add_entity(gs.morphs.Box(pos=(2.0, 0.0, 1.0), size=tuple(front_size)))
        scene.add_entity(gs.morphs.Box(pos=tuple(back_center), size=tuple(back_size), fixed=True))
        sensor = scene.add_sensor(gs.sensors.DepthCamera(
            pattern=gs.sensors.DepthCameraPattern(res=(width, height), fov_horizontal=90.0),
            pos_offset=tuple(origin), max_range=max_range, return_points=False,
        ))
        print("Building four environments and the native DepthCamera...", flush=True)
        scene.build(n_envs=batch)
        frames, expected_frames, checks = [], [], []
        tensor_device = None
        for label, movement in (("initial", 0.0), ("moved", 0.5)):
            centers = np.array([[2.0 + 0.25 * env + movement, 0.0, 1.0] for env in range(batch)])
            front.set_pos(torch.tensor(centers, dtype=gs.tc_float, device=gs.device))
            scene.step()
            depth = sensor.read_image()
            if depth.shape != (batch, height, width) or depth.dtype != torch.float32 or not depth.is_cuda:
                raise RuntimeError(f"Unexpected depth tensor: {depth.shape}, {depth.dtype}, {depth.device}")
            if sensor.read().points is not None:
                raise RuntimeError("Depth-only sensor unexpectedly returned points")
            tensor_device = str(depth.device)
            measured = depth.detach().cpu().numpy().copy()
            expected = np.stack([
                np.minimum(np.minimum(box_ranges(center, front_size), box_ranges(back_center, back_size)), max_range)
                for center in centers
            ])
            frames.append(measured)
            expected_frames.append(expected)
            for env in range(batch):
                hits = expected[env] < max_range
                misses = ~hits
                max_error = float(np.abs(measured[env][hits] - expected[env][hits]).max())
                miss_matches = bool(np.all(measured[env][misses] == max_range))
                expected_center = float(expected[env, height // 2, width // 2])
                measured_center = float(measured[env, height // 2, width // 2])
                checks.append({
                    "frame": label, "env": env,
                    "center_range_m": measured_center, "expected_center_range_m": expected_center,
                    "max_hit_error_m": max_error,
                    "hit_pixels": int(hits.sum()), "miss_pixels": int(misses.sum()),
                    "misses_equal_max_range": miss_matches,
                    "passed": bool(np.isfinite(measured[env]).all() and max_error <= tolerance
                                   and miss_matches and hits.any() and misses.any()),
                })
            print(f"Read {label}: {tuple(depth.shape)} {depth.dtype} {depth.device}; "
                  f"center ranges {measured[:, height // 2, width // 2].tolist()}", flush=True)

        ranges = np.stack(frames)
        expected_ranges = np.stack(expected_frames)
        axial = np.where(ranges < max_range, ranges * rays[None, None, ..., 0], np.nan)
        center_delta = ranges[1, :, height // 2, width // 2] - ranges[0, :, height // 2, width // 2]
        expected_delta = expected_ranges[1, :, height // 2, width // 2] - expected_ranges[0, :, height // 2, width // 2]
        motion_passed = bool(np.allclose(center_delta, expected_delta, atol=tolerance, rtol=0))
        passed = all(check["passed"] for check in checks) and motion_passed
        np.savez_compressed(args.output / "depth.npz", range_m=ranges, axial_depth_m=axial,
                            expected_range_m=expected_ranges, ray_directions=rays,
                            frame_names=np.array(["initial", "moved"]))

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("lightgray")
        for frame, label in enumerate(("Initial", "After moving 0.5 m away")):
            image = np.ma.masked_where(ranges[frame, 0] == max_range, ranges[frame, 0])
            plot = axes[frame].imshow(image, vmin=1.8, vmax=max_range, cmap=cmap)
            axes[frame].set_title(label)
        fig.colorbar(plot, ax=axes[:2], label="Native along-ray range (m)", shrink=0.8)
        error = np.ma.masked_where(expected_ranges[1, 0] == max_range,
                                   np.abs(ranges[1, 0] - expected_ranges[1, 0]) * 1000)
        plot_error = axes[2].imshow(error, cmap="magma", vmin=0, vmax=tolerance * 1000)
        axes[2].set_title("Absolute error against geometry")
        fig.colorbar(plot_error, ax=axes[2], label="Error (mm)", shrink=0.8)
        for axis in axes:
            axis.set_xlabel("Pixel column")
            axis.set_ylabel("Pixel row")
        fig.savefig(args.output / "depth-preview.png", dpi=150)
        plt.close(fig)
        result = {
            "status": "passed" if passed else "failed",
            "hostname": platform.node(), "gpu": torch.cuda.get_device_name(),
            "versions": {name: version(name) for name in ("genesis-world", "torch", "quadrants")},
            "sensor": "gs.sensors.DepthCamera", "backend": "Genesis CUDA geometric raycaster",
            "depth_shape": list(ranges.shape[1:]), "depth_dtype": "float32", "depth_device": tensor_device,
            "frame_names": ["initial", "moved"], "native_units": "along-ray range in meters",
            "max_range_m": max_range, "no_hit_value": max_range, "tolerance_m": tolerance,
            "intrinsics": {"width": width, "height": height, "fx": fx, "fy": fy,
                           "cx": width / 2, "cy": height / 2, "pixel_center_offset": 0.5},
            "pose": {"position_m": origin.tolist(), "optical_axis": "+X", "image_right": "-Y", "image_down": "-Z"},
            "front_box_size_m": front_size.tolist(), "initial_front_x_by_env_m": [2.0 + 0.25 * i for i in range(batch)],
            "back_box_center_m": back_center.tolist(), "back_box_size_m": back_size.tolist(),
            "movement_m": 0.5, "center_range_change_m": center_delta.tolist(), "motion_passed": motion_passed,
            "checks": checks,
            "artifacts": {name: hashlib.sha256((args.output / name).read_bytes()).hexdigest()
                          for name in ("depth.npz", "depth-preview.png")},
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2), flush=True)
        if not passed:
            raise RuntimeError("Native depth validation failed; inspect result.json")
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
