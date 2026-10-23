"""Open a Genesis viewer window and repeatedly open/close the community QArm.

Use --headless --cycles 2 to check the same control loop without a window.
"""

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from mmic_project.qarm_gripper import FINGER_JOINTS, gripper_targets

ARM_JOINTS = ("YAW", "SHOULDER", "ELBOW", "WRIST")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--cycles", type=int, default=0, help="0 repeats until window close / Ctrl+C")
    parser.add_argument("--period", type=float, default=4.0, help="seconds for open -> closed -> open")
    parser.add_argument("--open", dest="open_grip", type=float, default=0.1)
    parser.add_argument("--closed", dest="closed_grip", type=float, default=0.9)
    args = parser.parse_args()
    if args.cycles < 0 or not math.isfinite(args.period) or args.period <= 0:
        parser.error("--cycles must be >= 0 and --period must be finite and > 0")
    if not 0 <= args.open_grip < args.closed_grip <= 1:
        parser.error("require 0 <= --open < --closed <= 1")
    if args.headless and not args.cycles:
        parser.error("--headless requires a finite --cycles count")

    import numpy as np
    import genesis as gs

    urdf = ROOT / "assets/qarm-community/qarm_with_gripper.urdf"
    if not urdf.is_file():
        parser.error(f"Missing community model: {urdf}")
    dt = 0.01
    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, logging_level="warning")
    try:
        scene = gs.Scene(
            # A motion viewer, not a grasp/contact experiment. Recompute the
            # placeholder inertias, disable collisions, and remove gravity.
            sim_options=gs.options.SimOptions(dt=dt, substeps=4, gravity=(0, 0, 0)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(0.95, -0.65, 0.85), camera_lookat=(0.35, 0, 0.45),
                camera_fov=40, refresh_rate=60),
            show_viewer=not args.headless,
        )
        arm = scene.add_entity(gs.morphs.URDF(
            file=str(urdf), fixed=True, collision=False, recompute_inertia=True))
        scene.build()
        arm_dofs = [arm.get_joint(name).dofs_idx_local[0] for name in ARM_JOINTS]
        finger_dofs = [arm.get_joint(name).dofs_idx_local[0] for name in FINGER_JOINTS]
        dofs = arm_dofs + finger_dofs
        # Upstream arm effort limits are zero. These caps are demo settings.
        caps = np.array([10.6, 21.2, 10.6, 4.1, 1.9, 1.9, 1.9, 1.9])
        arm.set_dofs_force_range(-caps, caps, dofs)
        arm.set_dofs_kp(np.array([100., 100., 100., 50., 2., 2., 2., 2.]), dofs)
        arm.set_dofs_kv(np.array([10., 10., 10., 5., 0.1, 0.1, 0.1, 0.1]), dofs)
        grip = args.open_grip
        arm.set_dofs_position(np.r_[np.zeros(4), gripper_targets(grip)], dofs)
        step, next_cycle = 0, 0
        low, high = np.full(4, np.inf), np.full(4, -np.inf)
        deadline = time.perf_counter()
        print("One gripper scalar ->", ", ".join(FINGER_JOINTS), flush=True)
        print("Close the viewer or press Ctrl+C to stop.", flush=True)
        while args.cycles == 0 or step * dt < args.cycles * args.period:
            if not args.headless and not scene.viewer.is_alive():
                break
            t = step * dt
            goal = args.open_grip + (args.closed_grip - args.open_grip) * (
                0.5 - 0.5 * math.cos(2 * math.pi * t / args.period))
            # Same 0.2 s smoothing used by the reference controller.
            grip += (goal - grip) * min(1.0, dt / 0.2)
            arm.control_dofs_position(np.r_[np.zeros(4), gripper_targets(grip)], dofs)
            scene.step()
            q = arm.get_dofs_position(finger_dofs).cpu().numpy().reshape(-1)
            if not np.isfinite(q).all():
                raise RuntimeError(f"Non-finite finger state at step {step}: {q}")
            low, high = np.minimum(low, q), np.maximum(high, q)
            if t >= next_cycle * args.period:
                print(f"cycle {next_cycle + 1}: grip={grip:.3f}, finger angles={q.round(3)}", flush=True)
                next_cycle += 1
            step += 1
            if not args.headless:
                deadline += dt
                time.sleep(max(0.0, deadline - time.perf_counter()))
        if args.headless and np.any(high - low < 0.02):
            raise RuntimeError(f"One or more finger joints did not cycle: range={high - low}")
        print(json.dumps({"steps": step, "finger_min_rad": low.tolist(), "finger_max_rad": high.tolist()}))
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
