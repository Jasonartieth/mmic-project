"""Check Genesis with a headless falling-box simulation."""

import argparse
import json
import math
from importlib.metadata import version


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    args = parser.parse_args()

    import genesis as gs

    gs.init(backend=gs.cpu if args.backend == "cpu" else gs.gpu, seed=0)
    try:
        scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=0.01),
            show_viewer=False,
        )
        scene.add_entity(gs.morphs.Plane())
        box = scene.add_entity(gs.morphs.Box(pos=(0, 0, 0.5), size=(0.1, 0.1, 0.1)))
        scene.build()
        initial_z = float(box.get_pos()[2].item())
        for _ in range(100):
            scene.step()
        final_z = float(box.get_pos()[2].item())
        if not math.isfinite(final_z) or not 0.03 <= final_z <= 0.08:
            raise RuntimeError(f"Box failed to settle on the plane: z={final_z}")
        print(json.dumps({
            "genesis_version": version("genesis-world"),
            "backend": str(gs.backend),
            "steps": 100,
            "initial_z": initial_z,
            "final_z": final_z,
            "status": "passed",
        }, indent=2))
    finally:
        gs.destroy()


if __name__ == "__main__":
    main()
