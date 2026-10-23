# Quanser QArm assets

Use `urdf/qarm_gripper.urdf` for the arm with its attached gripper, or
`urdf/qarm.urdf` for the arm alone. Keep the directory structure intact: both
URDFs use relative paths to `meshes/visual/` and `meshes/collision/`.

The arm joints are `YAW`, `SHOULDER`, `ELBOW`, and `WRIST`. Angles are in radians
and lengths are in meters. The model includes a fixed base and a `tcp` frame.

These are derived simulation assets, copied without further modification from
`refs/course_project/models/qarm/` at commit
`306560fb613288041b83767755fdf89ce364a85f` of
[Jasonartieth/course_project](https://github.com/Jasonartieth/course_project).
They do not depend on that reference checkout at runtime.

The original models and meshes come from
[quanser/urdf_representations](https://github.com/quanser/urdf_representations)
at commit `071f9a66eada8318adff5815bb0302f38ac8b1a0`, under the BSD-3-Clause
license reproduced in `LICENSE`.

The reference project's generation scripts
(`scripts/model/simplify_meshes.py` and `scripts/model/build_qarm_urdf.py`)
produce simplified visual meshes, convex collision meshes, corrected joint
effort/velocity limits, transformed inertial parameters, an attached gripper,
and a TCP frame. The included `urdf/build_report.json` and
`meshes/mesh_report.json` record those transformations.

Gripper joint mapping, mounting alignment, and simulation dynamics still need
real-arm calibration. The gripper contains approximate mass and joint data;
this model is not evidence of physical collision safety or calibrated dynamics.
