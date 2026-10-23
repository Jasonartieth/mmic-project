# Community QArm model

Downloaded from [Georgiaac3/qarm-rl](https://github.com/Georgiaac3/qarm-rl),
commit `d13dfeb0fa47b1288bf91a77e0e06bf11227d20c`.
Source: `genesis/QARM/urdf/qarm_with_gripper.urdf` and its referenced meshes.

- `urdf/qarm_with_gripper.urdf`: identical except trailing whitespace cleanup and four mesh filename references
  use uppercase `LINK1A.STL`, `LINK2A.STL`, `LINK1B.STL`, and `LINK2B.STL`
  to match the upstream mesh filenames on case-sensitive filesystems.
- `meshes/`: the ten referenced upstream STL files, unchanged.
- `source-manifest.json`: source revision, path fixes, file sizes, and hashes.

Keep the directory structure intact for relative mesh references. This is the
only QArm model in the tracked assets; the previous derived model was removed.
The original upstream URDF hash is retained in the manifest.

The viewer demo at `src/scripts/qarm/gripper_demo.py` maps one gripper scalar
to the four finger targets using the reference coefficients
`[-0.45, -0.35, +0.45, -0.35]` radians in joint order
`[Joint1A, Joint2A, Joint1B, Joint2B]`. It uses 0.2 s command smoothing.
This is the requested initial approximation, not a calibrated mapping for these
community joint frames. The URDF itself retains four articulated finger joints.

This is an inspection candidate, not a validated replacement. The upstream file
states its masses/inertias are example values. It includes zero arm-joint
effort/velocity limits and a zero-mass finger link. Its gripper alignment and
motion have not been verified against the physical arm.

No license file was found in the community repository at this revision. This
download does not establish redistribution rights for community modifications.
