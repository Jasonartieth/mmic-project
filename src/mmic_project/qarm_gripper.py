"""Approximate single-input gripper mapping from course_project at 306560f.

Angles are radians. Coefficients are uncalibrated for the community joint frames.
"""

import math

FINGER_JOINTS = ("Joint1A", "Joint2A", "Joint1B", "Joint2B")
FINGER_CLOSE_RAD = (-0.45, -0.35, 0.45, -0.35)


def gripper_targets(grip: float) -> tuple[float, float, float, float]:
    """Map one scalar (0 open, 1 closed) to four coordinated position targets."""
    if not math.isfinite(grip) or not 0.0 <= grip <= 1.0:
        raise ValueError("grip must be finite and in [0, 1]")
    return tuple(grip * angle for angle in FINGER_CLOSE_RAD)
