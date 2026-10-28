"""Conversion between MuJoCo joint angles (radians) and the units LeRobot uses
for a real SO-101 follower, so that data recorded in simulation has the same
meaning as data recorded on the real arms.

LeRobot (0.6.x, `use_degrees=True`, the default) reports each arm joint in
degrees from the middle of its calibrated range: (raw - mid) * 360 / 4096.
The gripper is 0..100 (0 = closed). Keep degrees on: it does not depend on
each arm's calibration file, so sim data matches every calibrated arm.

The simulation model uses the "new calibration" convention (zero = middle of
each joint's range), which is the same zero.

VERIFY ON HARDWARE before trusting sim data: run
`python -m competition.sim.check_real_vs_sim` with a few poses recorded on
the real arm and flip any JOINT_SIGN entry whose joint moves the wrong way.
"""

import numpy as np

JOINT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
ARM_JOINTS = JOINT_NAMES[:5]

# +1 when increasing the real joint value turns the sim joint the positive way.
# TODO: confirm each sign on the real arm.
JOINT_SIGN = {name: 1.0 for name in ARM_JOINTS}
# Extra offset (degrees) if your real zero differs from the sim zero.
JOINT_OFFSET_DEG = {name: 0.0 for name in ARM_JOINTS}

GRIPPER_CLOSED_RAD = -0.17453
GRIPPER_OPEN_RAD = 1.74533


def sim_to_real(q: np.ndarray) -> np.ndarray:
    """6 sim joint angles (rad) -> LeRobot values (5 x degrees, gripper 0..100)."""
    q = np.asarray(q, dtype=float)
    out = np.empty(6, dtype=np.float32)
    for i, name in enumerate(ARM_JOINTS):
        out[i] = JOINT_SIGN[name] * np.degrees(q[i]) + JOINT_OFFSET_DEG[name]
    out[5] = 100.0 * (q[5] - GRIPPER_CLOSED_RAD) / (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    return out


def real_to_sim(v: np.ndarray) -> np.ndarray:
    """LeRobot values (5 x degrees, gripper 0..100) -> 6 sim joint angles (rad)."""
    v = np.asarray(v, dtype=float)
    out = np.empty(6)
    for i, name in enumerate(ARM_JOINTS):
        out[i] = np.radians((v[i] - JOINT_OFFSET_DEG[name]) / JOINT_SIGN[name])
    out[5] = GRIPPER_CLOSED_RAD + v[5] / 100.0 * (GRIPPER_OPEN_RAD - GRIPPER_CLOSED_RAD)
    return out


def feature_names(prefix: str = "") -> list[str]:
    """LeRobot feature names, e.g. 'left_shoulder_pan.pos' for a bimanual robot."""
    return [f"{prefix}{name}.pos" for name in JOINT_NAMES]
