"""Inverse kinematics for one SO-101 arm in the competition scene.

Only the scripted demonstrator uses this, to generate training data in
simulation. The policies you deploy learn from that data; nothing here runs
at competition time (rules 5.4.5.10-11 rule out IK-driven systems).
"""

import mujoco
import numpy as np

from .scene import JOINTS

ARM = JOINTS[:5]
# Starting guesses tried when the previous pose does not converge: arm
# reaching forward with the wrist bent down, at a few elbow angles.
SEEDS = tuple(np.array([0.0, lift, elbow, 1.0, 0.0]) for lift, elbow in ((-1, 0), (-1, 1), (0, 1), (-1, -1)))
# Approach tilts tried in order: vertical first, then leaning out or back.
TILTS = (0.0, 0.35, -0.35, 0.7, 1.0)
POS_TOL = 4e-3
ROT_TOL = 0.08


class ArmIK:
    def __init__(self, model: mujoco.MjModel, side: str):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site = model.site(f"{side}_gripperframe").id
        joints = [model.joint(f"{side}_{j}") for j in ARM]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dofs = np.array([j.dofadr[0] for j in joints])
        self.lo = np.array([j.range[0] for j in joints])
        self.hi = np.array([j.range[1] for j in joints])
        self.body = model.site_bodyid[self.site]
        self._jac_p = np.zeros((3, model.nv))
        self._jac_r = np.zeros((3, model.nv))

    @staticmethod
    def target_rotation(yaw: float, tilt: float = 0.0, radial=(0.0, 1.0)) -> np.ndarray:
        """Site orientation pointing the fingers down (site x = approach), with
        the closing direction (site z) horizontal at angle yaw. tilt leans the
        approach outward along radial (negative: back toward the arm base), for
        targets too far or too close for a vertical approach."""
        radial = np.array([radial[0], radial[1], 0.0])
        radial /= np.linalg.norm(radial) + 1e-12
        approach = np.array([0.0, 0.0, -1.0]) * np.cos(tilt) + radial * np.sin(tilt)
        close = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        close -= approach * (close @ approach)
        close /= np.linalg.norm(close)
        return np.column_stack([approach, np.cross(close, approach), close])

    def solve(self, pos, rot, q0, iters: int = 120, point=None):
        """Damped least squares on position + orientation. Returns (q, pos_err, rot_err).

        point, if given, is a point fixed in the site frame (e.g. the centre of
        a held object) that should end up at pos, instead of the site itself."""
        point = np.zeros(3) if point is None else np.asarray(point, dtype=float)
        m, d = self.model, self.data
        q = np.array(q0, dtype=float)
        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(target_quat, rot.ravel())
        cur_quat = np.zeros(4)
        err_r = np.zeros(3)
        err = np.zeros(6)
        jac = np.zeros((6, 5))
        damping = 1e-4 * np.eye(6)
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            site_rot = d.site_xmat[self.site].reshape(3, 3)
            world_point = d.site_xpos[self.site] + site_rot @ point
            err[:3] = pos - world_point
            mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site])
            mujoco.mju_subQuat(err_r, target_quat, cur_quat)
            # subQuat gives the error in the site frame; move it to world.
            err[3:] = 0.3 * (site_rot @ err_r)
            pos_err, rot_err = np.sqrt(err[:3] @ err[:3]), np.sqrt(err_r @ err_r)
            if pos_err < 0.5 * POS_TOL and rot_err < 0.5 * ROT_TOL:
                break
            mujoco.mj_jac(m, d, self._jac_p, self._jac_r, world_point, self.body)
            jac[:3] = self._jac_p[:, self.dofs]
            jac[3:] = 0.3 * self._jac_r[:, self.dofs]
            dq = jac.T @ np.linalg.solve(jac @ jac.T + damping, err)
            q = np.minimum(np.maximum(q + np.clip(dq, -0.2, 0.2), self.lo), self.hi)
        else:
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, d)
            site_rot = d.site_xmat[self.site].reshape(3, 3)
            pos_err = np.linalg.norm(pos - d.site_xpos[self.site] - site_rot @ point)
            mujoco.mju_mat2Quat(cur_quat, d.site_xmat[self.site])
            mujoco.mju_subQuat(err_r, target_quat, cur_quat)
            rot_err = np.linalg.norm(err_r)
        return q, pos_err, rot_err

    def solve_pose(self, pos, yaw, tilt, q0, base_xy, seeds=True, point=None):
        """One target orientation, trying the given start and then the seeds."""
        pos = np.asarray(pos, dtype=float)
        rot = self.target_rotation(yaw, tilt, pos[:2] - np.asarray(base_xy))
        for seed in (q0, *(SEEDS if seeds else ())):
            q, ep, er = self.solve(pos, rot, seed, point=point)
            if ep < POS_TOL and er < ROT_TOL:
                return q
        return None

    def solve_held(self, pos, yaws, q0, base_xy, point):
        """Joints putting a held object (centre at point in the site frame) at
        pos, fingers down. Returns q or None."""
        for tilt in TILTS:
            for yaw in yaws:
                q = self.solve_pose(pos, yaw, tilt, q0, base_xy, point=point)
                if q is not None:
                    return q
        return None

    def site_frame_point(self, q, world_point) -> np.ndarray:
        """A world point expressed in the site frame for arm joints q."""
        self.data.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        rot = self.data.site_xmat[self.site].reshape(3, 3)
        return rot.T @ (np.asarray(world_point) - self.data.site_xpos[self.site])

    def solve_grasp(self, pos, yaws, q0, base_xy, finger_offset: float = 0.0):
        """First solution over candidate yaws (in the order given), preferring
        a vertical approach. Returns (q, yaw, tilt) or None.

        The gripperframe site sits on the fixed finger's inner face, not
        between the fingers, so to grasp an object centred at pos the site
        goes finger_offset (half the object width plus clearance) back along
        the closing direction."""
        pos = np.asarray(pos, dtype=float)
        for tilt in TILTS:
            for yaw in yaws:
                target = pos - finger_offset * np.array([np.cos(yaw), np.sin(yaw), 0.0])
                q = self.solve_pose(target, yaw, tilt, q0, base_xy)
                if q is not None:
                    return q, yaw, tilt
        return None
