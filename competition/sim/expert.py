"""Scripted demonstrator for Game 1. It plays the role of the human on the
leader arms: it sees the true object poses (privileged simulator state),
works out each pick-and-place with IK and moves both arms in parallel.

The only point of this file is to generate demonstrations in simulation. The
policy you deploy learns from camera images and joint states, never from
this code (see ik.py).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from . import layout, units
from .env import SIDES, Game1Env
from .ik import ArmIK

GRIPPER_OPEN = 1.2  # rad, wide enough for the large objects
GRIPPER_CLOSED = units.GRIPPER_CLOSED_RAD
JOINT_SPEED = 1.6  # rad/s for the fastest joint; real demos are of this order
FINGER_REACH = 0.024  # fingertips extend this far past the gripperframe site
FINGER_CLEARANCE = 0.004  # gap between the fixed finger and the object before closing
JAW_OPEN_REACH = 0.049  # moving jaw's inner face, from the fixed finger, when open
MIDDLE_HALF_WIDTH = 0.12  # both arms can be in |x| < this, so it is shared


@dataclass
class Task:
    side: str
    obj: layout.ObjectSpec
    dest: str  # bin colour, or "handover"

    @property
    def text(self) -> str:
        """Instruction for language-conditioned policies (SmolVLA, pi0.5)."""
        name = f"{self.obj.size} {self.obj.color} {self.obj.shape.replace('_', ' ')}"
        where = "the handover zone" if self.dest == "handover" else f"the {self.dest} bin"
        return f"pick up the {name} and place it in {where}"


def _yaw(quat) -> float:
    w, x, y, z = quat
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _grasp_yaws(shape: str, yaw: float) -> list[float]:
    """Closing directions that put the fingers on flat faces."""
    if shape == "cube":
        return [yaw, yaw + np.pi / 2]
    if shape == "hex_prism":
        return [yaw + np.pi / 6 + k * np.pi / 3 for k in range(3)]
    if shape == "dodecahedron":
        return [yaw + k * np.pi / 5 for k in range(5)]
    return [yaw + k * np.pi / 4 for k in range(4)]  # cylinder


def _closing_yaw(ik: ArmIK, q) -> float:
    """Horizontal angle of the closing direction (site z) for arm joints q."""
    ik.data.qpos[ik.qadr] = q
    mujoco.mj_kinematics(ik.model, ik.data)
    z = ik.data.site_xmat[ik.site].reshape(3, 3)[:, 2]
    return np.arctan2(z[1], z[0])


def _min_jerk(q0, q1, duration):
    steps = max(2, int(round(duration * 30)))
    for t in np.linspace(0, 1, steps)[1:]:
        s = 10 * t ** 3 - 15 * t ** 4 + 6 * t ** 5
        yield q0 + (q1 - q0) * s


class Game1Expert:
    def __init__(self, env: Game1Env, speed: float = 1.0):
        self.env = env
        self.ik = {s: ArmIK(env.model, s) for s in SIDES}
        self.speed = speed
        self.claimed: set[str] = set()
        self.failed: dict[str, int] = {}
        self.middle_owner: str | None = None
        self.last_failure: str | None = None
        self.failure_log: list[tuple[str, str]] = []
        self.ready = env.ready_pose

    # ----------------------------------------------------------- task choice
    def _reaches(self, side: str, xy) -> bool:
        return np.hypot(*(np.asarray(xy) - layout.ARMS[side])) <= layout.MAX_REACH + 0.01

    def _bin_side(self, color: str) -> str:
        return min(SIDES, key=lambda s: np.hypot(*(self.env.bin_position(color)[:2] - layout.ARMS[s])))

    def next_task(self, side: str) -> Task | None:
        env, other = self.env, SIDES[1 - SIDES.index(side)]
        best = None
        for obj in env.objects:
            if obj.name in self.claimed or self.failed.get(obj.name, 0) >= 3:
                continue
            want = layout.target_bin(env.round_name, obj.color, obj.shape)
            if env.bin_of(obj.name) == want:
                continue
            pos, _ = env.object_pose(obj.name)
            if pos[2] < -0.05 or not self._reaches(side, pos[:2]):
                continue
            if self._bin_side(want) == side:
                dest = want
            elif self._reaches(other, pos[:2]):
                continue  # the other arm can do this one directly
            elif not self._handover_free():
                continue
            else:
                dest = "handover"
            key = (-obj.points, np.hypot(*(pos[:2] - layout.ARMS[side])))
            if best is None or key < best[0]:
                best = (key, Task(side, obj, dest))
        return best[1] if best else None

    def _handover_free(self) -> bool:
        hx, hy = layout.HANDOVER_XY
        for obj in self.env.objects:
            p, _ = self.env.object_pose(obj.name)
            if np.hypot(p[0] - hx, p[1] - hy) < 0.05 and p[2] < 0.08:
                return False
        return not any(n.endswith("@handover") for n in self.claimed)

    # --------------------------------------------------------------- motion
    def _move(self, side, q0, q1, slow=1.0):
        duration = max(0.25, np.abs(q1[:5] - q0[:5]).max() / (JOINT_SPEED * self.speed)) * slow
        yield from _min_jerk(q0, q1, duration)

    def _hold(self, q, seconds):
        for _ in range(int(seconds * 30)):
            yield q

    def _finger_clearance(self, task: Task, pos, yaw: float, offset: float) -> float:
        """Free space around both fingers for a grasp at this yaw, so the open
        jaw does not land on a neighbouring object."""
        c = np.array([np.cos(yaw), np.sin(yaw)])
        fingers = (pos[:2] - c * (offset + 0.006), pos[:2] + c * (JAW_OPEN_REACH + 0.006 - offset))
        free = np.inf
        for other in self.env.objects:
            if other is task.obj:
                continue
            p, _ = self.env.object_pose(other.name)
            if p[2] > 0.1 or p[2] < -0.05:
                continue
            r = 0.62 * layout.SIZES[other.size] + 0.006
            free = min(free, *(np.hypot(*(f - p[:2])) - r for f in fingers))
        return free

    def _plan_grasp(self, task: Task, pos, quat, offset, grasp_z, q0):
        """(above, grasp) arm joints for the clearest feasible grasp, or None."""
        ik, base = self.ik[task.side], layout.ARMS[task.side]
        yaws = _grasp_yaws(task.obj.shape, _yaw(quat))
        yaws += [y + np.pi for y in yaws]  # fixed finger on either side
        scored = sorted(((self._finger_clearance(task, pos, y, offset), y) for y in yaws), reverse=True)
        clear = [y for free, y in scored if free > 0.0] or [scored[0][1]]
        reason = "no IK at grasp height"
        for yaw in clear:
            c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            site = np.array([pos[0], pos[1], grasp_z]) - offset * c
            found = ik.solve_grasp([pos[0], pos[1], grasp_z], [yaw], q0, base, offset)
            if found is None:
                continue
            grasp, _, tilt = found
            reason = "no IK above object"
            for lift in (0.07, 0.05, 0.035):
                above = ik.solve_pose(site + [0, 0, lift], yaw, tilt, grasp, base, seeds=False)
                if above is not None:
                    return above, grasp
        self.last_failure = reason
        return None

    def run_task(self, task: Task):
        """Generator of 6 sim joint targets per control tick for one pick-and-place.
        Returns True if the object ended up where it was meant to go."""
        env, ik, side = self.env, self.ik[task.side], task.side
        base = layout.ARMS[side]
        s = layout.SIZES[task.obj.size]
        q = self.ready[side].copy()
        in_middle = False

        pos, quat = env.object_pose(task.obj.name)
        if abs(pos[0]) < MIDDLE_HALF_WIDTH or task.dest == "handover":
            while self.middle_owner not in (None, side):
                yield q
            self.middle_owner, in_middle = side, True
        try:
            # Re-read the pose: in Round 3 the judge may have moved the object.
            pos, quat = env.object_pose(task.obj.name)
            offset = s / 2 + FINGER_CLEARANCE
            grasp_z = max(pos[2], FINGER_REACH + 0.004)
            plan = self._plan_grasp(task, pos, quat, offset, grasp_z, q[:5])
            if plan is None:
                return False
            above, grasp = plan
            q_above = np.append(above, GRIPPER_OPEN)
            yield from self._move(side, q, q_above)
            q_grasp = np.append(grasp, GRIPPER_OPEN)
            yield from self._move(side, q_above, q_grasp, slow=1.5)
            q_closed = np.append(grasp, GRIPPER_CLOSED)
            yield from _min_jerk(q_grasp, q_closed, 0.4)
            yield from self._hold(q_closed, 0.2)
            q_lift = np.append(above, GRIPPER_CLOSED)
            yield from self._move(side, q_closed, q_lift, slow=1.5)
            if env.object_pose(task.obj.name)[0][2] < pos[2] + 0.02:
                yield from self._move(side, q_lift, self.ready[side])
                self.last_failure = "missed the grasp"
                return False

            if task.dest == "handover":
                hx, hy = layout.HANDOVER_XY
                drop_xy, drop_z = (hx, hy), s / 2 + 0.02
            else:
                b = env.bin_position(task.dest)
                drop_xy, drop_z = b[:2], layout.BIN_WALL_HEIGHT + s / 2 + 0.012
            # Where the object actually sits in the gripper, so the carry and
            # release poses put the object (not the gripper) over the bin.
            held = ik.site_frame_point(env.data.qpos[env.qpos_adr[side][:5]],
                                       env.object_pose(task.obj.name)[0])
            # Carry high enough to clear the bin walls, then let go above the
            # destination (an object falling a few cm into a bin is fine).
            carry_z = layout.BIN_WALL_HEIGHT + s / 2 + 0.035
            yaw0 = _closing_yaw(ik, above)
            yaws = [yaw0 + k * np.pi / 8 for k in (0, 1, -1, 2, -2, 3, -3, 4, -4)]
            q_path = [q_lift]
            for xy, required in ((pos[:2], False), (drop_xy, True)):
                found = None
                for z in np.linspace(carry_z, drop_z, 4) if required else (carry_z, carry_z - 0.02):
                    found = ik.solve_held([xy[0], xy[1], z], yaws, q_path[-1][:5], base, held)
                    if found is not None:
                        break
                if found is not None:
                    q_path.append(np.append(found, GRIPPER_CLOSED))
                elif required:
                    self.last_failure = "no IK above destination"
                    yield from self._move(side, q_lift, self.ready[side])
                    return False
            for q_from, q_to in zip(q_path, q_path[1:]):
                yield from self._move(side, q_from, q_to)
            q_drop = q_path[-1]
            # Round 3: the judge may have moved the bin while we were carrying.
            if task.dest != "handover":
                b = env.bin_position(task.dest)
                if np.hypot(*(b[:2] - drop_xy)) > 0.02:
                    found = ik.solve_held([b[0], b[1], carry_z], yaws, q_drop[:5], base, held)
                    if found is not None:
                        q_new = np.append(found, GRIPPER_CLOSED)
                        yield from self._move(side, q_drop, q_new)
                        q_drop = q_new
            if in_middle and task.dest != "handover":
                self.middle_owner, in_middle = None, False
            q_release = q_drop.copy()
            q_release[5] = GRIPPER_OPEN
            yield from _min_jerk(q_drop, q_release, 0.3)
            yield from self._hold(q_release, 0.3)
            yield from self._move(side, q_release, self.ready[side])
        finally:
            if in_middle:
                self.middle_owner = None
        if task.dest == "handover":
            p, _ = env.object_pose(task.obj.name)
            ok = np.hypot(p[0] - layout.HANDOVER_XY[0], p[1] - layout.HANDOVER_XY[1]) < 0.06
        else:
            ok = env.bin_of(task.obj.name) == task.dest
        self.last_failure = None if ok else "dropped on the way or missed the destination"
        return ok


def run_episode(env: Game1Env, expert: Game1Expert, time_limit: float = 300.0,
                on_step=None, on_task_end=None, judge=None, render: bool = True) -> dict:
    """Runs both arms until everything is sorted or time runs out.

    on_step(obs, action, segments) is called every control tick, where
    action is the 12 LeRobot-unit targets just applied and segments maps
    each side to (task, segment_id) or None while that arm is idle.
    on_task_end(side, task, segment_id, success) is called when an arm
    finishes a pick-and-place. judge(env), if given, runs every tick and may
    move bins and objects (Round 3)."""
    current = {s: None for s in SIDES}
    targets = {s: expert.ready[s].copy() for s in SIDES}
    counter = 0
    obs = env.observation(render=render)
    while env.time < time_limit:
        idle = 0
        for side in SIDES:
            if current[side] is None:
                task = expert.next_task(side)
                if task is not None:
                    key = task.obj.name + ("@handover" if task.dest == "handover" else "")
                    expert.claimed.add(key)
                    counter += 1
                    current[side] = (task, counter, key, expert.run_task(task))
                else:
                    idle += 1
            if current[side] is not None:
                task, seg, key, gen = current[side]
                try:
                    targets[side] = next(gen)
                except StopIteration as done:
                    expert.claimed.discard(key)
                    if on_task_end is not None:
                        on_task_end(side, task, seg, bool(done.value))
                    if not done.value:
                        expert.failure_log.append((task.obj.name, expert.last_failure))
                        expert.failed[task.obj.name] = expert.failed.get(task.obj.name, 0) + 1
                    current[side] = None
                    targets[side] = expert.ready[side].copy()
        action = np.concatenate([units.sim_to_real(targets[s]) for s in SIDES])
        segments = {s: (current[s][0], current[s][1]) if current[s] else None for s in SIDES}
        if on_step is not None:
            on_step(obs, action, segments)
        obs = env.step(action) if render else _step_no_render(env, action)
        if judge is not None:
            judge(env)
        if idle == len(SIDES):
            break  # nothing left that either arm can do
    return env.score()


def _step_no_render(env: Game1Env, action):
    renderer, env.renderer = env.renderer, None
    try:
        return env.step(action)
    finally:
        env.renderer = renderer
