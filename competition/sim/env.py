"""Game 1 (pick-and-place sorting) simulation environment.

Actions and joint states use the same units as a real LeRobot SO-101 follower
(see units.py): 12 values, left arm then right arm, each 5 joints in degrees
plus the gripper 0..100. Images come from the same cameras as the real setup.

    env = Game1Env(round_name="colour_sort", seed=0)
    obs = env.reset()
    obs = env.step(action)        # action: np.ndarray of shape (12,)
    env.score()
"""

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np

from . import layout, units
from .scene import CAMERAS, JOINTS, build_model

CONTROL_HZ = 30
SIDES = ("left", "right")


class Game1Env:
    def __init__(self, round_name: str = "colour_sort", seed: int | None = None,
                 image_size: tuple[int, int] | None = (480, 640), cameras=CAMERAS,
                 domain_randomization: bool = True):
        self.round_name = round_name
        self.model = build_model()
        self.model.opt.timestep = 1 / 600
        self.data = mujoco.MjData(self.model)
        self.substeps = round(1 / CONTROL_HZ / self.model.opt.timestep)
        self.rng = np.random.default_rng(seed)
        self.cameras = tuple(cameras)
        self.domain_randomization = domain_randomization
        if os.environ.get("MUJOCO_GL") == "osmesa":
            # CPU rendering: shadows and anti-aliasing cost about 3x the time.
            self.model.light_castshadow[:] = 0
            self.model.vis.quality.offsamples = 0
        self.renderer = mujoco.Renderer(self.model, *image_size) if image_size else None
        self.objects = layout.GAME1_OBJECTS
        m = self.model
        self.qpos_adr = {s: np.array([m.joint(f"{s}_{j}").qposadr[0] for j in JOINTS]) for s in SIDES}
        self.ctrl_idx = {s: np.array([m.actuator(f"{s}_{j}").id for j in JOINTS]) for s in SIDES}
        self.obj_qpos = {o.name: m.joint(f"{o.name}_free").qposadr[0] for o in self.objects}
        self.obj_body = {o.name: m.body(o.name).id for o in self.objects}
        self.bin_mocap = {c: m.body_mocapid[m.body(f"bin_{c}").id] for c in layout.COLORS}
        self._defaults = dict(light_diffuse=m.light_diffuse.copy(), geom_rgba=m.geom_rgba.copy(),
                              cam_pos=m.cam_pos.copy(), cam_quat=m.cam_quat.copy())
        self.ready_pose = {s: self._ready_pose(s) for s in SIDES}
        self.time = 0.0

    def _ready_pose(self, side: str) -> np.ndarray:
        """Start pose: gripper open, pointing down above the arm's side of the source area."""
        from .ik import ArmIK

        bx, by = layout.ARMS[side]
        q, _, _ = ArmIK(self.model, side).solve_grasp([bx, 0.22, 0.10], [0.0], np.zeros(5), (bx, by))
        return np.append(q, 1.2)

    # ------------------------------------------------------------------ reset
    def reset(self, seed: int | None = None, bin_positions: dict | None = None) -> dict:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        if self.domain_randomization:
            self._randomize_appearance()
        for color, xy in (bin_positions or layout.DEFAULT_BIN_POSITIONS).items():
            d.mocap_pos[self.bin_mocap[color]] = [xy[0], xy[1], 0.0]
        self._scatter_objects()
        for s in SIDES:
            d.qpos[self.qpos_adr[s]] = self.ready_pose[s]
            d.ctrl[self.ctrl_idx[s]] = self.ready_pose[s]
        # Let objects settle into stable resting poses.
        for _ in range(int(0.8 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        self.time = 0.0
        return self.observation()

    def _scatter_objects(self) -> None:
        """Random, non-overlapping resting poses inside the source area, each
        within reach of at least one arm (rule 5.8.1.22)."""
        (x0, x1), (y0, y1) = layout.SOURCE_AREA["x"], layout.SOURCE_AREA["y"]
        bases = np.array(list(layout.ARMS.values()))
        # Large objects first packs better; tighten the spacing if they do not fit.
        order = sorted(range(len(self.objects)),
                       key=lambda i: (-layout.SIZES[self.objects[i].size], self.rng.random()))
        for attempt in range(60):
            gap = 0.006 if attempt < 30 else 0.002
            placed = []
            for i in order:
                obj = self.objects[i]
                s = layout.SIZES[obj.size]
                r = 0.62 * s
                for _ in range(2000):
                    xy = self.rng.uniform([x0 + r, y0 + r], [x1 - r, y1 - r])
                    if (np.linalg.norm(bases - xy, axis=1).min() <= layout.MAX_REACH
                            and all(np.linalg.norm(xy - p) > r + q + gap for _, p, q in placed)):
                        placed.append((obj, xy, r))
                        break
                else:
                    break
            if len(placed) == len(self.objects):
                break
        else:
            raise RuntimeError("could not fit the object set into the reachable source area")
        for obj, xy, _ in placed:
            s = layout.SIZES[obj.size]
            yaw = self.rng.uniform(-np.pi, np.pi)
            adr = self.obj_qpos[obj.name]
            self.data.qpos[adr:adr + 3] = [xy[0], xy[1], s / 2 + 0.002]
            self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]

    def _randomize_appearance(self) -> None:
        """Lighting, colour and camera jitter so policies trained here are less
        brittle (the judges also vary the lighting, rule 5.4.3.7)."""
        m, d0, rng = self.model, self._defaults, self.rng
        m.light_diffuse[:] = d0["light_diffuse"] * rng.uniform(0.6, 1.4, size=(m.nlight, 1))
        # Only geoms without a material: writing rgba on the others would
        # override their material colour (the arm parts).
        plain = m.geom_matid < 0
        m.geom_rgba[:] = d0["geom_rgba"]
        jitter = rng.uniform(-0.06, 0.06, size=(int(plain.sum()), 3))
        m.geom_rgba[plain, :3] = np.clip(d0["geom_rgba"][plain, :3] + jitter, 0, 1)
        m.cam_pos[:] = d0["cam_pos"] + rng.normal(0, 0.005, size=d0["cam_pos"].shape)

    # ------------------------------------------------------------------- step
    def step(self, action: np.ndarray) -> dict:
        action = np.asarray(action, dtype=float).reshape(2, 6)
        for side, a in zip(SIDES, action):
            self.data.ctrl[self.ctrl_idx[side]] = units.real_to_sim(a)
        for _ in range(self.substeps):
            mujoco.mj_step(self.model, self.data)
        self.time += 1 / CONTROL_HZ
        return self.observation()

    def observation(self, render: bool = True) -> dict:
        state = np.concatenate([units.sim_to_real(self.data.qpos[self.qpos_adr[s]]) for s in SIDES])
        obs = {"state": state.astype(np.float32), "images": {}}
        if render and self.renderer is not None:
            for cam in self.cameras:
                self.renderer.update_scene(self.data, camera=cam)
                obs["images"][cam] = self.renderer.render().copy()
        return obs

    # -------------------------------------------------------- privileged info
    def object_pose(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        adr = self.obj_qpos[name]
        return self.data.qpos[adr:adr + 3].copy(), self.data.qpos[adr + 3:adr + 7].copy()

    def bin_position(self, color: str) -> np.ndarray:
        return self.data.mocap_pos[self.bin_mocap[color]].copy()

    def move_bins(self, targets: dict) -> None:
        """What the judge does in Round 3 (rule 5.8.1.47): move bins to new
        positions ({colour: (x, y)}); objects already in a bin move with it."""
        contents = {c: [o.name for o in self.objects if self.bin_of(o.name) == c] for c in targets}
        for color, xy in targets.items():
            old = self.bin_position(color)
            delta = np.array([xy[0] - old[0], xy[1] - old[1], 0.0])
            for name in contents[color]:
                adr = self.obj_qpos[name]
                self.data.qpos[adr:adr + 3] += delta
            self.data.mocap_pos[self.bin_mocap[color]] = old + delta
        mujoco.mj_forward(self.model, self.data)

    def nudge_object(self, name: str, dxy) -> None:
        """Slide an object on the table, as the judge may in Round 3 (rule 5.8.1.48)."""
        adr = self.obj_qpos[name]
        self.data.qpos[adr:adr + 2] += dxy
        mujoco.mj_forward(self.model, self.data)

    def bin_of(self, name: str) -> str | None:
        """Which bin an object rests fully inside, if any."""
        pos, _ = self.object_pose(name)
        half = layout.BIN_INNER / 2
        for color in layout.COLORS:
            b = self.bin_position(color)
            if (abs(pos[0] - b[0]) < half and abs(pos[1] - b[1]) < half
                    and pos[2] < layout.BIN_WALL_HEIGHT + 0.08):
                return color
        return None

    def score(self) -> dict:
        """Round score as in rules 5.8.1.51 - 5.8.1.62."""
        correct = wrong = 0
        for obj in self.objects:
            got = self.bin_of(obj.name)
            if got is None:
                continue
            if got == layout.target_bin(self.round_name, obj.color, obj.shape):
                correct += obj.points
            else:
                wrong += obj.points
        all_done = all(self.bin_of(o.name) == layout.target_bin(self.round_name, o.color, o.shape)
                       for o in self.objects)
        total = max(0, min(100, correct - wrong + (layout.COMPLETION_BONUS if all_done else 0)))
        return {"correct_points": correct, "wrong_points": wrong, "complete": all_done,
                "score": total, "time_s": round(self.time, 1)}
