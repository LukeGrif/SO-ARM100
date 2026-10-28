"""Generate Game 1 demonstrations in simulation and save them as LeRobot
datasets, in the same format `lerobot-record` writes on the real arms.

Two layouts:

  per_arm (default)  one dataset per arm (<repo-id>_left, <repo-id>_right).
                     Each successful pick-and-place is one episode, with an
                     instruction such as "pick up the large blue hex prism and
                     place it in the yellow bin". Matches the plan of running
                     one SmolVLA policy per arm. Cameras: top, wrist.
  bimanual           one dataset; each whole sorting run is one episode with
                     12 joints and all four cameras. Use for two-arm policies.

Failed attempts are not saved, so the policy only sees good demonstrations.

    python -m competition.sim.generate_dataset --round colour_sort --runs 20 \\
        --repo-id local/sim_colour_sort --root competition/sim/out/datasets

Rendering is the slow part; on a machine with a GPU use MUJOCO_GL=egl.
"""

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

from . import layout, units
from .env import CONTROL_HZ, SIDES, Game1Env
from .expert import Game1Expert, run_episode

ROUND_TASK = {
    "colour_sort": "sort every object into the bin of its own colour",
    "shape_sort": "sort the objects by shape: cubes red, cylinders blue, dodecahedra green, hex prisms yellow",
    "dynamic_sort": "sort the objects by shape while the bins and objects are moved",
}


def _image_feature(h, w):
    return {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]}


def _create(repo_id, root, features, robot_type):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        shutil.rmtree(root)
    return LeRobotDataset.create(repo_id=repo_id, fps=CONTROL_HZ, features=features, root=root,
                                 robot_type=robot_type, use_videos=True)


class Judge:
    """Round 3: every few seconds, swap the two bins on one side, shift a bin
    a few cm, or slide an object in the source area, as the judges will
    (rules 5.8.1.47-48). Bins stay where the arm that serves them can reach."""

    def __init__(self, rng, every_s=(6.0, 12.0)):
        self.rng, self.every = rng, every_s
        self.next_t = rng.uniform(*every_s)

    def __call__(self, env: Game1Env):
        if env.time < self.next_t:
            return
        self.next_t = env.time + self.rng.uniform(*self.every)
        action = self.rng.choice(["swap", "shift", "nudge"])
        if action == "swap":
            sign = self.rng.choice([-1, 1])
            a, b = [c for c in layout.COLORS if np.sign(env.bin_position(c)[0]) == sign]
            env.move_bins({a: env.bin_position(b)[:2], b: env.bin_position(a)[:2]})
        elif action == "shift":
            color = self.rng.choice(list(layout.COLORS))
            env.move_bins({color: env.bin_position(color)[:2] + self.rng.uniform(-0.025, 0.025, size=2)})
        else:
            loose = [o.name for o in env.objects if env.bin_of(o.name) is None
                     and 0 < env.object_pose(o.name)[0][2] < 0.06]
            if loose:
                env.nudge_object(self.rng.choice(loose), self.rng.uniform(-0.03, 0.03, size=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", default="colour_sort", choices=list(ROUND_TASK))
    ap.add_argument("--runs", type=int, default=10, help="full sorting runs to simulate")
    ap.add_argument("--layout", default="per_arm", choices=["per_arm", "bimanual"])
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--root", type=Path, default=Path(__file__).parent / "out" / "datasets")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--time-limit", type=float, default=300.0, help="seconds per run (rules: ~5 min)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-randomization", action="store_true")
    args = ap.parse_args()

    repo_id = args.repo_id or f"local/sim_{args.round}"
    h, w = args.height, args.width
    if args.layout == "per_arm":
        cams = {"top": "top", "wrist": None}
        datasets = {}
        for side in SIDES:
            features = {
                "observation.state": {"dtype": "float32", "shape": (6,), "names": units.feature_names()},
                "action": {"dtype": "float32", "shape": (6,), "names": units.feature_names()},
                **{f"observation.images.{c}": _image_feature(h, w) for c in cams},
            }
            name = f"{repo_id}_{side}"
            datasets[side] = _create(name, args.root / name.replace("/", "__"), features, "so101_follower")
    else:
        names = [n for s in SIDES for n in units.feature_names(f"{s}_")]
        features = {
            "observation.state": {"dtype": "float32", "shape": (12,), "names": names},
            "action": {"dtype": "float32", "shape": (12,), "names": names},
            **{f"observation.images.{c}": _image_feature(h, w) for c in ("top", "side", "left_wrist", "right_wrist")},
        }
        datasets = {"both": _create(repo_id, args.root / repo_id.replace("/", "__"), features, "bi_so_follower")}

    cameras = ("top", "left_wrist", "right_wrist") if args.layout == "per_arm" else ("top", "side", "left_wrist", "right_wrist")
    env = Game1Env(args.round, seed=args.seed, image_size=(h, w), cameras=cameras,
                   domain_randomization=not args.no_randomization)
    saved = {k: 0 for k in datasets}
    for run in range(args.runs):
        env.reset(seed=args.seed + run)
        expert = Game1Expert(env)
        buffers = {s: [] for s in SIDES}
        whole_run = []

        def on_step(obs, action, segments):
            if args.layout == "bimanual":
                whole_run.append({
                    "observation.state": obs["state"], "action": action.astype(np.float32),
                    **{f"observation.images.{c}": obs["images"][c] for c in cameras},
                    "task": ROUND_TASK[args.round]})
                return
            for i, side in enumerate(SIDES):
                if segments[side] is None:
                    continue
                buffers[side].append({
                    "observation.state": obs["state"][6 * i:6 * i + 6],
                    "action": action[6 * i:6 * i + 6].astype(np.float32),
                    "observation.images.top": obs["images"]["top"],
                    "observation.images.wrist": obs["images"][f"{side}_wrist"],
                    "task": segments[side][0].text})

        def on_task_end(side, task, seg, success):
            if args.layout == "per_arm":
                frames, buffers[side] = buffers[side], []
                if success and len(frames) > 10:
                    for frame in frames:
                        datasets[side].add_frame(frame)
                    datasets[side].save_episode()
                    saved[side] += 1

        judge = Judge(np.random.default_rng(args.seed + run)) if args.round == "dynamic_sort" else None
        result = run_episode(env, expert, time_limit=args.time_limit, on_step=on_step,
                             on_task_end=on_task_end, judge=judge)
        if args.layout == "bimanual" and result["score"] > 0:
            for frame in whole_run:
                datasets["both"].add_frame(frame)
            datasets["both"].save_episode()
            saved["both"] += 1
        print(f"run {run + 1}/{args.runs}: {result}  episodes saved so far: {saved}", flush=True)

    for ds in datasets.values():
        if hasattr(ds, "finalize"):
            ds.finalize()
    print("datasets written under", args.root)


if __name__ == "__main__":
    main()
