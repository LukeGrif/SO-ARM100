"""Check that the simulated arm moves like your real arm before you trust
simulated data. Run this as soon as one real follower is calibrated.

It replays the joint states of an episode recorded on the real arm (any
`lerobot-record` dataset recorded with `--robot.use_degrees=true`) in the
simulator and renders it next to the real camera frames, so you can see
whether every joint turns the same way and by the same amount.

    python -m competition.sim.check_real_vs_sim --repo-id you/real_test --episode 0 \\
        --real-camera front --arm left --out competition/sim/out/compare.gif

If a joint moves the opposite way, flip its sign in units.JOINT_SIGN; if it
is offset, set units.JOINT_OFFSET_DEG. Then run this again.

Also useful the other way round: --pose "0,0,0,0,0,50" renders one pose
(LeRobot degrees, gripper 0-100) so you can put the real arm in it with
`lerobot-teleoperate` and compare by eye.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image

from . import units
from .env import Game1Env


def render_pose(env: Game1Env, side: str, values, camera: mujoco.MjvCamera, renderer) -> np.ndarray:
    env.data.qpos[env.qpos_adr[side]] = units.real_to_sim(values)
    mujoco.mj_forward(env.model, env.data)
    renderer.update_scene(env.data, camera)
    return renderer.render().copy()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", help="LeRobot dataset recorded on the real arm")
    ap.add_argument("--root", type=Path, default=None, help="local dataset folder, if not on the Hub cache")
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--real-camera", default=None, help="camera key to show next to the sim, e.g. front")
    ap.add_argument("--arm", default="left", choices=["left", "right"])
    ap.add_argument("--state-slice", default="0:6", help="which state entries are this arm (bimanual: 6:12 for right)")
    ap.add_argument("--pose", default=None, help="render one pose instead, e.g. '0,0,0,0,0,50'")
    ap.add_argument("--every", type=int, default=3, help="use every n-th frame")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "compare.gif")
    args = ap.parse_args()

    env = Game1Env(image_size=None, domain_randomization=False)
    env.reset(seed=0)
    renderer = mujoco.Renderer(env.model, 360, 480)
    cam = mujoco.MjvCamera()
    base = np.array([-0.25 if args.arm == "left" else 0.25, 0.1, 0.12])
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = base, 0.7, 120, -20
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if args.pose:
        img = render_pose(env, args.arm, np.array([float(v) for v in args.pose.split(",")]), cam, renderer)
        Image.fromarray(img).save(args.out.with_suffix(".png"))
        print("wrote", args.out.with_suffix(".png"))
        return

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset(args.repo_id, root=args.root, episodes=[args.episode])
    lo, hi = (int(v) for v in args.state_slice.split(":"))
    frames = []
    for i in range(0, len(ds), args.every):
        item = ds[i]
        state = item["observation.state"].numpy()[lo:hi]
        sim = render_pose(env, args.arm, state, cam, renderer)
        if args.real_camera:
            real = (item[f"observation.images.{args.real_camera}"].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            real = np.array(Image.fromarray(real).resize((sim.shape[1], sim.shape[0])))
            sim = np.hstack([real, sim])
        frames.append(Image.fromarray(sim))
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=1000 * args.every // 30, loop=0)
    print(f"wrote {args.out} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
