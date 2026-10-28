"""Watch the scripted demonstrator sort objects: renders a GIF of one run
from the top camera and prints the score.

    python -m competition.sim.preview --round shape_sort --seconds 60
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image

from .env import Game1Env
from .expert import Game1Expert, run_episode
from .generate_dataset import Judge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--round", default="colour_sort", choices=["colour_sort", "shape_sort", "dynamic_sort"])
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--camera", default="top")
    ap.add_argument("--every", type=int, default=6, help="keep every n-th frame (30 fps sim)")
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "out" / "preview.gif")
    args = ap.parse_args()

    env = Game1Env(args.round, seed=args.seed, image_size=(360, 480), cameras=(args.camera,))
    env.reset()
    frames, tick = [], [0]

    def on_step(obs, action, segments):
        # Render only the frames we keep.
        if tick[0] % args.every == 0:
            env.renderer.update_scene(env.data, camera=args.camera)
            frames.append(Image.fromarray(env.renderer.render()))
        tick[0] += 1

    judge = Judge(np.random.default_rng(args.seed)) if args.round == "dynamic_sort" else None
    result = run_episode(env, Game1Expert(env), time_limit=args.seconds, on_step=on_step, judge=judge,
                         render=False)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=1000 * args.every // 30, loop=0)
    print(result)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
