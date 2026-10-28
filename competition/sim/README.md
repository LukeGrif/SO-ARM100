# Competition simulator (Robot Arm AI Challenge, Game 1)

A MuJoCo copy of the competition workspace for the IEEE Robot Championship
2026 Robot Arm AI Challenge (rules v0.5, section 5). It has:

- two SO-101 followers 500 mm apart on a 1.7 x 1.2 m table
- the four colour bins and the 32 Game 1 objects (4 colours x 4 shapes x 2 sizes)
- top, side and wrist cameras
- the scoring rules for all three Game 1 rounds

A scripted demonstrator sorts the objects with both arms and records what it
does as **LeRobot datasets in the same format `lerobot-record` writes on the
real arms**: same joint names, same units (degrees, gripper 0–100), same
camera layout. You can train SmolVLA / ACT on that data with `lerobot-train`.

## What this is and isn't

A policy trained **only** on this simulation will not work straight away on
the real arms. The rendered images don't look like your webcams, and the
servos, friction and gripper behave a bit differently. That gap is normal
and is why the data recipe below mixes sim and real.

The simulator is good for:

1. **Lots of extra training data.** Use it to pre-train or co-train, then
   fine-tune on 50–100 real demos per task. That needs far fewer real demos
   than training from scratch.
2. **Testing your pipeline before the arms are built.** Check dataset
   format, training, and the Game 1 controller logic (which object next,
   which arm, handover, Round 3 bin moves) without hardware.
3. **Planning the layout.** Check where each arm reaches and how the
   source area and bins fit. Measured here: an SO-101 reaches about
   0.36 m from its base at table height, so the far middle of the table
   is out of reach of both arms.

Training data from a scripted demonstrator is allowed: the rules accept
simulation as a training method (5.4.6.3). The deployed policy must still
work from live camera images (5.4.3), and it does: the IK in `ik.py` only
generates demonstrations and never runs at competition time.

## Setup

```bash
pip install mujoco pillow 'lerobot[dataset]'
```

On a machine with an NVIDIA GPU, rendering uses EGL by default. Without a
GPU, set `MUJOCO_GL=osmesa`; it works but is slow.

## Use it

Watch the demonstrator (writes `competition/sim/out/preview.gif`):

```bash
python -m competition.sim.preview --round colour_sort --seconds 60
```

Generate per-arm datasets for Round 1 (one episode per successful pick-and-place,
each with an instruction like "pick up the large blue hex prism and place it in the
blue bin"):

```bash
python -m competition.sim.generate_dataset --round colour_sort --runs 20 --repo-id local/sim_colour_sort
```

Rounds 2 and 3 use `--round shape_sort` and `--round dynamic_sort`. For a
two-arm dataset (12 joints, 4 cameras, one episode per full run) add
`--layout bimanual`.

Train on it (as with any LeRobot dataset; `--dataset.root` points at the local folder):

```bash
lerobot-train --policy.path=lerobot/smolvla_base --dataset.repo_id=local/sim_colour_sort_left --dataset.root=competition/sim/out/datasets/local__sim_colour_sort_left --batch_size=64 --steps=20000 --output_dir=outputs/smolvla_sim_left
```

## Before you trust the data: match sim to your real setup

1. **Joint directions.** As soon as one real follower is calibrated, record a
   short episode with `lerobot-record` while moving each joint, then:

   ```bash
   python -m competition.sim.check_real_vs_sim --repo-id <you>/real_test --episode 0 --real-camera front --arm left
   ```

   This replays the real joint states in the sim next to the real video.
   If a joint turns the wrong way, flip it in `units.JOINT_SIGN`; if it is
   offset, set `units.JOINT_OFFSET_DEG`.
2. **Camera positions.** Measure your real top and side cameras and put
   their positions in `layout.TOP_CAMERA` / `layout.SIDE_CAMERA`. Keep the
   real cameras fixed afterwards. Policies are very sensitive to this.
3. **Official sizes and layout.** Object sizes, bin size and bin and source
   positions are placeholders (`TODO` in `layout.py`). Update them when the
   organisers publish the STL files and layouts.
4. **Real data.** Fine-tune on real demos recorded on your real setup. A
   good mix is roughly 1 real episode to every 3–5 sim episodes when
   co-training, and real-only for the final fine-tune.

## Files

| File | What it does |
|---|---|
| `layout.py` | Competition dimensions, objects, points, round rules, camera placement |
| `scene.py` | Builds the MuJoCo scene from the repo's SO-101 model (`Simulation/SO101`) |
| `env.py` | Game 1 environment: reset with random object placement, step with LeRobot-unit actions, scoring |
| `units.py` | Sim radians ↔ LeRobot degrees / gripper 0–100 |
| `ik.py` | Inverse kinematics for the demonstrator (data generation only) |
| `expert.py` | Scripted two-arm demonstrator: picks the highest-value reachable object, uses the handover spot when an object's bin belongs to the other arm, keeps the arms out of each other's way |
| `generate_dataset.py` | Runs the demonstrator and writes LeRobot datasets; includes the Round 3 "judge" that moves bins and objects |
| `preview.py` | GIF of a demonstrator run |
| `check_real_vs_sim.py` | Replays real recordings in the sim to check the joint mapping |

## Known limitations

- The demonstrator sorts roughly half of the 32 objects within 5 minutes.
  Picks fail mostly at the edge of reach and when objects are tightly packed.
  Failed attempts aren't saved, so the data stays clean, but coverage of
  hard placements is thinner.
- The jaws collide using simple box pads instead of the real finger shape,
  and objects are rigid with a single friction value.
- Game 2 (rice, cloth, bubble wrap) and the tower final aren't simulated.
  Deformables are slow to simulate and hard to make realistic in the time
  left, so use real demonstrations for those.
- Running one policy per arm at the same time on the real robot needs a
  small custom runner. `lerobot-record --policy.path` drives one policy at
  a time.
