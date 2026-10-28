"""Builds the Robot Arm AI Challenge workspace in MuJoCo: the competition table,
two SO-101 follower arms 500 mm apart, the four bins, the Game 1 objects and
the top, side and wrist cameras.

    python -m competition.sim.scene          # writes competition/sim/out/scene_preview.png
"""

from pathlib import Path

import mujoco
import numpy as np

from . import layout

REPO_ROOT = Path(__file__).resolve().parents[2]
ARM_XML = REPO_ROOT / "Simulation" / "SO101" / "so101_new_calib_camera.xml"
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
CAMERAS = ("top", "side", "left_wrist", "right_wrist")

PHI = (1 + 5 ** 0.5) / 2


def _dodecahedron_vertices(face_distance: float) -> np.ndarray:
    """Regular dodecahedron resting on a face, scaled to the given distance
    between opposite faces."""
    v = [(x, y, z) for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)]
    for a in (-1, 1):
        for b in (-1, 1):
            v += [(0, a / PHI, b * PHI), (a / PHI, b * PHI, 0), (a * PHI, 0, b / PHI)]
    v = np.array(v, dtype=float)
    # Face normals point at icosahedron vertices; rotate (0, 1, phi) onto +z.
    n = np.array([0.0, 1.0, PHI]) / np.linalg.norm([0.0, 1.0, PHI])
    axis = np.cross(n, [0, 0, 1.0])
    angle = np.arccos(n[2])
    axis /= np.linalg.norm(axis)
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    rot = np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * k @ k
    v = v @ rot.T
    return v * (face_distance / (v[:, 2].max() - v[:, 2].min()))


def _hex_prism_vertices(across_flats: float, height: float) -> np.ndarray:
    r = across_flats / np.sqrt(3)  # circumradius
    ring = [(r * np.cos(a), r * np.sin(a)) for a in np.arange(6) * np.pi / 3]
    return np.array([(x, y, z) for z in (-height / 2, height / 2) for x, y in ring])


def _camera_quat(pos, lookat, up=(0, 0, 1)) -> np.ndarray:
    """Quaternion for a MuJoCo camera at pos looking at lookat (cameras look along -z)."""
    z = np.asarray(pos, float) - np.asarray(lookat, float)
    z /= np.linalg.norm(z)
    x = np.cross(up, z)
    if np.linalg.norm(x) < 1e-6:
        x = np.cross((0, 1, 0), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.column_stack([x, y, z]).ravel())
    return quat


def _wrist_camera_pose() -> tuple[np.ndarray, np.ndarray]:
    """Pose of the wrist camera lens in the frame of the arm's gripper body.

    The arm model carries the camera mesh but no MuJoCo camera, so we put one
    at the camera body, looking just past the jaw tips."""
    model = mujoco.MjModel.from_xml_path(str(ARM_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body = data.body("gripper")
    cam = data.body("wrist_camera").xpos
    tip = data.site("gripperframe").xpos
    body_rot = body.xmat.reshape(3, 3)
    # Up in the image = away from the arm, i.e. the gripper body's -x roughly.
    up = body_rot @ np.array([0.0, 0.0, 1.0])
    # Put the lens just in front of the camera housing so it doesn't see the inside of it.
    look = tip + (tip - cam) * 0.3
    cam = cam + (look - cam) / np.linalg.norm(look - cam) * 0.02
    world_quat = _camera_quat(cam, look, up)
    world_rot = np.zeros(9)
    mujoco.mju_quat2Mat(world_rot, world_quat)
    local_rot = body_rot.T @ world_rot.reshape(3, 3)
    local_quat = np.zeros(4)
    mujoco.mju_mat2Quat(local_quat, local_rot.ravel())
    return body_rot.T @ (cam - body.xpos), local_quat


def _finger_pads() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Box colliders for the two fingers, as (pos, quat, half-size) in the
    frame of the body that carries each finger.

    MuJoCo collides meshes through their convex hulls, and the hull of the
    fixed jaw part fills the space between the fingers, so objects could never
    get between the jaws. The pads replace those meshes for contact. Extents
    were measured from the meshes in the gripperframe site frame (x = approach
    direction, z = closing direction) with the gripper joint at 0 rad."""
    model = mujoco.MjModel.from_xml_path(str(ARM_XML))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    site_rot = data.site("gripperframe").xmat.reshape(3, 3)
    site_pos = data.site("gripperframe").xpos
    half = np.array([0.036, 0.008, 0.006])
    pads = {
        "gripper": np.array([-0.012, 0.0, -0.006]),  # fixed finger, inner face at z = 0
        "moving_jaw_so101_v1": np.array([-0.012, 0.0, 0.022]),  # inner face at z = 0.016
    }
    out = {}
    for body_name, centre in pads.items():
        body = data.body(body_name)
        body_rot = body.xmat.reshape(3, 3)
        pos = body_rot.T @ (site_pos + site_rot @ centre - body.xpos)
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, (body_rot.T @ site_rot).ravel())
        out[body_name] = (pos, quat, half)
    return out


def _add_bin(spec: mujoco.MjSpec, color: str, xy) -> None:
    rgba = layout.COLORS[color]
    body = spec.worldbody.add_body(name=f"bin_{color}", pos=[xy[0], xy[1], 0], mocap=True)
    half, h, t = layout.BIN_INNER / 2, layout.BIN_WALL_HEIGHT, layout.BIN_WALL
    body.add_geom(name=f"bin_{color}_floor", type=mujoco.mjtGeom.mjGEOM_BOX,
                  size=[half + t, half + t, t / 2], pos=[0, 0, t / 2], rgba=rgba)
    for i, (dx, dy) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        size = [t / 2, half + t, h / 2] if dx else [half + t, t / 2, h / 2]
        body.add_geom(name=f"bin_{color}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size,
                      pos=[dx * (half + t / 2), dy * (half + t / 2), h / 2], rgba=rgba)


def _add_object(spec: mujoco.MjSpec, obj: layout.ObjectSpec, index: int) -> None:
    s = layout.SIZES[obj.size]
    # Parked off the table; the environment places objects on reset.
    body = spec.worldbody.add_body(name=obj.name, pos=[-1.5 + 0.06 * index, -0.6, s])
    body.add_freejoint(name=f"{obj.name}_free")
    geom = dict(name=f"{obj.name}_geom", rgba=layout.COLORS[obj.color],
                density=layout.OBJECT_DENSITY, friction=[1.2, 0.02, 0.002],
                condim=4, solref=[0.01, 1])
    if obj.shape == "cube":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[s / 2] * 3, **geom)
    elif obj.shape == "cylinder":
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[s / 2, s / 2, 0], **geom)
    else:
        verts = (_dodecahedron_vertices(s) if obj.shape == "dodecahedron"
                 else _hex_prism_vertices(s, s))
        mesh = f"{obj.shape}_{obj.size}"
        if spec.mesh(mesh) is None:
            spec.add_mesh(name=mesh, uservert=verts.ravel().tolist())
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, **geom)


def build_spec(objects=layout.GAME1_OBJECTS) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "robot_arm_ai_challenge"
    spec.option.timestep = 0.002
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 960
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.headlight.diffuse = [0.4, 0.4, 0.4]

    world = spec.worldbody
    world.add_light(name="ceiling", pos=[0, 0.4, 2.5], dir=[0, 0, -1],
                    diffuse=[0.7, 0.7, 0.7], castshadow=True)
    world.add_light(name="fill", pos=[-1.0, -0.8, 1.5], dir=[0.5, 0.5, -0.7],
                    diffuse=[0.3, 0.3, 0.3], castshadow=False)
    spec.add_texture(name="floor", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=256, height=256,
                     rgb1=[0.25, 0.25, 0.28], rgb2=[0.2, 0.2, 0.23])
    spec.add_material(name="floor", textures=["", "floor"], texrepeat=[8, 8])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05],
                   pos=[0, 0, -0.75], material="floor")

    # Table top: its upper surface is z = 0.
    y_mid = (layout.TABLE_Y_MIN + layout.TABLE_Y_MAX) / 2
    world.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[layout.TABLE_WIDTH / 2, layout.TABLE_DEPTH / 2, 0.02],
                   pos=[0, y_mid, -0.02], rgba=[0.86, 0.86, 0.84, 1], friction=[1, 0.01, 0.001])
    # Source area outline (black tape).
    (x0, x1), (y0, y1) = layout.SOURCE_AREA["x"], layout.SOURCE_AREA["y"]
    for i, (cx, cy, sx, sy) in enumerate((
            ((x0 + x1) / 2, y0, (x1 - x0) / 2, 0.009), ((x0 + x1) / 2, y1, (x1 - x0) / 2, 0.009),
            (x0, (y0 + y1) / 2, 0.009, (y1 - y0) / 2), (x1, (y0 + y1) / 2, 0.009, (y1 - y0) / 2))):
        world.add_geom(name=f"tape{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[sx, sy, 0.0005],
                       pos=[cx, cy, 0.0005], rgba=[0.05, 0.05, 0.05, 1], contype=0, conaffinity=0)

    hx, hy = layout.HANDOVER_XY
    for i, (dx, dy, sx, sy) in enumerate(((0, -1, 1, 0), (0, 1, 1, 0), (-1, 0, 0, 1), (1, 0, 0, 1))):
        world.add_geom(name=f"handover_tape{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[0.035 * sx + 0.005 * sy, 0.035 * sy + 0.005 * sx, 0.0005],
                       pos=[hx + 0.035 * dx, hy + 0.035 * dy, 0.0005], rgba=[0.05, 0.05, 0.05, 1],
                       contype=0, conaffinity=0)

    cam_pos, cam_quat = _wrist_camera_pose()
    pads = _finger_pads()
    for side, (x, y) in layout.ARMS.items():
        arm = mujoco.MjSpec.from_file(str(ARM_XML))
        arm.meshdir = str(ARM_XML.parent / "assets")
        arm.body("gripper").add_camera(name="wrist", pos=cam_pos, quat=cam_quat, fovy=75)
        for body_name, (pos, quat, half) in pads.items():
            body = arm.body(body_name)
            for geom in body.geoms:
                if geom.meshname in ("wrist_roll_follower_so101_v1", "moving_jaw_so101_v1"):
                    geom.contype = geom.conaffinity = 0
            body.add_geom(name=f"{body_name}_pad", type=mujoco.mjtGeom.mjGEOM_BOX, size=half,
                          pos=pos, quat=quat, friction=[1.5, 0.02, 0.002], condim=4,
                          rgba=[0.2, 0.2, 0.2, 0.0], group=3)
        # The arm reaches along its +x at zero pan; turn it to face +y.
        frame = world.add_frame(pos=[x, y, 0], quat=[np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
        spec.attach(arm, prefix=f"{side}_", frame=frame)

    for name, cfg in (("top", layout.TOP_CAMERA), ("side", layout.SIDE_CAMERA)):
        world.add_camera(name=name, pos=cfg["pos"], fovy=cfg["fovy"],
                         quat=_camera_quat(cfg["pos"], cfg["lookat"], up=(0, 1, 0) if name == "top" else (0, 0, 1)))

    for color, xy in layout.DEFAULT_BIN_POSITIONS.items():
        _add_bin(spec, color, xy)
    for i, obj in enumerate(objects):
        _add_object(spec, obj, i)
    return spec


def build_model(objects=layout.GAME1_OBJECTS) -> mujoco.MjModel:
    return build_spec(objects).compile()


if __name__ == "__main__":
    import os

    os.environ.setdefault("MUJOCO_GL", "egl")
    from PIL import Image

    model = build_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    renderer = mujoco.Renderer(model, 480, 640)
    tiles = []
    for cam in CAMERAS:
        renderer.update_scene(data, camera=cam)
        tiles.append(renderer.render())
    Image.fromarray(np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:])])).save(out / "scene_preview.png")
    print(f"bodies={model.nbody} actuators={model.nu} cameras={[model.camera(i).name for i in range(model.ncam)]}")
