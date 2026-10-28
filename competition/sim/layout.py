"""Competition geometry, object set and scoring, taken from the IEEE Robot
Championship 2026 rules (v0.5, section 5). Everything the organisers have not
published yet (object sizes, bin size and positions) is marked TODO and is a
placeholder: update it when the official STL files and layouts come out.

World frame: origin on the table top, midway between the two arm bases.
+x points to the right (seen from the team), +y points away from the team,
+z points up. Units are metres.
"""

from dataclasses import dataclass

# --- Workspace (rules 5.1.3) -------------------------------------------------
TABLE_WIDTH = 1.70  # x
TABLE_DEPTH = 1.20  # y
ARM_SEPARATION = 0.50  # base centre to base centre, +/-10 mm
# TODO: the rules do not say how far the bases sit from the near table edge.
ARM_BASE_FROM_NEAR_EDGE = 0.10
TABLE_Y_MIN = -ARM_BASE_FROM_NEAR_EDGE
TABLE_Y_MAX = TABLE_Y_MIN + TABLE_DEPTH

ARMS = {
    "left": (-ARM_SEPARATION / 2, 0.0),
    "right": (ARM_SEPARATION / 2, 0.0),
}

# --- Game 1 objects (rules 5.8.1.11 - 5.8.1.43) -------------------------------
COLORS = {
    "red": (0.80, 0.10, 0.10, 1.0),
    "blue": (0.10, 0.30, 0.80, 1.0),
    "green": (0.10, 0.60, 0.20, 1.0),
    "yellow": (0.95, 0.80, 0.10, 1.0),
}
SHAPES = ("cube", "cylinder", "dodecahedron", "hex_prism")
# TODO: replace with the nominal sizes published with the official STLs.
# Characteristic size = cube edge, cylinder diameter/height, prism
# across-flats/height, dodecahedron distance between opposite faces.
SIZES = {"small": 0.030, "large": 0.045}
# PLA at 15% infill is roughly 0.4 g/cm^3 for small parts.
OBJECT_DENSITY = 400.0

# Points per correctly sorted object (rules 5.8.1.58). Colour does not matter.
POINTS = {
    ("cube", "small"): 1, ("cylinder", "small"): 2,
    ("dodecahedron", "small"): 3, ("hex_prism", "small"): 5,
    ("cube", "large"): 2, ("cylinder", "large"): 3,
    ("dodecahedron", "large"): 4, ("hex_prism", "large"): 5,
}
COMPLETION_BONUS = 10

# Which bin each object belongs in, per round (rules 5.8.1.34 and 5.8.1.41).
SHAPE_TO_BIN = {"cube": "red", "cylinder": "blue",
                "dodecahedron": "green", "hex_prism": "yellow"}


def target_bin(round_name: str, color: str, shape: str) -> str:
    if round_name == "colour_sort":
        return color
    if round_name in ("shape_sort", "dynamic_sort"):
        return SHAPE_TO_BIN[shape]
    raise ValueError(f"unknown round {round_name!r}")


@dataclass(frozen=True)
class ObjectSpec:
    color: str
    shape: str
    size: str

    @property
    def name(self) -> str:
        return f"{self.size}_{self.color}_{self.shape}"

    @property
    def points(self) -> int:
        return POINTS[(self.shape, self.size)]


# Rounds 1-3 all use one of every colour x shape x size: 32 objects.
GAME1_OBJECTS = tuple(ObjectSpec(c, s, z) for z in SIZES for c in COLORS for s in SHAPES)

# --- Source area and bins -----------------------------------------------------
# An SO-101 reaches about 0.36 m from its base at table height (measured in
# this model, gripper tilted), so the far middle of the table is out of reach
# of both arms. Objects are only placed where at least one arm reaches.
MAX_REACH = 0.36
# TODO: organiser-defined. The source area sits between the arms; each arm
# gets the two bins on its outer side.
SOURCE_AREA = {"x": (-0.24, 0.24), "y": (0.10, 0.40)}
BIN_INNER = 0.13  # TODO: from the official bin STL
BIN_WALL_HEIGHT = 0.06
BIN_WALL = 0.004
DEFAULT_BIN_POSITIONS = {
    "red": (-0.48, 0.20),
    "blue": (-0.36, 0.36),
    "green": (0.36, 0.36),
    "yellow": (0.48, 0.20),
}
# Objects whose bin is only reachable by the other arm are put down here, in
# reach of both arms, and the other arm takes them on.
HANDOVER_XY = (0.0, 0.045)

# --- Cameras (rules 5.1.7) ----------------------------------------------------
# Top camera on the H-rig, side camera on the 25 mm pole. Positions are
# yours to choose at setup: put the numbers of your real mounts here.
TOP_CAMERA = {"pos": (0.0, 0.20, 0.95), "lookat": (0.0, 0.20, 0.0), "fovy": 55}
SIDE_CAMERA = {"pos": (0.80, 0.25, 0.55), "lookat": (0.0, 0.25, 0.05), "fovy": 60}
