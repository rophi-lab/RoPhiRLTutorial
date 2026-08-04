"""Finger-reach robot: Flexiv Rizon 4 + RH-5 only (no work-cell fixture).

Loads the shared MJCF from the ``rophi-tutorial`` submodule, then **deletes** the
world-parented ``enclosure`` body (and its mesh) so the free-space example has
just the arm and hand. Actuators / sites / geom names are defined here — this
module does **not** import grasping assets.
"""

from __future__ import annotations

import os
import re

import mujoco
from mjlab.actuator import BuiltinMotorActuatorCfg, BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from rophi_rl_tutorial import ROPHI_TUTORIAL_PATH

##
# MJCF.
##

RIZON4_HAND_XML = (
    ROPHI_TUTORIAL_PATH / "assets" / "scene" / "flexiv_arm" / "Rizon4_hand_robot.xml"
)
assert RIZON4_HAND_XML.exists(), (
    f"Robot MJCF not found at {RIZON4_HAND_XML}. Did you init the submodule? "
    "Run: git submodule update --init"
)

##
# Joint / site names.
##

ARM_JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))

FINGERS = ("1_thumb", "2_index", "3_middle", "4_ring", "5_little")

HAND_JOINT_NAMES = tuple(
    f"a{8 + 4 * f + j:02d}_finger_r_joint_{finger}{j + 1}"
    for f, finger in enumerate(FINGERS)
    for j in range(4)
)

INDEX_TIP_SITE = "index_tip_site"
_INDEX_TIP_BODY = "index_tip"

# Palm vs proximal-phalanx overlaps at rest (same exclusions as the hardware scene).
_CONTACT_EXCLUDES = tuple(("palm", f"finger_r_link_{finger}2") for finger in FINGERS)

##
# Spec editing.
##


def _is_collision_geom(geom: mujoco.MjsGeom) -> bool:
    return bool(geom.contype) or bool(geom.conaffinity)


def _name_geoms(spec: mujoco.MjSpec) -> None:
    """Deterministic ``<body>_collision`` / ``<body>_visual`` names for CollisionCfg."""
    for body in spec.bodies:
        collision_idx = 0
        visual_idx = 0
        for geom in body.geoms:
            if _is_collision_geom(geom):
                suffix = "" if collision_idx == 0 else f"_{collision_idx}"
                geom.name = f"{body.name}_collision{suffix}"
                collision_idx += 1
            else:
                suffix = "" if visual_idx == 0 else f"_{visual_idx}"
                geom.name = f"{body.name}_visual{suffix}"
                visual_idx += 1


def _strip_fixture(spec: mujoco.MjSpec) -> None:
    """Remove the work-cell enclosure body + mesh (fixture, not part of the robot)."""
    try:
        spec.delete(spec.body("enclosure"))
    except Exception:
        pass
    try:
        spec.delete(spec.mesh("enclosure_decimated"))
    except Exception:
        pass


def _add_sites(spec: mujoco.MjSpec) -> None:
    """Index fingertip site used by the tip-velocity MDP."""
    spec.body(_INDEX_TIP_BODY).add_site(name=INDEX_TIP_SITE, pos=(0.0, 0.0, 0.0))


def get_spec() -> mujoco.MjSpec:
    """Load MJCF, strip enclosure, name geoms, add tip site + contact excludes."""
    spec = mujoco.MjSpec.from_file(str(RIZON4_HAND_XML))
    _strip_fixture(spec)
    _name_geoms(spec)
    _add_sites(spec)
    for body1, body2 in _CONTACT_EXCLUDES:
        spec.add_exclude(
            name=f"exclude_{body1}_{body2}", bodyname1=body1, bodyname2=body2
        )
    return spec


##
# Actuators.
##

_ARM_EFFORT_LIMITS = {
    "joint1": 123.0,
    "joint2": 123.0,
    "joint3": 64.0,
    "joint4": 64.0,
    "joint5": 39.0,
    "joint6": 39.0,
    "joint7": 39.0,
}

HAND_EFFORT_LIMIT = float(os.environ.get("ROPHI_HAND_EFFORT", "1.0"))
HAND_ARMATURE = 0.002418

ARM_ACTUATORS = tuple(
    BuiltinMotorActuatorCfg(
        target_names_expr=(name,),
        effort_limit=effort_limit,
    )
    for name, effort_limit in _ARM_EFFORT_LIMITS.items()
)

_HAND_GAINS_BY_LEVEL = {
    1: (6.0, 0.25),
    2: (5.0, 0.20),
    3: (3.0, 0.12),
    4: (2.0, 0.08),
}


def _joint_level(joint_name: str) -> int:
    match = re.search(r"(\d)$", joint_name)
    assert match is not None, f"Unexpected hand joint name: {joint_name}"
    return int(match.group(1))


HAND_ACTUATORS = tuple(
    BuiltinPositionActuatorCfg(
        target_names_expr=(name,),
        stiffness=_HAND_GAINS_BY_LEVEL[_joint_level(name)][0],
        damping=_HAND_GAINS_BY_LEVEL[_joint_level(name)][1],
        effort_limit=HAND_EFFORT_LIMIT,
        armature=HAND_ARMATURE,
    )
    for name in HAND_JOINT_NAMES
)

ARTICULATION = EntityArticulationInfoCfg(
    actuators=(*ARM_ACTUATORS, *HAND_ACTUATORS),
    soft_joint_pos_limit_factor=1.0,
)

##
# Initial state (hardware home: arm pose + open hand).
##

HAND_Q_OPEN = dict.fromkeys(HAND_JOINT_NAMES, 0.0)

ARM_Q_HOME = {
    "joint1": -0.4363323,
    "joint2": -0.6981317,
    "joint3": 0.6108652,
    "joint4": 1.7453293,
    "joint5": 0.8726646,
    "joint6": 0.5235988,
    "joint7": 1.0471976,
}

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={**ARM_Q_HOME, **HAND_Q_OPEN},
    joint_vel={".*": 0.0},
)

##
# Collisions.
##

_GRASP_SURFACES = r"(palm|finger_r_link_\d_[a-z]+[234])_collision"

COLLISION = CollisionCfg(
    geom_names_expr=(r".*_collision",),
    condim={_GRASP_SURFACES: 6, r".*_collision": 3},
    friction={_GRASP_SURFACES: (1.0, 5e-3, 5e-4), r".*_collision": (0.6,)},
    solref={_GRASP_SURFACES: (0.01, 1.0)},
    priority={_GRASP_SURFACES: 1},
)


def get_finger_reach_robot_cfg() -> EntityCfg:
    """Rizon 4 + RH-5 only: torque arm, position hand, no enclosure fixture."""
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        collisions=(COLLISION,),
        spec_fn=get_spec,
        articulation=ARTICULATION,
    )


__all__ = [
    "ARM_JOINT_NAMES",
    "ARM_Q_HOME",
    "HAND_JOINT_NAMES",
    "HAND_Q_OPEN",
    "INDEX_TIP_SITE",
    "RIZON4_HAND_XML",
    "get_finger_reach_robot_cfg",
    "get_spec",
]
