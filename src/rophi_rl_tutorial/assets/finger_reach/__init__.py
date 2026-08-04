"""Finger-reach assets: Flexiv arm + RH-5 only (no work-cell fixture)."""

from rophi_rl_tutorial.assets.finger_reach.robot import (
    ARM_JOINT_NAMES,
    ARM_Q_HOME,
    HAND_JOINT_NAMES,
    HAND_Q_OPEN,
    INDEX_TIP_SITE,
    get_finger_reach_robot_cfg,
)

__all__ = [
    "ARM_JOINT_NAMES",
    "ARM_Q_HOME",
    "HAND_JOINT_NAMES",
    "HAND_Q_OPEN",
    "INDEX_TIP_SITE",
    "get_finger_reach_robot_cfg",
]
