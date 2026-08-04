"""High-level grasp-phase arm controller.

The supervisory layer above :class:`~rophi_rl_tutorial.controllers.grasping.arm_impedance.ArmImpedanceController`.
It watches the object, watches the ``grasp_score`` published by the hand policy, and emits a
palm setpoint. It commands nothing directly -- all authority over the arm stays with the
impedance controller, and the hand is never referenced. The only coupling between hand and
arm is the scalar grasp score, which is what keeps the reflex arm-agnostic.

Phase logic
-----------
Two phases selected by a Schmitt trigger on the grasp score::

    REACH  score <= score_low   setpoint = object position + pre_grasp_offset
    LIFT   score >= score_high  setpoint = the fixed retreat pose

Between the thresholds the phase is *latched* at its previous value, so a score hovering
around 0.5 cannot chatter the arm between reaching and lifting. The latch is reversible: if
the grasp fails and the score collapses, the arm returns to reaching. Reset forces REACH.

Setpoint radius clamp
---------------------
On a phase flip the target jumps tens of centimetres at once. Fed to the impedance
controller directly that is a step input, and the resulting torque transient saturates the
actuators -- which both looks wrong and destroys the null-space decoupling, since clipped
torque no longer respects the projector. ``max_setpoint_radius`` projects the commanded
setpoint onto a sphere around the *current* palm point, so the arm pursues a target that is
never more than that far ahead. It converts a step into a bounded-rate ramp and caps the
spring force at roughly ``stiffness * radius``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Literal

import mujoco
import numpy as np

from rophi_rl_tutorial.controllers.grasping.arm_impedance import ArmImpedanceController, quat_from_rpy

__all__ = ["GraspPhase", "GraspPhaseCfg", "GraspPhaseController"]


class GraspPhase(IntEnum):
    REACH = 0
    LIFT = 1


@dataclass
class GraspPhaseCfg:
    """Fields are mutable at runtime so the offset and thresholds can be tuned live."""

    pre_grasp_offset: tuple[float, float, float] = (0.0, 0.0, -0.01)
    """Palm setpoint offset from the object centre while reaching. Frame set below.

    In the palm frame, +z is the fingertip direction, so a small positive z puts the grasp
    point just past the object and leaves the object sitting inside the finger cage, near
    the palm, rather than out at the fingertips.
    """

    offset_frame: Literal["palm", "world"] = "palm"
    """Frame the offset is expressed in.

    ``palm`` keeps its meaning as the wrist attitude changes: -z is always "back along the
    approach direction". A world-axis offset only reads naturally when the palm points
    straight down -- at the hold attitude, which is 63 deg off vertical, the fingertips are
    displaced ~0.09 m sideways from the grasp point, so a world (0, 0, z) offset lands the
    fingers beside the object rather than around it. ``world`` reproduces the IsaacLab
    ``pre_grasp_target_offset_env`` convention.
    """

    approach_rpy: tuple[float, float, float] | None = None
    """Palm orientation while approaching and at the pre-grasp point.

    ``None`` -- the default -- reuses the hold orientation, so the wrist keeps one attitude
    through approach, grasp and lift. Rotating between phases would swing an already-grasped
    object about the palm and add a wrench the impedance controller has to fight at exactly
    the moment the grasp is weakest.

    Set explicitly only to approach at a different attitude than the object is carried at.
    """

    lift_pos: tuple[float, float, float] | None = None
    """Retreat position. ``None`` captures the palm pose held at :meth:`reset`."""

    lift_rpy: tuple[float, float, float] | None = None
    """Retreat orientation. ``None`` captures the palm pose held at :meth:`reset`."""

    approach_height: float = 0.25
    """Height above the object held while still moving laterally, before descending.

    Without this the arm pulls straight from its home pose to the pre-grasp point, and the
    hand -- whose fingertips lead the palm by ~12 cm in the open pose -- drags across the
    table and bulldozes the object out of reach. Since the setpoint tracks the object, that
    is a runaway: the arm chases the object it is pushing. Approaching above and descending
    only once roughly overhead keeps the fingers clear.
    """

    descend_xy_tolerance: float = 0.04
    """Horizontal distance to the pre-grasp point at which the descent is allowed to start."""

    residual_scale: float = 0.05
    """Per-axis bound (m) on the policy's XYZ setpoint residual."""

    max_setpoint_radius: float = 0.05
    """Cap on how far ahead of the palm the setpoint may sit. 0 disables the clamp."""

    score_low: float = 0.45
    score_high: float = 0.55
    min_setpoint_z: float = 0.0
    """Floor on the commanded height, so a low object cannot drive the palm into the table."""

    def __post_init__(self) -> None:
        if self.score_high < self.score_low:
            raise ValueError("score_high must be >= score_low.")


@dataclass
class GraspPhaseState:
    """Everything the phase controller decided this step, for logging and visualisation."""

    phase: GraspPhase = GraspPhase.REACH
    grasp_score: float = 0.0
    object_pos: np.ndarray = field(default_factory=lambda: np.zeros(3))
    base_target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    commanded_target: np.ndarray = field(default_factory=lambda: np.zeros(3))
    clamped: bool = False
    """True when the radius clamp is active, i.e. the arm is still pursuing a distant goal."""


class GraspPhaseController:
    """Drives an :class:`ArmImpedanceController` from the object pose and a grasp score."""

    def __init__(
        self,
        impedance: ArmImpedanceController,
        model: mujoco.MjModel,
        object_body_name: str = "object",
        cfg: GraspPhaseCfg | None = None,
    ) -> None:
        self.impedance = impedance
        self.cfg = cfg if cfg is not None else GraspPhaseCfg()
        self.object_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
        if self.object_body_id < 0:
            raise ValueError(f"No body named {object_body_name!r} in the model.")
        self.state = GraspPhaseState()
        self._lift_pos = np.zeros(3)
        self._lift_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self._descending = False

    ##
    # Lifecycle.
    ##

    def reset(self, data: mujoco.MjData) -> None:
        """Return to REACH and, if the retreat pose is unset, latch the current palm pose."""
        self.state = GraspPhaseState(phase=GraspPhase.REACH)
        self._descending = False
        if self.cfg.lift_pos is None:
            self._lift_pos = data.site_xpos[self.impedance.site_id].copy()
        else:
            self._lift_pos = np.asarray(self.cfg.lift_pos, dtype=float)
        if self.cfg.lift_rpy is None:
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, data.site_xmat[self.impedance.site_id])
            self._lift_quat = quat
        else:
            self._lift_quat = quat_from_rpy(np.asarray(self.cfg.lift_rpy, dtype=float))

    ##
    # Control.
    ##

    def object_pos(self, data: mujoco.MjData) -> np.ndarray:
        return data.xpos[self.object_body_id]

    def _latch_phase(self, grasp_score: float) -> GraspPhase:
        phase = self.state.phase
        if phase is GraspPhase.REACH and grasp_score >= self.cfg.score_high:
            phase = GraspPhase.LIFT
        elif phase is GraspPhase.LIFT and grasp_score <= self.cfg.score_low:
            phase = GraspPhase.REACH
        return phase

    def _clamp_radius(self, palm_pos: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, bool]:
        if self.cfg.max_setpoint_radius <= 0.0:
            return target, False
        delta = target - palm_pos
        distance = float(np.linalg.norm(delta))
        if distance <= self.cfg.max_setpoint_radius:
            return target, False
        return palm_pos + delta * (self.cfg.max_setpoint_radius / distance), True

    def update(
        self,
        data: mujoco.MjData,
        grasp_score: float,
        residual: np.ndarray | None = None,
    ) -> GraspPhaseState:
        """Advance the phase and write the resulting setpoint to the impedance controller.

        ``grasp_score`` is the hand policy's scalar in [0, 1]. ``residual`` is the policy's
        optional XYZ nudge in [-1, 1] per axis, scaled by ``cfg.residual_scale``.
        """
        cfg = self.cfg
        grasp_score = float(np.clip(grasp_score, 0.0, 1.0))
        phase = self._latch_phase(grasp_score)

        object_pos = self.object_pos(data).copy()
        palm = data.site_xpos[self.impedance.site_id]
        if phase is GraspPhase.REACH:
            target_quat = (
                self._lift_quat
                if cfg.approach_rpy is None
                else quat_from_rpy(np.asarray(cfg.approach_rpy, dtype=float))
            )
            offset = np.asarray(cfg.pre_grasp_offset, dtype=float)
            if cfg.offset_frame == "palm":
                # Rotate by the commanded attitude, not the palm's current one, so the
                # target does not move as the wrist settles onto it.
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, target_quat)
                offset = rot.reshape(3, 3) @ offset
            base_target = object_pos + offset
            # Latch the descent with a 2x release band, so drifting slightly outside the
            # tolerance mid-descent does not yo-yo the arm back up to the hover height.
            xy_error = float(np.linalg.norm(palm[:2] - base_target[:2]))
            if xy_error < cfg.descend_xy_tolerance:
                self._descending = True
            elif xy_error > 2.0 * cfg.descend_xy_tolerance:
                self._descending = False
            if not self._descending:
                base_target = base_target.copy()
                base_target[2] = object_pos[2] + cfg.approach_height
        else:
            self._descending = False
            base_target = self._lift_pos.copy()
            target_quat = self._lift_quat

        target = base_target
        if residual is not None:
            target = target + np.clip(residual, -1.0, 1.0) * cfg.residual_scale

        palm_pos = data.site_xpos[self.impedance.site_id]
        target, clamped = self._clamp_radius(palm_pos, target)
        target[2] = max(target[2], cfg.min_setpoint_z)

        self.impedance.set_target(target, target_quat)
        self.state = GraspPhaseState(
            phase=phase,
            grasp_score=grasp_score,
            object_pos=object_pos,
            base_target=base_target,
            commanded_target=target,
            clamped=clamped,
        )
        return self.state
