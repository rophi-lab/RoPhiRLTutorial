"""6-DOF task-space impedance control for the Rizon 4 arm.

Drives a frame on the palm to a goal pose with a spring/damper in SE(3), so the arm can be
commanded in task space while the hand runs its own grasp reflex underneath. The two never
talk to each other: the hand's reaction torques enter the arm only as a disturbance that
the impedance law rejects, which is what keeps the reflex arm-agnostic.

Formulation
-----------
Operational-space control (Khatib), not a joint-space PD on an IK solution::

    F   = Lambda (kp e - kd v)          task-space wrench
    tau = J^T F + N tau_posture + h     joint torques

with ``Lambda = (J M^-1 J^T)^-1`` the task-space inertia and ``h = qfrc_bias`` cancelling
gravity and Coriolis terms. Weighting by ``Lambda`` matters on this arm specifically: its
effective joint inertias span three orders of magnitude (joint2 ~3.9, joint7 ~0.0025
kg m^2), and an unweighted diagonal PD on it is not merely detuned but unstable -- joint3
buzzes at several rad/s. Routing the commanded acceleration through the inertia decouples
the task axes and makes one gain pair valid across the workspace.

Because gains multiply an acceleration, they carry units of 1/s^2 and 1/s and are given as
a bandwidth and damping ratio (``kp = wn^2``, ``kd = 2 zeta wn``) rather than as raw
stiffness in N/m. A stiffness figure like the IsaacLab config's ``stiffness=300`` is not
portable across configurations here, since the inertia seen at the palm changes with pose.

The arm is 7-DOF against a 6-DOF task, leaving a 1-DOF null space -- the elbow swivel.
It is resolved by a posture term projected through the dynamically-consistent null-space
projector, so holding the elbow never disturbs the palm.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

__all__ = ["ArmImpedanceController", "ImpedanceGains", "quat_from_rpy", "rpy_from_quat"]


def quat_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """wxyz quaternion from roll/pitch/yaw, ``R = Rz(yaw) Ry(pitch) Rx(roll)``.

    Defined explicitly rather than via ``mju_euler2Quat`` so that it round-trips exactly
    with :func:`rpy_from_quat`; GUI sliders read and write the same angles.
    """
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ]
    )


def rpy_from_quat(quat: np.ndarray) -> np.ndarray:
    """Roll/pitch/yaw from a wxyz quaternion. Inverse of :func:`quat_from_rpy`."""
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(quat, dtype=float))
    rot = mat.reshape(3, 3)
    # Clipped because rounding can push the sine argument just outside [-1, 1] at
    # pitch = +-90 deg, where roll and yaw are degenerate anyway.
    pitch = np.arcsin(-np.clip(rot[2, 0], -1.0, 1.0))
    return np.array(
        [
            np.arctan2(rot[2, 1], rot[2, 2]),
            pitch,
            np.arctan2(rot[1, 0], rot[0, 0]),
        ]
    )


@dataclass
class ImpedanceGains:
    """Impedance gains expressed as bandwidth (Hz) and damping ratio.

    ``damping_ratio`` 1.0 is critically damped; above 1.0 is overdamped and slower but
    tolerates contact and model error better.
    """

    pos_bandwidth_hz: float = 4.0
    pos_damping_ratio: float = 4.0
    rot_bandwidth_hz: float = 4.0
    rot_damping_ratio: float = 4.0
    # The posture task must be an order slower than the palm task, otherwise the two fight
    # inside the null space and the elbow drifts while the palm oscillates.
    posture_bandwidth_hz: float = 1.0
    posture_damping_ratio: float = 1.0

    def _pd(self, bandwidth_hz: float, damping_ratio: float) -> tuple[float, float]:
        natural_freq = 2.0 * np.pi * bandwidth_hz
        return natural_freq**2, 2.0 * damping_ratio * natural_freq

    @property
    def pos_pd(self) -> tuple[float, float]:
        return self._pd(self.pos_bandwidth_hz, self.pos_damping_ratio)

    @property
    def rot_pd(self) -> tuple[float, float]:
        return self._pd(self.rot_bandwidth_hz, self.rot_damping_ratio)

    @property
    def posture_pd(self) -> tuple[float, float]:
        return self._pd(self.posture_bandwidth_hz, self.posture_damping_ratio)


class ArmImpedanceController:
    """Task-space impedance controller for a serial arm, single environment, NumPy.

    Deliberately not batched: this is the reference implementation to validate the control
    law interactively. The mjlab action term will port the same maths to batched torch.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...],
        site_name: str,
        posture_q: np.ndarray | None = None,
        gains: ImpedanceGains | None = None,
        singular_rcond: float = 1e-3,
        char_length: float = 0.1,
    ) -> None:
        self.gains = gains if gains is not None else ImpedanceGains()
        self.singular_rcond = singular_rcond
        # Rotational rows of the Jacobian are dimensionless while translational rows carry
        # metres, so J M^-1 J^T mixes 1/(kg m^2) with 1/kg and has a condition number of
        # ~5000 on this arm for reasons that have nothing to do with kinematics. A relative
        # rcond applied to it silently truncates the three translational directions and the
        # controller then tracks orientation while ignoring position entirely.
        #
        # Scaling the rotational rows by a characteristic length re-expresses angle as arc
        # length, making the task space homogeneous: the condition number drops to ~200 and
        # rcond once again means "kinematic singularity". The scaling cancels exactly, so
        # this changes conditioning only, never the commanded torque.
        self.task_scale = np.array([1.0, 1.0, 1.0, char_length, char_length, char_length])

        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"No site named {site_name!r} in the model.")

        joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if any(j < 0 for j in joint_ids):
            missing = [n for n, j in zip(joint_names, joint_ids, strict=True) if j < 0]
            raise ValueError(f"Joints not found in the model: {missing}")
        self.joint_names = tuple(joint_names)
        self.qpos_adr = np.array([model.jnt_qposadr[j] for j in joint_ids])
        self.dof_adr = np.array([model.jnt_dofadr[j] for j in joint_ids])

        act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in joint_names]
        if any(a < 0 for a in act_ids):
            raise ValueError("Each arm joint needs a like-named actuator to receive torque.")
        self.act_adr = np.array(act_ids)
        # Motor actuators pass ctrl straight through as torque, so the ctrl range is the
        # effort limit. Saturating here rather than letting MuJoCo clamp keeps the
        # commanded and applied torque identical, which matters for the null-space term.
        limited = model.actuator_ctrllimited[self.act_adr].astype(bool)
        self.effort_limit = np.where(
            limited,
            np.abs(model.actuator_ctrlrange[self.act_adr, 1]),
            np.abs(model.actuator_forcerange[self.act_adr, 1]),
        )

        self.posture_q = (
            np.asarray(posture_q, dtype=float)
            if posture_q is not None
            else np.zeros(len(joint_ids))
        )

        self._target_pos = np.zeros(3)
        self._target_quat = np.array([1.0, 0.0, 0.0, 0.0])

        # Scratch buffers; mj_jacSite needs C-contiguous float64 of shape (3, nv).
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))
        self._mass_matrix = np.zeros((model.nv, model.nv))

    ##
    # Target.
    ##

    @property
    def target_pos(self) -> np.ndarray:
        return self._target_pos

    @property
    def target_quat(self) -> np.ndarray:
        return self._target_quat

    def set_target(self, pos: np.ndarray, quat: np.ndarray) -> None:
        """Set the goal pose. ``quat`` is wxyz and is renormalised."""
        # Rebind rather than mutate in place: the viewer writes targets from its GUI thread
        # while the physics thread reads them, and rebinding cannot be torn.
        quat = np.asarray(quat, dtype=float)
        self._target_pos = np.asarray(pos, dtype=float).copy()
        self._target_quat = quat / np.linalg.norm(quat)

    def set_target_euler(self, pos: np.ndarray, rpy: np.ndarray) -> None:
        """Set the goal pose from roll/pitch/yaw in radians. See :func:`quat_from_rpy`."""
        self.set_target(pos, quat_from_rpy(rpy))

    def capture_target(self, data: mujoco.MjData) -> None:
        """Adopt the site's current pose as the goal, so enabling the controller is a no-op."""
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, data.site_xmat[self.site_id])
        self.set_target(data.site_xpos[self.site_id], quat)

    ##
    # Errors.
    ##

    def pose_error(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        """Return (position error, orientation error) in the world frame.

        The orientation error is the rotation-vector form of the residual rotation, i.e.
        an axis scaled by the angle in radians, which is what the wrench needs.
        """
        pos_err = self._target_pos - data.site_xpos[self.site_id]

        quat_cur = np.zeros(4)
        mujoco.mju_mat2Quat(quat_cur, data.site_xmat[self.site_id])
        quat_cur_inv = np.zeros(4)
        mujoco.mju_negQuat(quat_cur_inv, quat_cur)
        quat_err = np.zeros(4)
        mujoco.mju_mulQuat(quat_err, self._target_quat, quat_cur_inv)
        rot_err = np.zeros(3)
        mujoco.mju_quat2Vel(rot_err, quat_err, 1.0)
        return pos_err, rot_err

    ##
    # Control.
    ##

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Torque for the arm joints. Requires current kinematics (call after ``mj_step1``)."""
        dof, qadr = self.dof_adr, self.qpos_adr

        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.site_id)
        jac = np.vstack([self._jacp[:, dof], self._jacr[:, dof]])  # (6, n)
        # Work in the homogeneous (arc-length) task space; see task_scale in __init__.
        jac = jac * self.task_scale[:, None]

        mujoco.mj_fullM(model, data, self._mass_matrix)
        mass = self._mass_matrix[np.ix_(dof, dof)]
        mass_inv = np.linalg.inv(mass)

        # Task-space inertia. Pseudo-inverse rather than inverse: near a kinematic
        # singularity J M^-1 J^T loses rank, and a plain inverse would command unbounded
        # torque along the lost direction. rcond simply drops that direction instead.
        task_inertia = np.linalg.pinv(jac @ mass_inv @ jac.T, rcond=self.singular_rcond)

        pos_err, rot_err = self.pose_error(data)
        vel = jac @ data.qvel[dof]
        kp_pos, kd_pos = self.gains.pos_pd
        kp_rot, kd_rot = self.gains.rot_pd
        scale_pos, scale_rot = self.task_scale[:3], self.task_scale[3:]
        desired_acc = np.concatenate(
            [
                scale_pos * (kp_pos * pos_err) - kd_pos * vel[:3],
                scale_rot * (kp_rot * rot_err) - kd_rot * vel[3:],
            ]
        )
        torque = jac.T @ (task_inertia @ desired_acc)

        # Posture task in the null space. The dynamically-consistent projector guarantees
        # this cannot accelerate the palm, so the elbow can be held without fighting the
        # task above it.
        jac_bar = mass_inv @ jac.T @ task_inertia
        null_proj = np.eye(len(dof)) - jac.T @ jac_bar.T
        kp_post, kd_post = self.gains.posture_pd
        posture_acc = kp_post * (self.posture_q - data.qpos[qadr]) - kd_post * data.qvel[dof]
        torque += null_proj @ (mass @ posture_acc)

        # Gravity and Coriolis. Taken from the full model, so the hand's weight is carried
        # by the arm rather than showing up as steady-state droop.
        torque += data.qfrc_bias[dof]

        return np.clip(torque, -self.effort_limit, self.effort_limit)

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        """Compute the torque and write it to ``data.ctrl``."""
        torque = self.compute(model, data)
        data.ctrl[self.act_adr] = torque
        return torque


##
# Interactive playground.
##


def _run_playground(port: int, timestep: float) -> None:
    """Serve a viser page for driving the palm target by hand.

    Only the arm actuators are written here. The hand keeps mjviser's own actuator
    sliders, so the two subsystems can be exercised independently -- which is the property
    the architecture is meant to have.
    """
    import viser
    from mjlab.entity.entity import Entity
    from mjviser import Viewer

    from rophi_rl_tutorial.assets.grasping import ARM_JOINT_NAMES, ARM_Q_HOME, get_rizon4_hand_robot_cfg
    from rophi_rl_tutorial.assets.grasping.rizon4_hand import _reset_to_home

    model = Entity(get_rizon4_hand_robot_cfg()).spec.compile()
    model.opt.timestep = timestep
    # The robot MJCF declares no <option>, so it would otherwise compile to explicit Euler,
    # where the arm's damping=2.0 sits right at the stability bound (dt < 2 I / c) on the
    # low-inertia wrist joints.
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    data = mujoco.MjData(model)
    _reset_to_home(model, data)

    controller = ArmImpedanceController(
        model,
        joint_names=ARM_JOINT_NAMES,
        site_name="grasp_site",
        posture_q=np.array([ARM_Q_HOME[n] for n in ARM_JOINT_NAMES]),
    )
    controller.capture_target(data)
    home_pos = controller.target_pos.copy()
    home_rpy = rpy_from_quat(controller.target_quat)

    def step_fn(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        # Split step so the controller sees kinematics, Jacobians and qfrc_bias evaluated
        # at the *current* state. Calling it around a plain mj_step would feed it values one
        # step stale. Requires a non-RK4 integrator, which is set above.
        mujoco.mj_step1(model, data)
        controller.apply(model, data)
        mujoco.mj_step2(model, data)

    server = viser.ViserServer(port=port, label="rizon4 arm impedance")

    with server.gui.add_folder("Palm target"):
        slider_x = server.gui.add_slider("x [m]", -0.2, 0.9, 0.005, float(home_pos[0]))
        slider_y = server.gui.add_slider("y [m]", -0.7, 0.7, 0.005, float(home_pos[1]))
        slider_z = server.gui.add_slider("z [m]", -0.1, 1.1, 0.005, float(home_pos[2]))
        slider_roll = server.gui.add_slider(
            "roll [deg]", -180, 180, 1, float(np.degrees(home_rpy[0]))
        )
        slider_pitch = server.gui.add_slider(
            "pitch [deg]", -90, 90, 1, float(np.degrees(home_rpy[1]))
        )
        slider_yaw = server.gui.add_slider(
            "yaw [deg]", -180, 180, 1, float(np.degrees(home_rpy[2]))
        )
        pose_sliders = (slider_x, slider_y, slider_z, slider_roll, slider_pitch, slider_yaw)

        btn_here = server.gui.add_button("Target = current palm pose")
        btn_home = server.gui.add_button("Target = home pose")
        btn_down = server.gui.add_button("Point palm down")

    with server.gui.add_folder("Impedance gains", expand_by_default=False):
        gain_pos_bw = server.gui.add_slider("position bandwidth [Hz]", 0.5, 12.0, 0.1, 4.0)
        gain_pos_z = server.gui.add_slider("position damping ratio", 0.3, 3.0, 0.05, 1.0)
        gain_rot_bw = server.gui.add_slider("rotation bandwidth [Hz]", 0.5, 12.0, 0.1, 4.0)
        gain_rot_z = server.gui.add_slider("rotation damping ratio", 0.3, 3.0, 0.05, 1.0)
        gain_null_bw = server.gui.add_slider("posture (null-space) [Hz]", 0.0, 5.0, 0.1, 1.0)

    with server.gui.add_folder("Tracking"):
        readout_pos = server.gui.add_text("position error", "0.0 mm", disabled=True)
        readout_rot = server.gui.add_text("orientation error", "0.0 deg", disabled=True)
        readout_tau = server.gui.add_text("peak |torque| / limit", "0 %", disabled=True)

    # Marker for the commanded pose, so the goal and the achieved pose are both visible.
    # mjviser adds its geometry at the scene root untransformed, so MuJoCo world
    # coordinates can be used directly here.
    target_frame = server.scene.add_frame(
        "/impedance_target",
        axes_length=0.12,
        axes_radius=0.004,
        position=tuple(home_pos),
        wxyz=tuple(controller.target_quat),
    )

    # Assigning to a viser slider's .value server-side fires its on_update handler, so
    # writing six sliders in a loop would otherwise push six partially-updated targets --
    # each mixing new position with stale orientation -- before the intended one.
    suppress_push = False

    def push_target(_=None) -> None:
        if suppress_push:
            return
        pos = np.array([slider_x.value, slider_y.value, slider_z.value])
        rpy = np.radians([slider_roll.value, slider_pitch.value, slider_yaw.value])
        controller.set_target_euler(pos, rpy)
        target_frame.position = tuple(pos)
        target_frame.wxyz = tuple(controller.target_quat)

    def pull_target(pos: np.ndarray, rpy: np.ndarray) -> None:
        """Write a pose back into the sliders, pushing it to the controller exactly once."""
        nonlocal suppress_push
        suppress_push = True
        try:
            for slider, value in zip(pose_sliders, [*pos, *np.degrees(rpy)], strict=True):
                slider.value = float(value)
        finally:
            suppress_push = False
        push_target()

    def push_gains(_=None) -> None:
        controller.gains.pos_bandwidth_hz = gain_pos_bw.value
        controller.gains.pos_damping_ratio = gain_pos_z.value
        controller.gains.rot_bandwidth_hz = gain_rot_bw.value
        controller.gains.rot_damping_ratio = gain_rot_z.value
        controller.gains.posture_bandwidth_hz = gain_null_bw.value

    for slider in pose_sliders:
        slider.on_update(push_target)
    for slider in (gain_pos_bw, gain_pos_z, gain_rot_bw, gain_rot_z, gain_null_bw):
        slider.on_update(push_gains)

    @btn_here.on_click
    def _(_) -> None:
        controller.capture_target(data)
        pull_target(controller.target_pos, rpy_from_quat(controller.target_quat))

    @btn_home.on_click
    def _(_) -> None:
        pull_target(home_pos, home_rpy)

    @btn_down.on_click
    def _(_) -> None:
        # Palm +z is the fingertip direction, so roll 180 deg aims the hand at the table.
        pull_target(controller.target_pos, np.array([np.pi, 0.0, 0.0]))

    def render_fn(scene) -> None:
        scene.update_from_mjdata(data)
        pos_err, rot_err = controller.pose_error(data)
        readout_pos.value = f"{np.linalg.norm(pos_err) * 1000:.1f} mm"
        readout_rot.value = f"{np.degrees(np.linalg.norm(rot_err)):.1f} deg"
        peak = np.abs(data.ctrl[controller.act_adr]) / controller.effort_limit
        readout_tau.value = f"{peak.max() * 100:.0f} %"

    def reset_fn(model: mujoco.MjModel, data: mujoco.MjData) -> None:
        _reset_to_home(model, data)
        pull_target(home_pos, home_rpy)

    push_target()
    push_gains()
    Viewer(
        model, data, step_fn=step_fn, render_fn=render_fn, reset_fn=reset_fn, server=server
    ).run()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8080, help="viser server port.")
    parser.add_argument("--timestep", type=float, default=0.002, help="simulation timestep [s].")
    _run_playground(parser.parse_args().port, parser.parse_args().timestep)


##
# Jacobian-transpose variant.
##


@dataclass
class JacobianTransposeGains:
    """Gains for :class:`JacobianTransposeController`, in physical units.

    Unlike :class:`ImpedanceGains` these are true stiffnesses (N/m, N m/rad) because the
    wrench is applied directly rather than through the task-space inertia. Defaults are the
    values the IsaacLab ``GraspPhaseSpringDamperControlActionCfg`` was configured with.
    """

    stiffness: float = 300.0
    damping: float = 80.0
    orientation_stiffness: float = 8.0
    orientation_damping: float = 1.5
    joint_damping: float = 2.0
    """Joint-space damping. This is what stabilises the law in place of inertia weighting."""


class JacobianTransposeController:
    """Cartesian impedance via ``tau = J^T F``, with no task-space inertia weighting.

    The formulation IsaacLab's ``SpringDamperControlAction`` uses. It exists here because it
    is the only one portable into an mjlab action term: ``Lambda = (J M^-1 J^T)^-1`` needs a
    mass matrix, and mjlab exposes neither ``qM`` nor a host-callable ``mul_m``.

    Dropping ``Lambda`` means the closed loop is no longer decoupled -- the effective
    stiffness seen at the palm varies with configuration by whatever the task-space inertia
    happens to be -- so stability rests on ``joint_damping`` and effort clamping instead.
    Shares the target/error interface with :class:`ArmImpedanceController` so the two can be
    swapped and compared directly.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        joint_names: tuple[str, ...],
        site_name: str,
        gains: JacobianTransposeGains | None = None,
        gravity_comp: bool = True,
    ) -> None:
        self.gains = gains if gains is not None else JacobianTransposeGains()
        self.gravity_comp = gravity_comp
        # Reuse the sibling controller's model introspection: same site, joints, actuators
        # and effort limits, so a comparison differs only in the control law.
        self._ref = ArmImpedanceController(model, joint_names, site_name)
        self.site_id = self._ref.site_id
        self.dof_adr = self._ref.dof_adr
        self.qpos_adr = self._ref.qpos_adr
        self.act_adr = self._ref.act_adr
        self.effort_limit = self._ref.effort_limit
        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    @property
    def target_pos(self) -> np.ndarray:
        return self._ref.target_pos

    @property
    def target_quat(self) -> np.ndarray:
        return self._ref.target_quat

    def set_target(self, pos: np.ndarray, quat: np.ndarray) -> None:
        self._ref.set_target(pos, quat)

    def set_target_euler(self, pos: np.ndarray, rpy: np.ndarray) -> None:
        self._ref.set_target_euler(pos, rpy)

    def capture_target(self, data: mujoco.MjData) -> None:
        self._ref.capture_target(data)

    def pose_error(self, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
        return self._ref.pose_error(data)

    def compute(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        dof = self.dof_adr
        mujoco.mj_jacSite(model, data, self._jacp, self._jacr, self.site_id)
        jac = np.vstack([self._jacp[:, dof], self._jacr[:, dof]])

        pos_err, rot_err = self.pose_error(data)
        vel = jac @ data.qvel[dof]
        gains = self.gains
        wrench = np.concatenate(
            [
                gains.stiffness * pos_err - gains.damping * vel[:3],
                gains.orientation_stiffness * rot_err - gains.orientation_damping * vel[3:],
            ]
        )
        torque = jac.T @ wrench
        torque -= gains.joint_damping * data.qvel[dof]
        if self.gravity_comp:
            torque += data.qfrc_bias[dof]
        return np.clip(torque, -self.effort_limit, self.effort_limit)

    def apply(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        torque = self.compute(model, data)
        data.ctrl[self.act_adr] = torque
        return torque
