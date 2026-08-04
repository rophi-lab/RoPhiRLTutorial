"""mjlab task for reflexive grasping.

Wraps the verified batched controllers as an mjlab ``ActionTerm`` and the MDP terms as
manager terms, so training runs on mujoco-warp (measured ~57k env-steps/s at 256 envs against
388/s for the CPU reference env in ``env.py``).

Object diversity via per-world mesh variants
--------------------------------------------
mjlab compiles one model shared by every environment, but its per-world mesh variant feature
lets a single ``object`` free body hold a *different* YCB mesh (and its own mass/inertia) in
each parallel world, all sharing one kinematic structure. So ``make_task`` accepts either one
object name (same object everywhere, e.g. ``RoPhi-Grasp-Apple``) or a list (a different
object per world, evenly allocated, e.g. ``RoPhi-Grasp-YCB`` over the EASY set). Because
the body is always named ``object``, every observation, reward and contact match is identical
in both cases.

Stateful rewards
----------------
mjlab reward terms are plain functions, so the improvement-reward watermarks live on the env
(lazily created, keyed by name) and are cleared by a ``mode="reset"`` event term. That mirrors
how the IsaacLab env carried them and keeps the reset in one place.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import torch
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.action_manager import ActionTerm, ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.scene import SceneCfg
from mjlab.sensor.contact_sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import SimulationCfg

from rophi_rl_tutorial.assets.grasping import (
    ARM_JOINT_NAMES,
    ARM_Q_HOME,
    HAND_JOINT_NAMES,
    HAND_Q_CLOSE,
    HAND_Q_GRASP_OPEN,
    get_rizon4_hand_robot_cfg,
)
from rophi_rl_tutorial.assets.grasping.ycb import get_ycb_object_cfg, get_ycb_variant_object_cfg
from rophi_rl_tutorial.controllers.grasping.batched import (
    BatchedArmImpedance,
    BatchedGraspPhase,
    BatchedGraspPhaseCfg,
    BatchedImpedanceGains,
    quat_conjugate,
    quat_mul,
    quat_rotate,
    quat_to_rotvec,
)
from rophi_rl_tutorial.mjlab_patches import apply_variant_indexing_patch
from rophi_rl_tutorial.tasks.grasping.mdp.terms import (
    CONTACT_WEIGHTS,
    REWARD_CONTACT_LINKS,
)

# Fix mjlab's per-world-variant sibling-entity indexing bug before any env is built. Without
# this the robot's forward-kinematics data is garbage in any multi-object (variant) scene.
apply_variant_indexing_patch()

FLOOR_HEIGHT = -0.04
MAT_THICKNESS = 0.01
TABLE_HEIGHT = FLOOR_HEIGHT + MAT_THICKNESS
MAT_CENTRE_XY = (0.55, 0.0)
MAT_HALF_EXTENT = (0.47, 0.43)
SPAWN_XY = (0.55, 0.15)
SPAWN_Z = TABLE_HEIGHT + 0.06

HAND_LINK_PATTERN = r"(palm|finger_r_link_.*)"


##
# Action term.
##


@dataclass(kw_only=True)
class GraspActionCfg(ActionTermCfg):
    entity_name: str = "robot"
    object_name: str = "object"
    site_name: str = "grasp_site"
    impedance: BatchedImpedanceGains = field(default_factory=BatchedImpedanceGains)
    phase: BatchedGraspPhaseCfg = field(
        # Two REACH-pose regimes, selected by reach_randomize (the --arm-residual flag turns it
        # off). Both were tuned in the play --tune-reach GUI.
        #  * reach_randomize True  -> DR: object + sampled world offset, flat-hold x sampled rpy.
        #  * reach_randomize False -> fixed tuned pose: object + pre_grasp_offset (world),
        #    flat-hold x reach_rpy_fixed, with the hand's bounded pos/rot residual on top.
        default_factory=lambda: BatchedGraspPhaseCfg(
            min_setpoint_z=TABLE_HEIGHT,
            reach_randomize=True,
            reach_pos_bounds=((-0.045, 0.05), (-0.04, 0.03), (0.04, 0.075)),
            reach_rpy_bounds=(
                (math.radians(-80.0), math.radians(80.0)),
                (math.radians(-5.0), math.radians(30.0)),
                (math.radians(-44.0), math.radians(44.0)),
            ),
            # Fixed tuned grasp pose (used when reach_randomize is False): read from the tuner.
            pre_grasp_offset=(0.03, 0.015, 0.065),
            offset_frame="world",
            reach_rpy_fixed=(0.0, math.radians(11.0), 0.0),
            # Room for the +-3 cm position residual around the fixed offset without clipping.
            reach_object_radius=0.15,
        )
    )
    arm_control: str = "impedance"
    """Arm control law. ``"impedance"`` (default): the operational-space Cartesian spring/damper
    (``BatchedArmImpedance``). ``"velocity"``: a Jacobian-pseudoinverse velocity controller -- a
    Cartesian velocity field drives the grasp_site to the phase target, a *damped Jacobian
    pseudoinverse* (no QP -- fast, batched) maps that twist to arm joint velocities, and a
    joint-space PD (+ gravity feedforward) tracks ``q_des = q + qd*dt``. This is the sim-to-real
    deploy arm controller (the deploy adds joint-limit handling); training/play with it via
    ``--arm-velocity`` co-adapts the hand policy to the deploy arm."""
    vf_lin_speed: float = 1.0
    """Velocity field: far-field linear speed [m/s] (ramps to 0 within vf_lin_eps of the target)."""
    vf_lin_eps: float = 0.15
    vf_ang_speed: float = 2.0
    vf_ang_eps: float = 0.25
    vel_damping: float = 0.02
    """Damped-pseudoinverse regularisation lambda^2: qd = J^T (J J^T + lambda^2 I)^-1 twist."""
    qd_limit: float = 2.5
    """Per-joint velocity clamp [rad/s] on the pseudoinverse solution."""
    arm_kp: tuple[float, ...] = (80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 20.0)
    arm_kd: tuple[float, ...] = (4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 1.0)

    arm_residual: bool = False
    """Whether the hand policy also outputs a 6-D palm residual: a 3-D position offset (dims 1:4)
    and a 3-D orientation offset (dims 4:7, euler XYZ) added to the arm phase target. With the
    REACH-pose randomisation above the hand is kept fully arm-agnostic, so this is off by default
    (action = [grasp_score, hand]). Set True *and* reach_randomize False (the --arm-residual train
    flag does both) for the residual-controlled arm, where the hand shapes both where and how the
    palm meets the object."""
    hand_mode: str = "joints"
    """Hand action space: "joints" (20 direct joint targets) or "pca" (5 PCA latent dims).

    "joints" gives full independent per-joint control and learns the grasp faster; "pca"
    commands coordinated grasp shapes on the fitted human-grasp manifold (5-DoF, more natural
    looking) but explores slower.
    """
    hand_pca_project: float = 0.0
    """(joints mode only) Blend the commanded 20-joint hand pose toward its PCA-manifold projection
    by this factor before sending it: ``cmd = (1-a)*cmd + a*reconstruct(cmd)``. 0 = off (raw
    joints), 1 = hard-project onto the manifold (exactly PCA-like shapes), 0.5 = halfway. Lets a
    joint-space policy be fine-tuned to produce natural PCA-like hand shapes while keeping its
    20-D action space."""
    hand_gate: bool = False
    """Arm-gated hand: the arm tells the hand when to grasp. When True, the hand policy controls the
    fingers only once the palm is within the phase's ``engage_radius`` of the object (the latched
    ``phase.engaged`` flag); until then the fingers are held at the default pre-grasp pose
    (``HAND_Q_GRASP_OPEN``). Fixes the "hand closes while standing still" failure: a proprioception
    policy has no notion of *when* to grasp, so the arm supplies that signal. The velocity field
    makes the open<->policy hand-off smooth. Off by default (unchanged behaviour); enable at play
    time to gate an already-trained policy, or at train time to learn with the gate."""
    arm_residual_scale: float = 0.03
    """Max policy offset [m per axis] added to the arm phase target, ported from IsaacLab's
    ``palm_target_residual``. Gives the hand policy a bounded say in where the palm is driven so
    it can fine-tune the approach per object around the fixed tuned pose. Raw residual dims (1:4)
    are clamped to [-1, 1] then scaled by this (+-3 cm)."""
    arm_residual_rot_scale: float = math.radians(5.0)
    """Max policy palm-orientation offset [rad per axis, +-5 deg] added to the arm phase target
    (only when arm_residual). Raw dims (4:7) are clamped to [-1, 1], scaled by this, and applied
    as an euler-XYZ rotation in the palm/body frame, letting the hand tilt the palm per object."""
    hand_preshape: float = 0.0
    hand_v_max: float = 0.8
    """Max finger-joint closing speed [rad/s]: the fingers close quasi-statically so contact
    builds gently (less dynamic overshoot and fewer force spikes). Note the *steady* grip force
    -- and thus penetration -- is set by the hand actuator effort limit, not this; see
    ``HAND_EFFORT_LIMIT`` in assets/rizon4_hand.py. Ported from the IsaacLab hand velocity field."""
    hand_slowdown_distance: float = 0.35
    """Joint error [rad] below which the closing speed ramps down linearly to zero."""
    palm_flatten_euler: tuple[float, float, float] = (0.38, 0.0, 0.0)
    """World-frame XYZ rotation [rad] applied to the FK home palm attitude to hold the palm
    flatter (more level). The home pose from ARM_Q_HOME has the palm grasp axis ~63 deg off
    straight-down; a positive rotation about world X tips it toward horizontal (~85 deg off
    down, i.e. a level/flat palm). Tune in the viewer."""
    squash: str = "sigmoid"
    """How raw policy output maps to [0, 1]: "sigmoid" (gradient everywhere) or "clamp".

    "clamp" is retained only to replay run 1 faithfully; it has zero gradient outside the box
    and is what let run 1's grasp_score get stuck at a raw -14. New runs use "sigmoid".
    """

    def build(self, env) -> GraspAction:
        return GraspAction(self, env)


class GraspAction(ActionTerm):
    """Two scalars: grasp score drives the arm phase, hand closure drives the hand.

    Nothing routes between them. The arm's only input from the hand is the scalar score, which
    is the property that keeps the grasp reflex arm-agnostic.

    The setpoint is latched in ``process_actions`` at policy rate while the impedance law runs
    in ``apply_actions`` at physics rate, matching how the IsaacLab term split the two.
    """

    cfg: GraspActionCfg

    def __init__(self, cfg: GraspActionCfg, env) -> None:
        super().__init__(cfg, env)
        # A standalone compile of the robot spec, not env.sim.mj_model: EntityData arrays are
        # entity-local (39 robot bodies) while the scene model also holds the object and world
        # (41), so scene-global indices would silently misalign. The attached spec cannot be
        # recompiled in place, hence a fresh Entity from the same cfg -- identical ordering.
        import mujoco
        from mjlab.entity.entity import Entity as _Entity

        model = _Entity(get_rizon4_hand_robot_cfg()).spec.compile()
        self._object = env.scene[cfg.object_name]
        self._num = env.num_envs

        # Home palm pose, computed once by FK at ARM_Q_HOME. This is the fixed hold pose the
        # arm returns to on LIFT and the fixed orientation held throughout the episode.
        #
        # It is NOT read from live state at reset. mjlab's _reset_idx runs the joint-reset
        # event and then action_manager.reset() with no sim.forward() between, so site_pos_w /
        # site_quat_w read in reset() are stale -- they reflect the *previous* episode's end
        # pose. Capturing from them gave a hold orientation 100+ deg off the home pose and
        # different every episode, which is why the hand orientation drifted in playback.
        _data = mujoco.MjData(model)
        for name in ARM_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            _data.qpos[model.jnt_qposadr[jid]] = ARM_Q_HOME[name]
        mujoco.mj_forward(model, _data)
        _gsid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, cfg.site_name)
        import numpy as _np

        _home_quat = _np.zeros(4)
        mujoco.mju_mat2Quat(_home_quat, _data.site_xmat[_gsid])
        # Flatten the held palm attitude by a fixed world-frame rotation (left-multiply, so it
        # is applied in world, not palm, coordinates -- "flat" is relative to gravity). Composed
        # X then Y then Z from palm_flatten_euler; the default tips the palm toward palm-down.
        _off = _np.array([1.0, 0.0, 0.0, 0.0])
        for _ax, _ang in enumerate(cfg.palm_flatten_euler):
            if _ang != 0.0:
                _aq = _np.zeros(4)
                _aq[0] = _np.cos(_ang / 2.0)
                _aq[1 + _ax] = _np.sin(_ang / 2.0)
                _tmp = _np.zeros(4)
                mujoco.mju_mulQuat(_tmp, _aq, _off)
                _off = _tmp
        _flat_quat = _np.zeros(4)
        mujoco.mju_mulQuat(_flat_quat, _off, _home_quat)
        self._home_site_pos = torch.as_tensor(
            _data.site_xpos[_gsid], dtype=torch.float32, device=env.device
        )
        self._home_site_quat = torch.as_tensor(
            _flat_quat, dtype=torch.float32, device=env.device
        )

        self.impedance = BatchedArmImpedance(
            model,
            ARM_JOINT_NAMES,
            cfg.site_name,
            env.num_envs,
            device=env.device,
            gains=cfg.impedance,
            body_offset=1,
        )
        self.phase = BatchedGraspPhase(cfg.phase, env.num_envs, device=env.device)

        ids, _ = self._entity.find_joints(list(ARM_JOINT_NAMES), preserve_order=True)
        self._arm_joint_ids = ids
        ids, _ = self._entity.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
        self._hand_joint_ids = ids

        q_open = torch.tensor(
            [HAND_Q_GRASP_OPEN[n] for n in HAND_JOINT_NAMES], device=env.device
        )
        q_close = torch.tensor(
            [HAND_Q_CLOSE[n] for n in HAND_JOINT_NAMES], device=env.device
        )
        self._q_open = q_open.unsqueeze(0)
        self._q_close = q_close.unsqueeze(0)

        # Per-joint position limits for the 20 hand joints, from the model. The policy now
        # commands each joint directly; its squashed [0, 1] output maps across [lo, hi] so the
        # full range is reachable. Read from jnt_range (soft factor is 1.0 for this hand).
        lo, hi = [], []
        for name in HAND_JOINT_NAMES:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo.append(float(model.jnt_range[jid, 0]))
            hi.append(float(model.jnt_range[jid, 1]))
        self._q_lo = torch.tensor(lo, device=env.device).unsqueeze(0)
        self._q_hi = torch.tensor(hi, device=env.device).unsqueeze(0)

        # Velocity (Jacobian-pseudoinverse) arm control (arm_control="velocity"): reuses the
        # impedance controller's batched Jacobian + gravity + effort-limit; drives the SAME phase
        # target (set on self.impedance) via a velocity field -> damped pinv -> joint PD. These
        # params are held as mutable instance attrs so play_grasp can tune them live
        # (set_velocity_params); read fresh each apply_actions.
        self._vel = {
            "vf_lin_speed": float(cfg.vf_lin_speed),
            "vf_lin_eps": float(cfg.vf_lin_eps),
            "vf_ang_speed": float(cfg.vf_ang_speed),
            "vf_ang_eps": float(cfg.vf_ang_eps),
            "vel_damping": float(cfg.vel_damping),
            "qd_limit": float(cfg.qd_limit),
        }
        self._arm_kp = torch.tensor(cfg.arm_kp, device=env.device).unsqueeze(0)
        self._arm_kd = torch.tensor(cfg.arm_kd, device=env.device).unsqueeze(0)

        # Hand action space: PCA latent or direct joints. PCA loads the fitted grasp basis and
        # exposes only z_dim (=5) action dims; the decoded 20-vector is clamped to joint limits.
        self._hand_mode = cfg.hand_mode
        if cfg.hand_mode == "pca":
            from rophi_rl_tutorial.assets.grasping.hand_pca import load_hand_pca

            self._pca = load_hand_pca(device=env.device)
            self._num_hand = self._pca.z_dim
        else:
            self._pca = None
            self._num_hand = len(HAND_JOINT_NAMES)

        # (joints mode) optional projection of the commanded pose toward the PCA manifold, for
        # fine-tuning a joint-space policy to natural PCA-like hand shapes. Blend factor in [0, 1].
        self._pca_project = float(cfg.hand_pca_project)
        self._proj_pca = None
        if self._pca_project > 0.0 and self._pca is None:
            from rophi_rl_tutorial.assets.grasping.hand_pca import load_hand_pca

            self._proj_pca = load_hand_pca(device=env.device)

        # Action layout: [grasp_score(1), (palm pos residual(3) + rot residual(3) if
        # arm_residual), hand(5/20)]. With arm_residual False the hand policy is arm-agnostic --
        # it never touches the arm setpoint; the arm sees only the scalar grasp score, and the
        # REACH pose is randomised instead.
        self._use_residual = cfg.arm_residual
        self._residual_scale = cfg.arm_residual_scale
        self._residual_rot_scale = cfg.arm_residual_rot_scale
        self._n_pre = (
            7 if self._use_residual else 1
        )  # action dims before the hand block
        self._raw = torch.zeros(
            env.num_envs, self._n_pre + self._num_hand, device=env.device
        )

        # Contact-gate buffers: body ids (entity-local, for body_link_quat_w) and per-link
        # inward axis, in REWARD_CONTACT_LINKS order. The inside-of-hand gate keeps only
        # contacts whose normal opposes the link's inward face -- +Z for the palm (toward the
        # fingers) and +X for the phalanges, the mjlab-frame directions verified in mdp.terms.
        rew_ids, _ = self._entity.find_bodies(
            list(REWARD_CONTACT_LINKS), preserve_order=True
        )
        self._reward_body_ids = torch.tensor(
            rew_ids, device=env.device, dtype=torch.long
        )
        self._inward_local = torch.tensor(
            [
                [0.0, 0.0, 1.0] if n == "palm" else [1.0, 0.0, 0.0]
                for n in REWARD_CONTACT_LINKS
            ],
            device=env.device,
        )
        # Hand velocity field. hand_target is the *commanded* pose the policy asked for; the
        # position target actually sent each physics step is rate-limited toward it so the
        # fingers close smoothly instead of snapping shut. Ported from IsaacLab's hand action:
        # per-joint speed = v_max * clamp(|err| / slowdown, 1), applied against measured q.
        self.hand_target = self._q_open.repeat(env.num_envs, 1)
        # Internal command pose, integrated open-loop toward hand_target at the capped speed.
        # It is the position target actually sent to the servo. Open-loop (not measured q) so
        # a finger that lags still gets a growing position error -- and therefore real torque
        # from the soft hand servo -- while the command itself stays rate-limited.
        self._q_cmd = self._q_open.repeat(env.num_envs, 1)
        self._v_max = cfg.hand_v_max
        self._slowdown = max(cfg.hand_slowdown_distance, 1e-6)
        self._hand_dt = env.physics_dt
        self._site_id = self.impedance.site_id
        self._lift_captured = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )

    @property
    def action_dim(self) -> int:
        return (
            self._n_pre + self._num_hand
        )  # grasp + [3 residual] + hand (5 PCA or 20 joints)

    @property
    def home_goal_pos(self) -> torch.Tensor:
        """Fixed end-effector goal position (the LIFT retreat pose). Shape (3,)."""
        return self._home_site_pos

    @property
    def grasp_score(self) -> torch.Tensor:
        """The squashed grasp score in [0, 1] the policy last output. Shape (num_envs,).

        This is the scalar the arm phase controller sees -- <0.5 reach, >0.5 lift -- computed
        from the raw action the same way ``process_actions`` does, so a viewer reads exactly
        what drives the phase latch.
        """
        raw = self._raw[:, 0]
        return (
            torch.sigmoid(raw) if self.cfg.squash == "sigmoid" else raw.clamp(0.0, 1.0)
        )

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._num, device=self.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        # Fixed home pose, not live state: see __init__. Broadcasting the same constant to
        # every reset env keeps the retreat position and the held orientation identical to the
        # default pose every episode. env_origins are zero under mujoco-warp (each env is its
        # own world), so no per-env position offset is needed.
        n = env_ids.numel()
        site_pos = self._home_site_pos.unsqueeze(0).expand(n, 3)
        site_quat = self._home_site_quat.unsqueeze(0).expand(n, 4)
        self.phase.reset(env_ids, site_pos, site_quat)
        self.impedance.set_target(site_pos, site_quat, env_ids)
        self._raw[env_ids] = 0.0
        self.hand_target[env_ids] = self._q_open
        self._q_cmd[env_ids] = self._q_open

    def process_actions(self, actions: torch.Tensor) -> None:
        # Squash to [0, 1] with a sigmoid, not clamp. clamp has zero gradient outside the
        # box, so once the policy drifts a channel past the edge it is stuck there with no
        # signal to return -- run 1 drove grasp_score to a raw mean of -14 and, with action
        # std ~0.9, every sample clamped to 0, permanently disabling the LIFT phase. sigmoid
        # keeps gradient and exploration alive across the whole real line.
        actions = torch.nan_to_num(actions, nan=0.0)
        self._raw[:] = actions
        # grasp score (dim 0): squashed to [0, 1] -- drives the arm phase latch.
        raw_score = actions[:, 0]
        score = (
            torch.sigmoid(raw_score)
            if self.cfg.squash == "sigmoid"
            else raw_score.clamp(0.0, 1.0)
        )
        # palm residual (only when arm_residual): position (dims 1:4) and orientation (dims 4:7),
        # each clamped to [-1, 1] (symmetric, so zero action = no offset -- NOT sigmoid, which
        # would bias the palm) then scaled to metres / radians. With the arm-agnostic hand policy
        # there is no residual and the arm ignores the hand.
        if self._use_residual:
            residual = actions[:, 1:4].clamp(-1.0, 1.0) * self._residual_scale
            rot_residual = actions[:, 4:7].clamp(-1.0, 1.0) * self._residual_rot_scale
        else:
            residual = None
            rot_residual = None
        # hand (dims _n_pre:): squashed to [0, 1]. PCA: [0,1] -> latent in [p01, p99] -> decode to
        # a coordinated 20-joint pose, clamped to limits. Joints: [0,1] mapped per joint to
        # [lo, hi]. Either way this is the *commanded* pose; the velocity field in apply_actions
        # rate-limits toward it, so the fingers close smoothly rather than snapping.
        raw_hand = actions[:, self._n_pre :]
        hand = (
            torch.sigmoid(raw_hand)
            if self.cfg.squash == "sigmoid"
            else raw_hand.clamp(0.0, 1.0)
        )
        if self._pca is not None:
            z = self._pca.latent_from_alpha(hand)
            self.hand_target = self._pca.decode(z).clamp(self._q_lo, self._q_hi)
        else:
            self.hand_target = self._q_lo + hand * (self._q_hi - self._q_lo)
            if self._proj_pca is not None:
                # Blend the joint target toward its PCA-manifold projection (natural grasp shape).
                recon = self._proj_pca.reconstruct(self.hand_target).clamp(
                    self._q_lo, self._q_hi
                )
                self.hand_target = (
                    1.0 - self._pca_project
                ) * self.hand_target + self._pca_project * recon
        object_pos = self._object.data.root_link_pos_w
        palm_pos = self._entity.data.site_pos_w[:, self._site_id]
        target, target_quat = self.phase.update(
            score, object_pos, palm_pos, residual, rot_residual
        )
        # Arm-gated hand: until the arm brings the palm within engage_radius of the object, hold the
        # fingers at the default pre-grasp pose instead of the policy's commanded pose. phase.engaged
        # is the latched proximity flag the phase just updated; the velocity field smooths the switch.
        if self.cfg.hand_gate:
            self.hand_target = torch.where(
                self.phase.engaged.unsqueeze(-1), self.hand_target, self._q_open
            )
        # REACH-target override (tune-reach GUI only): drive the palm to an externally-set pose
        # instead of the phase target, so the offset sliders can pose the real hand by hand.
        override_pos = getattr(self, "_reach_override_pos", None)
        if override_pos is not None:
            target, target_quat = override_pos, self._reach_override_quat
        self.impedance.set_target(target, target_quat)

    def set_velocity_params(self, **kw) -> None:
        """Live-update the velocity-control params (for play_grasp tuning). Recognised keys:
        vf_lin_speed, vf_lin_eps, vf_ang_speed, vf_ang_eps, vel_damping, qd_limit, arm_kp, arm_kd.
        """
        for k in (
            "vf_lin_speed",
            "vf_lin_eps",
            "vf_ang_speed",
            "vf_ang_eps",
            "vel_damping",
            "qd_limit",
        ):
            if k in kw and kw[k] is not None:
                self._vel[k] = float(kw[k])
        for k in ("arm_kp", "arm_kd"):
            if kw.get(k) is not None:
                t = torch.as_tensor(
                    kw[k], dtype=torch.float32, device=self._arm_kp.device
                ).reshape(1, -1)
                setattr(self, f"_{k}", t)

    @staticmethod
    def _vf_speed(err_norm: torch.Tensor, v: float, eps: float) -> torch.Tensor:
        """Quadratic-decay velocity-field speed: ``v`` when far (err>eps), ramping to 0 within eps.
        Batched form of ``utils.velocity_fields.grasping.goto_x_des_with_v_quad_decay_from_eps``.
        """
        near = v * (1.0 - (err_norm - eps) ** 2 / (eps**2))
        speed = torch.where(err_norm > eps, torch.full_like(err_norm, float(v)), near)
        return torch.where(err_norm < 1e-6, torch.zeros_like(err_norm), speed)

    def _velocity_torque(self, data) -> torch.Tensor:
        """Jacobian-pseudoinverse velocity arm control (the deploy method), batched over envs.

        Drives the grasp_site to the phase target latched on ``self.impedance`` via a Cartesian
        velocity field -> damped Jacobian pseudoinverse (NO QP) -> joint PD + gravity feedforward.
        Reuses the impedance controller's batched Jacobian, gravity and effort limit; only the
        control law differs. Params are read from ``self._vel`` so play_grasp can tune them live.
        """
        p = self._vel
        site_pos = data.site_pos_w[:, self._site_id]  # (N, 3)
        site_quat = data.site_quat_w[:, self._site_id]  # (N, 4)
        J = self.impedance.jacobian.compute(
            data.body_link_pos_w, data.body_link_quat_w, site_pos
        )  # (N, 6, 7), rows [lin; ang]

        # Cartesian velocity field toward the phase target (position + orientation).
        pos_err = self.impedance.target_pos - site_pos
        e_pos = torch.linalg.norm(pos_err, dim=-1, keepdim=True).clamp(min=1e-9)
        v_des = self._vf_speed(e_pos, p["vf_lin_speed"], p["vf_lin_eps"]) * (
            pos_err / e_pos
        )
        rot_err = quat_to_rotvec(
            quat_mul(self.impedance.target_quat, quat_conjugate(site_quat))
        )
        e_rot = torch.linalg.norm(rot_err, dim=-1, keepdim=True).clamp(min=1e-9)
        w_des = self._vf_speed(e_rot, p["vf_ang_speed"], p["vf_ang_eps"]) * (
            rot_err / e_rot
        )
        twist = torch.cat([v_des, w_des], dim=-1).unsqueeze(-1)  # (N, 6, 1)

        # Damped Jacobian (right) pseudoinverse: qd = J^T (J J^T + lambda^2 I)^-1 twist. The 6x6
        # solve is the minimum-norm joint velocity for the desired twist -- fast, batched, no QP.
        JJt = J @ J.transpose(1, 2) + p["vel_damping"] * torch.eye(
            6, device=J.device
        ).unsqueeze(0)
        qd = (J.transpose(1, 2) @ torch.linalg.solve(JJt, twist)).squeeze(-1)  # (N, 7)
        qd = torch.clamp(qd, -p["qd_limit"], p["qd_limit"])

        # Joint PD (+ gravity FF) tracking q_des = q + qd*dt -- the law the deploy plant applies.
        qd_meas = data.joint_vel[:, self._arm_joint_ids]
        grav = self.impedance.gravity.compute(
            data.body_link_pos_w, data.body_link_quat_w, data.body_com_pos_w
        )
        tau = self._arm_kp * (qd * self._hand_dt) + self._arm_kd * (qd - qd_meas) + grav
        return torch.clamp(
            tau, -self.impedance.effort_limit, self.impedance.effort_limit
        )

    def apply_actions(self) -> None:
        data = self._entity.data
        if self.cfg.arm_control == "velocity":
            torque = self._velocity_torque(data)
        else:
            torque = self.impedance.compute(
                data.body_link_pos_w,
                data.body_link_quat_w,
                data.site_pos_w[:, self._site_id],
                data.site_quat_w[:, self._site_id],
                data.joint_vel[:, self._arm_joint_ids],
                data.body_com_pos_w,
            )
        self._entity.set_joint_effort_target(torque, joint_ids=self._arm_joint_ids)

        # Velocity field on the internal command: per-joint saturated proportional speed,
        # v_max * clamp(|err| / slowdown, 1), integrated at the physics dt. A finger blocked by
        # the object stalls at its measured position while _q_cmd keeps advancing to the
        # commanded pose, so the position error -- and grip torque -- builds smoothly rather
        # than snapping shut.
        err = self.hand_target - self._q_cmd
        speed = self._v_max * (err.abs() / self._slowdown).clamp(max=1.0)
        self._q_cmd = self._q_cmd + torch.sign(err) * speed * self._hand_dt
        self._entity.set_joint_position_target(
            self._q_cmd, joint_ids=self._hand_joint_ids
        )


##
# Shared per-env state.
##


def _state(env) -> dict:
    """Lazily-created per-env buffers for the stateful rewards, hung off the env."""
    if not hasattr(env, "_grasp_state"):
        n, dev = env.num_envs, env.device
        env._grasp_state = {
            "contact_best": torch.full((n,), float("inf"), device=dev),
            "hold_streak": torch.zeros(n, dtype=torch.long, device=dev),
            "rest_z": torch.zeros(n, device=dev),
            "rest_ready": torch.zeros(n, dtype=torch.bool, device=dev),
            "rest_count": torch.zeros(n, dtype=torch.long, device=dev),
            "weights": torch.tensor(CONTACT_WEIGHTS, device=dev),
        }
    return env._grasp_state


def reset_grasp_state(env, env_ids: torch.Tensor) -> None:
    """Event term (mode='reset'). Clears every improvement watermark and the lift latch.

    Without this the watermarks carry across episodes and the improvement rewards go silently
    dead after the first episode -- the arm would appear to stop being rewarded for progress.
    """
    s = _state(env)
    s["contact_best"][env_ids] = float("inf")
    s["hold_streak"][env_ids] = 0
    s["rest_ready"][env_ids] = False
    s["rest_count"][env_ids] = 0


def _reward_link_order(env, sensor_name: str) -> torch.Tensor:
    """Permutation from sensor primary order to REWARD_CONTACT_LINKS order.

    The sensor returns primaries in model order (palm first here), while CONTACT_WEIGHTS is
    written in the IsaacLab order (palm last). Applying the weights without this permutation
    would silently weight the thumb's 5/5/10 onto the palm and index links -- wrong, and
    invisible in any shape check.
    """
    key = f"_reward_perm_{sensor_name}"
    if not hasattr(env, key):
        names = list(env.scene[sensor_name].primary_names)
        perm = [names.index(n) for n in REWARD_CONTACT_LINKS]
        setattr(env, key, torch.tensor(perm, device=env.device, dtype=torch.long))
    return getattr(env, key)


def _contact_found(env, sensor_name: str) -> torch.Tensor:
    """(num_envs, num_links) bool of object<->link contact, from the contact sensor."""
    data = env.scene[sensor_name].data
    found = data.found
    assert found is not None, f"sensor {sensor_name!r} does not publish 'found'"
    if found.ndim == 3:  # (envs, primaries, slots)
        found = found.any(dim=-1)
    return found > 0


# Sign of the inside-of-hand gate. A contact counts as "inside" when the contact normal
# opposes the link's inward face, i.e. normal . inward < 0. The mjlab sensor normal points
# primary->secondary (link->object), which is the negative of the object->link normal the
# verified mdp.terms reference gates on -- hence the < 0. Confirmed empirically (a power grasp
# on a ball classifies nearly all pad contacts as inside; see the smoke test).
_INSIDE_ALIGN_MAX = 0.0


def _contact_inside(
    env, sensor_name: str = "hand_object", action_name: str = "grasp"
) -> torch.Tensor:
    """(num_envs, 16) bool: object<->link contact kept only if it loads the link's *inside* face.

    Rejects back-of-hand and knuckle contacts so the reward only credits contacts that could be
    part of a real grasp. Uses the sensor's world-frame contact ``normal`` and each reward
    link's inward axis (from the action term), in REWARD_CONTACT_LINKS order.
    """
    from rophi_rl_tutorial.controllers.grasping.batched import quat_rotate

    term = env.action_manager.get_term(action_name)
    perm = _reward_link_order(env, sensor_name)
    data = env.scene[sensor_name].data
    found = _contact_found(env, sensor_name)[:, perm]  # (N, 16) bool
    normal = data.normal
    assert (
        normal is not None
    ), f"sensor {sensor_name!r} must publish 'normal' for the inside gate"
    if normal.ndim == 4:  # (envs, primaries, slots, 3)
        normal = normal[:, :, 0]
    normal = normal[:, perm]  # (N, 16, 3) world, link->object
    quat = env.scene["robot"].data.body_link_quat_w[
        :, term._reward_body_ids
    ]  # (N, 16, 4)
    inward_w = quat_rotate(
        quat, term._inward_local.unsqueeze(0).expand_as(quat[..., :3])
    )
    aligns = (normal * inward_w).sum(dim=-1)
    return found & (aligns < _INSIDE_ALIGN_MAX)


def _min_step(
    best: torch.Tensor, error: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    seeded = torch.isfinite(best)
    reward = torch.zeros_like(error)
    eligible = mask & seeded
    reward[eligible] = (best[eligible] - error[eligible]).clamp(min=0.0)
    improved = mask & ((~seeded) | (error < best))
    best[improved] = error[improved]
    return reward


##
# Observations.
##


def contact_binary(env, sensor_name: str = "hand_object") -> torch.Tensor:
    return _contact_found(env, sensor_name).float()


# Fingertip = the distal (*_4) phalanx of each of the five fingers within REWARD_CONTACT_LINKS.
_FINGERTIP_CONTACT_IDX = tuple(
    i for i, n in enumerate(REWARD_CONTACT_LINKS) if n.endswith("4")
)


def fingertip_contact_binary(env, sensor_name: str = "hand_object") -> torch.Tensor:
    """Just the 5 fingertip binary contacts (thumb/index/middle/ring/little tips).

    The reduced-observation the distillation *student* gets: the real hand only instruments its
    fingertips, so the student must grasp from fingertip touch + proprioception alone, without the
    palm->object vector or the proximal-phalanx contacts the privileged teacher sees.
    """
    return contact_binary(env, sensor_name)[:, _FINGERTIP_CONTACT_IDX]


def _grasp_site_id(env, site_name: str = "grasp_site") -> int:
    """Index of the palm site within the robot entity, cached on the env."""
    key = f"_grasp_site_{site_name}"
    if not hasattr(env, key):
        ids, _ = env.scene["robot"].find_sites([site_name])
        setattr(env, key, int(ids[0]))
    return getattr(env, key)


def palm_object_vector(env, site_name: str = "grasp_site", object_name: str = "object"):
    robot = env.scene["robot"]
    sid = _grasp_site_id(env, site_name)
    palm_pos = robot.data.site_pos_w[:, sid]
    palm_quat = robot.data.site_quat_w[:, sid]
    obj = env.scene[object_name].data.root_link_pos_w
    return quat_rotate(quat_conjugate(palm_quat), obj - palm_pos)


def hand_joint_pos(env) -> torch.Tensor:
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    return robot.data.joint_pos[:, ids]


def hand_tracking_error(env, action_name: str = "grasp") -> torch.Tensor:
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    term = env.action_manager.get_term(action_name)
    return term.hand_target - robot.data.joint_pos[:, ids]


def object_pos(env, object_name: str = "object") -> torch.Tensor:
    return env.scene[object_name].data.root_link_pos_w


def object_lin_vel(env, object_name: str = "object") -> torch.Tensor:
    return env.scene[object_name].data.root_link_lin_vel_w


def object_height_above_rest(env, object_name: str = "object") -> torch.Tensor:
    s = _state(env)
    z = env.scene[object_name].data.root_link_pos_w[:, 2]
    return (z - s["rest_z"]).unsqueeze(-1)


##
# Rewards.
##


def contact_coverage_improvement(env, sensor_name: str = "hand_object") -> torch.Tensor:
    """Improvement watermark on weighted inside-of-hand contact coverage (IsaacLab weights).

    Only contacts loading the *inside* of each hand link count (see :func:`_contact_inside`), so
    the reward credits forming a real grasp, not brushing the object with the back of the hand.
    Pays ``max(best_uncovered - uncovered, 0)`` and lowers the watermark, so it telescopes to
    total coverage gained over the episode and cannot be farmed by making and breaking contact.
    """
    s = _state(env)
    inside = _contact_inside(
        env, sensor_name
    ).float()  # (N, 16), already in reward-link order
    uncovered = ((1.0 - inside) * s["weights"].unsqueeze(0)).sum(dim=1)
    return _min_step(
        s["contact_best"], uncovered, torch.ones_like(uncovered, dtype=torch.bool)
    )


def object_height_reward(
    env, settle_steps: int = 5, object_name: str = "object"
) -> torch.Tensor:
    """Dense reward for how high the object is lifted above its settled rest height.

    Deliberately height, not distance-to-goal. A goal pose has a horizontal component, so a
    distance reward is gamed by sliding the object along the mat toward it without ever
    lifting. Height cannot be gamed that way -- sliding does not change z.

    Dense (paid every step, not a one-shot or an improvement watermark), so it rewards
    *holding* the object aloft, which favours a stable sustained grasp over a brief upward
    knock. Measured against the settled rest height so the object's initial drop onto the mat
    is not credited.
    """
    s = _state(env)
    z = env.scene[object_name].data.root_link_pos_w[:, 2]
    pending = ~s["rest_ready"]
    s["rest_count"][pending] += 1
    capture = pending & (s["rest_count"] >= settle_steps)
    s["rest_z"][capture] = z[capture]
    s["rest_ready"][capture] = True

    lift = (z - s["rest_z"]).clamp(min=0.0)
    return lift * s["rest_ready"].float()


def hold_at_goal_reward(
    env,
    threshold: float = 0.15,
    action_name: str = "grasp",
    object_name: str = "object",
) -> torch.Tensor:
    """Bonus each step the object is held within ``threshold`` of the end-effector goal pose.

    Discourages the drop-and-re-grasp behaviour: under the dense height reward alone the policy
    can farm height by repeatedly lifting and dropping, since the height signal is the same
    whether the object is held or bouncing. This pays only while the object stays *at* the
    goal, so dropping it out of the threshold immediately forfeits the reward.

    Slide-proof, unlike the earlier distance-to-goal reward: the goal pose is the elevated LIFT
    retreat pose (z ~ 0.21 m), so an object on the mat is always >0.2 m away -- reaching the
    threshold requires actually lifting and carrying it there, not sliding along the ground.
    """
    goal = env.action_manager.get_term(action_name).home_goal_pos  # (3,)
    obj = env.scene[object_name].data.root_link_pos_w  # (N, 3)
    dist = torch.linalg.norm(obj - goal.unsqueeze(0), dim=-1)
    return (dist < threshold).float()


def _hold_steps(env, hold_time_s: float) -> int:
    return max(1, round(hold_time_s / env.step_dt))


def held_at_goal_success(
    env,
    threshold: float = 0.15,
    hold_time_s: float = 2.0,
    action_name: str = "grasp",
    object_name: str = "object",
) -> torch.Tensor:
    """Termination: object held within ``threshold`` of the goal for ``hold_time_s`` straight.

    Owns the consecutive-hold counter (runs before the reward manager each step), so
    :func:`success_bonus` just reads it. The streak resets to zero the moment the object
    leaves the threshold, so the hold must be continuous, not cumulative.
    """
    s = _state(env)
    goal = env.action_manager.get_term(action_name).home_goal_pos
    obj = env.scene[object_name].data.root_link_pos_w
    in_thr = torch.linalg.norm(obj - goal.unsqueeze(0), dim=-1) < threshold
    s["hold_streak"] = torch.where(
        in_thr, s["hold_streak"] + 1, torch.zeros_like(s["hold_streak"])
    )
    return s["hold_streak"] >= _hold_steps(env, hold_time_s)


def success_bonus(env, hold_time_s: float = 2.0) -> torch.Tensor:
    """One-shot bonus on the step the hold streak reaches the success duration.

    Reads the streak that :func:`held_at_goal_success` already advanced this step. It fires
    exactly once because the episode terminates on the same step. Sized (via the reward
    weight) to exceed the most hold reward obtainable by *avoiding* success -- otherwise the
    early termination would make the policy hold just short of the threshold on purpose.
    """
    s = _state(env)
    return (s["hold_streak"] >= _hold_steps(env, hold_time_s)).float()


##
# Grasp-quality rewards (net contact wrench + grasp matrix).
#
# Both read a *dedicated* contact sensor ``hand_object_grasp`` (separate from the grasp-tuned
# ``hand_object`` sensor so nothing here can perturb the existing contact/coverage rewards). That
# sensor publishes, per hand-link<->object contact, the WORLD-frame contact force, the world
# contact point, and the world normal -- everything needed to (a) sum the net wrench the hand
# applies to the object and (b) assemble the grasp matrix.
##

_GRASP_SENSOR = "hand_object_grasp"


def _model_tensor(env, path: str) -> torch.Tensor:
    """Read a (possibly per-world) mujoco-warp model array as a torch tensor, expanded to B rows.

    The warp bridge exposes model arrays as torch-backed proxies whose leading dim is nworld for
    variant fields (per-env mass/inertia -- this project's objects differ per world) but may be 1
    for a shared field. Indexing materialises a real tensor; we broadcast the size-1 case to B.
    """
    obj = env.sim.model
    for attr in path.split("."):
        obj = getattr(obj, attr)
    t = obj[:]  # TorchArray.__getitem__ -> torch.Tensor (shares memory)
    if t.shape[0] == 1 and env.num_envs > 1:
        t = t.expand(env.num_envs, *t.shape[1:])
    return t


def _object_mass_gravity(env, object_name: str = "object"):
    """Per-env object mass [B], world gravity vector [B,3], and weight magnitude m*|g| [B]."""
    bid = env.scene[object_name].data.indexing.root_body_id
    mass = _model_tensor(env, "body_mass")[:, bid]  # [B]
    g = _model_tensor(env, "opt.gravity")  # [B, 3]
    weight_mag = (mass * torch.linalg.norm(g, dim=-1)).clamp(min=1e-6)  # [B]
    return mass, g, weight_mag


def _char_length(env, object_name: str = "object") -> torch.Tensor:
    """Per-env characteristic object length [B] from the inertia: sqrt(tr(I)/m) ~ a size scale.

    Used to non-dimensionalise moment arms so the net-wrench torque term and the grasp matrix are
    comparable across the 78 YCB objects, which span an order of magnitude in size.
    """
    bid = env.scene[object_name].data.indexing.root_body_id
    inertia = _model_tensor(env, "body_inertia")[:, bid]  # [B, 3] principal diagonal
    mass = _model_tensor(env, "body_mass")[:, bid].clamp(min=1e-6)  # [B]
    return torch.sqrt((inertia.sum(dim=-1) / mass).clamp(min=1e-9))  # [B]


def _grasp_contacts_world(env, sensor_name: str = _GRASP_SENSOR):
    """(found, force, pos, normal) for every hand<->object contact, flattened to [B, M(, 3)] world.

    ``found`` [B, M] bool, ``force`` [B, M, 3] world-frame contact force, ``pos`` [B, M, 3] world
    contact point, ``normal`` [B, M, 3] world unit normal (hand-link -> object). M = 16 links x
    num_slots. Non-contact slots are False in ``found`` and (masked to) zero elsewhere.
    """
    data = env.scene[sensor_name].data
    b = env.num_envs
    found = (data.found > 0).reshape(b, -1)  # [B, M]
    force = torch.nan_to_num(data.force).reshape(b, -1, 3)  # [B, M, 3]
    pos = torch.nan_to_num(data.pos).reshape(b, -1, 3)  # [B, M, 3]
    normal = torch.nan_to_num(data.normal).reshape(b, -1, 3)  # [B, M, 3]
    return found, force, pos, normal


def _grasped_gate(
    env, found, lift_thresh: float, min_contacts: int, object_name: str = "object"
):
    """True where the object is lifted off its rest height AND held by >= min_contacts contacts.

    Gates the grasp-quality rewards so they only credit a genuine in-air hold. Without the lift
    gate an object resting on the mat trivially satisfies "net wrench ~ 0" (the table, not the
    hand, balances gravity), and the grasp matrix would be scored on incidental brushing contacts.
    """
    s = _state(env)
    z = env.scene[object_name].data.root_link_pos_w[:, 2]
    lifted = s["rest_ready"] & ((z - s["rest_z"]) > lift_thresh)
    return lifted & (found.sum(dim=1) >= min_contacts)


def _net_wrench(env, object_name: str = "object"):
    """Core of the net-wrench reward, shared with the viewer readout.

    Returns the world-frame net force + torque the hand contacts and gravity apply to the object
    about its COM, plus the weight magnitude and size scale used to non-dimensionalise them:
    ``(net_force[B,3], net_torque[B,3], weight_mag[B], length[B], found[B,M])``.
    """
    found, force, pos, _ = _grasp_contacts_world(env)
    mask = found.unsqueeze(-1).float()  # [B, M, 1]
    force = force * mask
    com = env.scene[object_name].data.root_com_pos_w.unsqueeze(1)  # [B, 1, 3]
    r = (pos - com) * mask  # [B, M, 3] moment arm from COM
    contact_force = force.sum(dim=1)  # [B, 3]
    contact_torque = torch.cross(r, force, dim=-1).sum(dim=1)  # [B, 3]
    mass, g, weight_mag = _object_mass_gravity(env)
    net_force = contact_force + mass.unsqueeze(-1) * g  # +m*g is the (downward) weight
    net_torque = contact_torque  # gravity acts at the COM -> no gravity torque
    return net_force, net_torque, weight_mag, _char_length(env), found


def net_wrench_components(env):
    """Viewer helper: ``(net_force[B,3], net_torque[B,3], force_resid[B], torque_resid[B])`` where
    the residuals are the weight-normalised imbalances the net-wrench reward drives to zero.
    """
    net_force, net_torque, weight_mag, length, _ = _net_wrench(env)
    force_resid = torch.linalg.norm(net_force, dim=-1) / weight_mag
    torque_resid = torch.linalg.norm(net_torque, dim=-1) / (weight_mag * length)
    return net_force, net_torque, force_resid, torque_resid


def net_wrench_reward(
    env,
    sharpness: float = 3.0,
    torque_weight: float = 0.3,
    lift_thresh: float = 0.03,
    min_contacts: int = 2,
) -> torch.Tensor:
    """Reward a *balanced* in-hand hold: the net wrench the hand + gravity apply to the object -> 0.

    Sums the measured world-frame contact wrench about the object COM and adds the gravity wrench
    (m*g at the COM, zero torque). At a stable static hold this net wrench is zero -- the hand's
    contact forces exactly bear the object's weight with no residual force/torque accelerating it.
    Normalised by the object weight (and a size scale for the torque term) so it is comparable
    across YCB objects, then mapped through exp(-sharpness * residual) to [0, 1].

    Note this is an *equilibrium* signal, not a force-closure one: it says "currently held steady,"
    which is why gravity is included (so it fires only when the HAND bears the weight, not when the
    object rests on the mat) and why it is complementary to :func:`grasp_matrix_quality_reward`,
    which scores robustness to disturbance.
    """
    net_force, net_torque, weight_mag, length, found = _net_wrench(env)
    residual = torch.linalg.norm(net_force, dim=-1) / weight_mag + torque_weight * (
        torch.linalg.norm(net_torque, dim=-1) / (weight_mag * length)
    )
    gate = _grasped_gate(env, found, lift_thresh, min_contacts).float()
    return torch.exp(-sharpness * residual) * gate


def _skew(v: torch.Tensor) -> torch.Tensor:
    """Batched skew-symmetric matrix of ``v`` [..., 3] -> [..., 3, 3] (so skew(v) @ w = v x w)."""
    zero = torch.zeros_like(v[..., 0])
    vx, vy, vz = v[..., 0], v[..., 1], v[..., 2]
    row0 = torch.stack([zero, -vz, vy], dim=-1)
    row1 = torch.stack([vz, zero, -vx], dim=-1)
    row2 = torch.stack([-vy, vx, zero], dim=-1)
    return torch.stack([row0, row1, row2], dim=-2)


def _frame_from_normal(n: torch.Tensor) -> torch.Tensor:
    """Orthonormal contact frame [..., 3, 3] with columns (n, t1, t2) from a unit normal n."""
    ex = torch.zeros_like(n)
    ex[..., 0] = 1.0
    ey = torch.zeros_like(n)
    ey[..., 1] = 1.0
    axis = torch.where((n[..., 0].abs() > 0.9).unsqueeze(-1), ey, ex)
    t1 = torch.cross(n, axis, dim=-1)
    t1 = t1 / (t1.norm(dim=-1, keepdim=True) + 1e-9)
    t2 = torch.cross(n, t1, dim=-1)
    return torch.stack([n, t1, t2], dim=-1)


def grasp_matrix_metrics(env, min_contacts: int = 3):
    """The three classical grasp-matrix (SVD) quality measures, per env. Each [B].

    Builds the point-contact-with-friction grasp matrix G in R^{6x3n} (Roa & Suarez 2015): each
    contact contributes a 6x3 block [[R],[skew(r/L) R]] mapping a contact-frame force to the object
    wrench about the COM, with moment arms r non-dimensionalised by the object size L. From the
    singular values sigma_1..6 of G it returns:

    * Q_MSV  = sigma_min      -- worst-case: the grasp's ability to resist a wrench in its weakest
      direction (0 if the contacts fail to span the 6-D wrench space).
    * Q_VEW  = prod(sigma_i)  -- volume of the wrench ellipsoid (average-case capability).
    * Q_GII  = sigma_min/sigma_max -- isotropy: how uniformly the grasp resists across directions.

    These are grasp-*matrix* metrics: cheap, batchable, and dominated by how well the contacts are
    distributed around the object. They ignore the friction-cone unilaterality that true
    force-closure (Ferrari-Canny epsilon / FRoGGeR min-weight) enforces -- see the module notes.
    """
    found, _, pos, normal = _grasp_contacts_world(env)
    b, m = found.shape
    mask = found.float()  # [B, M]
    com = env.scene["object"].data.root_com_pos_w.unsqueeze(1)  # [B, 1, 3]
    length = _char_length(env).clamp(min=1e-3).view(b, 1, 1)  # [B, 1, 1]
    r = (pos - com) / length  # [B, M, 3] non-dimensional moment arm
    n = normal / (normal.norm(dim=-1, keepdim=True) + 1e-9)  # [B, M, 3] unit
    rot = _frame_from_normal(n)  # [B, M, 3, 3]
    block = torch.cat([rot, torch.matmul(_skew(r), rot)], dim=2)  # [B, M, 6, 3]
    block = block * mask.view(
        b, m, 1, 1
    )  # zero non-contact columns (don't change sigma)
    gram = torch.einsum("bmij,bmkj->bik", block, block)  # G G^T, [B, 6, 6]
    # cusolver's batched eigvalsh intermittently fails to converge on an ill-conditioned or
    # eigenvalue-degenerate grasp matrix (a near-singular contact configuration -- e.g. a few
    # collinear contacts). Condition it with a tiny diagonal jitter (negligible vs the O(0.1-1)
    # eigenvalues that matter), and fall back to the robust CPU/double LAPACK path, then to zeros,
    # so one degenerate env can never crash a whole training run or the cross-eval report.
    gram = torch.nan_to_num(gram) + 1e-4 * torch.eye(
        6, device=gram.device, dtype=gram.dtype
    )
    try:
        eig = torch.linalg.eigvalsh(gram)
    except Exception:
        try:
            eig = torch.linalg.eigvalsh(gram.double().cpu()).to(
                device=gram.device, dtype=gram.dtype
            )
        except Exception:
            eig = torch.zeros(b, 6, device=gram.device, dtype=gram.dtype)
    eig = eig.clamp(min=0.0)  # [B, 6] ascending
    sigma = eig.sqrt()
    enough = found.sum(dim=1) >= min_contacts
    q_msv = sigma[:, 0] * enough
    q_vew = sigma.prod(dim=1) * enough
    q_gii = torch.where(
        sigma[:, -1] > 1e-9, sigma[:, 0] / sigma[:, -1], torch.zeros_like(sigma[:, 0])
    )
    q_gii = q_gii * enough
    return q_msv, q_vew, q_gii


def grasp_matrix_quality_reward(
    env, scale: float = 1.2, lift_thresh: float = 0.03, min_contacts: int = 3
) -> torch.Tensor:
    """Reward grasp-matrix quality Q_MSV (min singular value of G), bounded to [0, 1) via tanh.

    Rewards forming a grasp whose contacts are distributed so the hand can exert a wrench in every
    direction -- contacts clustered on one side (which a disturbance could peel off) score near
    zero. Gated on a genuine lifted, multi-contact hold. ``scale`` sets the Q_MSV at which the
    reward reaches tanh(1)~0.76; calibrated from measured ranges on held YCB objects.
    """
    q_msv, _, _ = grasp_matrix_metrics(env, min_contacts=min_contacts)
    found, *_ = _grasp_contacts_world(env)
    gate = _grasped_gate(env, found, lift_thresh, min_contacts).float()
    return torch.tanh(q_msv / scale) * gate


def _grasp_metric_sensor_cfg() -> ContactSensorCfg:
    """Dedicated world-frame contact sensor for the grasp-quality rewards (separate from the
    grasp-tuned ``hand_object`` sensor). Publishes world force + contact point + normal per
    hand-link<->object contact; ``global_frame`` needs ``normal`` and ``tangent`` in fields.
    """
    return ContactSensorCfg(
        name=_GRASP_SENSOR,
        primary=ContactMatch(
            mode="body", pattern=f"^({'|'.join(REWARD_CONTACT_LINKS)})$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="^object$", entity="object"),
        fields=("found", "force", "normal", "tangent", "pos"),
        global_frame=True,
        reduce="none",
        num_slots=3,
    )


# The grasp-quality reward terms, keyed by the --grasp-reward name train_grasp.py exposes. Not in
# the default reward stack (the reward manager skips weight-0 terms, so they must be injected with
# a real weight); set_grasp_metric_reward puts the chosen one in with the given weight.
_GRASP_REWARD_FUNCS = {
    "wrench": net_wrench_reward,
    "matrix": grasp_matrix_quality_reward,
}


def set_grasp_metric_reward(env_cfg, method: str, weight: float) -> None:
    """Inject one grasp-quality reward term into an env cfg (used by train_grasp.py --grasp-reward).

    Adds it *on top of* the existing contact/height/hold/success stack: these metrics only produce
    signal once the hand already holds the object, so they refine grasp quality rather than replace
    the shaping that teaches grasping in the first place.
    """
    if method == "none":
        return
    func = _GRASP_REWARD_FUNCS[method]
    env_cfg.rewards[f"grasp_{method}"] = RewardTermCfg(
        func=func, params={}, weight=weight
    )


def success_held_to_end(
    env,
    threshold: float = 0.15,
    action_name: str = "grasp",
    object_name: str = "object",
) -> torch.Tensor:
    """One-shot success bonus on the episode's final step if the object is still held at the goal.

    The alternative to :func:`success_bonus` for a *hold-as-long-as-possible* task: instead of
    firing (and terminating) after a fixed 2 s hold -- which teaches the policy it can cash out
    early and then slip -- this pays only when the object is STILL within ``threshold`` of the goal
    at the moment the episode times out. Combined with dropping the early ``success`` termination
    and the dense per-step hold reward, the policy is rewarded for holding all the way to the end.

    Fires exactly once: ``episode_length_buf`` reaches ``max_episode_length`` only on the final
    step (it is incremented before rewards are computed), and the env resets immediately after.
    """
    goal = env.action_manager.get_term(action_name).home_goal_pos
    obj = env.scene[object_name].data.root_link_pos_w
    held = torch.linalg.norm(obj - goal.unsqueeze(0), dim=-1) < threshold
    at_end = env.episode_length_buf >= env.max_episode_length
    return (held & at_end).float()


def _hand_pca(env):
    """Lazily-loaded, env-cached HandPCA (the fitted 5-DoF human-grasp manifold)."""
    if not hasattr(env, "_hand_pca_cache"):
        from rophi_rl_tutorial.assets.grasping.hand_pca import load_hand_pca

        env._hand_pca_cache = load_hand_pca(device=env.device)
    return env._hand_pca_cache


def hand_pca_manifold_reward(
    env, sharpness: float = 1.0, lift_thresh: float = 0.03, min_contacts: int = 3
) -> torch.Tensor:
    """Reward the hand joint pose lying ON the fitted PCA grasp manifold (natural human-grasp shapes).

    ``||q - reconstruct(q)||`` is the distance of the 20-joint pose off the PCA manifold (reconstruct
    = decode(encode(q)) projects onto it); ``exp(-sharpness * dist)`` is 1 on the manifold and decays
    off it. Gated on a real held grasp so it shapes the *grasp* pose, not the open reach pose. Used to
    fine-tune a joint-space policy toward PCA-like hand shapes without changing its action space.
    """
    pca = _hand_pca(env)
    robot = env.scene["robot"]
    ids, _ = robot.find_joints(list(HAND_JOINT_NAMES), preserve_order=True)
    q = robot.data.joint_pos[:, ids]
    dist = torch.linalg.norm(q - pca.reconstruct(q), dim=-1)
    found, *_ = _grasp_contacts_world(env)
    gate = _grasped_gate(env, found, lift_thresh, min_contacts).float()
    return torch.exp(-sharpness * dist) * gate


##
# Task assembly.
##


def make_task(
    objects: str | Sequence[str] = "v013_apple",
    num_envs: int = 4096,
    squash: str = "sigmoid",
    source: str = "ycb",
    max_collision_hulls: int = 16,
) -> ManagerBasedRlEnvCfg:
    robot = get_rizon4_hand_robot_cfg()
    # A single name spawns that object in every world; a list spawns a *per-world mesh variant*
    # (a different object per world, evenly allocated) so one policy trains across the set. The
    # object stays one free body named "object" either way, so nothing downstream changes.
    # `source` selects the prepared asset tree the names index into: "ycb" (v0xx_* names) or
    # "objaverse" (GraspXL UID names); both expose the identical single/variant entity configs.
    if source == "objaverse":
        from rophi_rl_tutorial.assets.grasping.objaverse import (
            get_objaverse_object_cfg as single_cfg,
        )
        from rophi_rl_tutorial.assets.grasping.objaverse import (
            get_objaverse_variant_object_cfg as variant_cfg,
        )
    elif source == "primitives":
        from rophi_rl_tutorial.assets.grasping.primitives import (
            get_primitive_object_cfg as single_cfg,
        )
        from rophi_rl_tutorial.assets.grasping.primitives import (
            get_primitive_variant_object_cfg as variant_cfg,
        )
    elif source == "gso":
        from rophi_rl_tutorial.assets.grasping.gso import (
            get_gso_object_cfg as single_cfg,
        )
        from rophi_rl_tutorial.assets.grasping.gso import (
            get_gso_variant_object_cfg as variant_cfg,
        )
    elif source == "ycb":
        single_cfg, variant_cfg = get_ycb_object_cfg, get_ycb_variant_object_cfg
    else:
        raise ValueError(
            f"Unknown object source {source!r} (expected 'ycb', 'objaverse', 'primitives' or 'gso')."
        )
    names = (objects,) if isinstance(objects, str) else tuple(objects)
    if len(names) == 1:
        obj = single_cfg(
            names[0], pos=(*SPAWN_XY, SPAWN_Z), max_collision_hulls=max_collision_hulls
        )
    else:
        obj = variant_cfg(
            names, pos=(*SPAWN_XY, SPAWN_Z), max_collision_hulls=max_collision_hulls
        )

    def add_ground(spec):
        import mujoco

        spec.worldbody.add_geom(
            name="floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=(3.0, 3.0, 0.1),
            pos=(0.0, 0.0, FLOOR_HEIGHT),
            condim=6,
            friction=(1.0, 5e-3, 1e-4),
        )
        spec.worldbody.add_geom(
            name="mat",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=(*MAT_HALF_EXTENT, MAT_THICKNESS * 0.5),
            pos=(*MAT_CENTRE_XY, FLOOR_HEIGHT + MAT_THICKNESS * 0.5),
            condim=6,
            friction=(1.2, 5e-3, 1e-4),
            solref=(0.01, 1.0),
        )

    contact = ContactSensorCfg(
        name="hand_object",
        # Patterns are regexes matched against names *within* the entity, so they are
        # unprefixed. The parentheses matter: "^a|b|c$" parses as "^a" OR "b" OR "c$".
        primary=ContactMatch(
            mode="body", pattern=f"^({'|'.join(REWARD_CONTACT_LINKS)})$", entity="robot"
        ),
        secondary=ContactMatch(mode="body", pattern="^object$", entity="object"),
        # "normal" (world frame, primary->secondary) feeds the inside-of-hand contact gate.
        fields=("found", "force", "normal"),
        reduce="none",
        num_slots=1,
    )

    scene = SceneCfg(
        num_envs=num_envs,
        env_spacing=2.5,
        entities={"robot": robot, "object": obj},
        # Two contact sensors over the same hand<->object pairs: ``hand_object`` (grasp-tuned:
        # found/normal for the coverage reward + contact obs) and ``hand_object_grasp`` (world
        # force/point/normal for the net-wrench + grasp-matrix quality rewards). Kept separate so
        # the grasp-quality additions cannot perturb the existing, tuned contact pipeline.
        sensors=(contact, _grasp_metric_sensor_cfg()),
        spec_fn=add_ground,
    )

    actor_terms = {
        "contacts": ObservationTermCfg(func=contact_binary, params={}),
        "palm_to_object": ObservationTermCfg(func=palm_object_vector, params={}),
        "hand_joint_pos": ObservationTermCfg(func=hand_joint_pos, params={}),
        "hand_tracking": ObservationTermCfg(func=hand_tracking_error, params={}),
    }
    critic_terms = dict(actor_terms)
    critic_terms.update(
        {
            "object_pos": ObservationTermCfg(func=object_pos, params={}),
            "object_lin_vel": ObservationTermCfg(func=object_lin_vel, params={}),
            "object_height": ObservationTermCfg(
                func=object_height_above_rest, params={}
            ),
        }
    )

    return ManagerBasedRlEnvCfg(
        decimation=20,
        episode_length_s=5.0,
        scene=scene,
        # Defaults overflow: 16 object hulls against 21 hand links plus the mat generates far
        # more contacts and constraint rows than the bundled tasks, and an nefc overflow
        # corrupts the solve rather than erroring.
        sim=SimulationCfg(nconmax=400, njmax=1200),
        actions={"grasp": GraspActionCfg(squash=squash)},
        observations={
            "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True),
            "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True),
        },
        rewards={
            # Inside-of-hand contact coverage (IsaacLab weights: thumb tip 10, thumb mids 5,
            # palm 2, rest 1), weight 0.1 as in IsaacLab. Dense shaping toward wrapping fingers
            # onto the object -- object-agnostic, so it is what lets one policy learn to grasp a
            # diverse set rather than memorising a single shape's lift.
            "contact": RewardTermCfg(
                func=contact_coverage_improvement, params={}, weight=0.1
            ),
            "height": RewardTermCfg(func=object_height_reward, params={}, weight=100.0),
            # Hold-at-goal bonus: rewards keeping the object at the elevated retreat pose,
            # which penalises drop-and-re-grasp (dropping leaves the threshold) and cannot be
            # slid for on the ground. weight 50 -> ~2/step held; height still gives the lift
            # gradient before the object first reaches the goal.
            "hold": RewardTermCfg(
                func=hold_at_goal_reward, params={"threshold": 0.15}, weight=50.0
            ),
            "success": RewardTermCfg(
                func=success_bonus, params={"hold_time_s": 2.0}, weight=5000.0
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
            # Real termination (time_out=False -> GAE cuts the value bootstrap), so the policy
            # learns holding the object at the goal for 2 s is the terminal goal, not a state
            # to prolong.
            "success": TerminationTermCfg(
                func=held_at_goal_success,
                params={"threshold": 0.15, "hold_time_s": 2.0},
                time_out=False,
            ),
            # Safety reset for the intermittent contact/solver blow-ups: a world whose object has
            # gone non-finite or been flung fast/far is reset in place. time_out=True so GAE
            # treats it as a truncation (a sim glitch, not a policy failure to penalise). Because
            # the env recomputes observations *after* the reset, catching the divergence here --
            # ideally while the state is still finite -- keeps the NaN out of the returned obs.
            "instability": TerminationTermCfg(func=_unstable, time_out=True),
        },
        events={
            # Required for a fixed-base robot: without it every env's robot sits at the
            # world origin instead of its own env origin.
            "reset_robot_base": EventTermCfg(
                func=_reset_robot_base,
                mode="reset",
                params={},
            ),
            # Overriding `events` drops mjlab's default reset_scene_to_default, so without
            # this the arm resets to all-zero joints (pointing straight up, grasp_site at
            # z=1.25) instead of ARM_Q_HOME.
            "reset_robot_joints": EventTermCfg(
                func=_reset_robot_joints, mode="reset", params={}
            ),
            "reset_object": EventTermCfg(
                func=_reset_object,
                mode="reset",
                params={
                    "pose_range": {
                        "x": (-0.05, 0.05),
                        "y": (-0.05, 0.05),
                        "yaw": (-3.14, 3.14),
                    }
                },
            ),
            "reset_state": EventTermCfg(
                func=reset_grasp_state, mode="reset", params={}
            ),
        },
    )


def _time_out(env) -> torch.Tensor:
    from mjlab.envs.mdp.terminations import time_out

    return time_out(env)


def _unstable(env, max_speed: float = 15.0, max_dist: float = 2.0) -> torch.Tensor:
    """Fire on a diverged world so it gets reset before its blow-up poisons training.

    A contact/solver instability sends the object non-finite or flings it fast (``max_speed``
    m/s) or far (``max_dist`` m from its env origin). Caught here -- usually while the state is
    still finite -- the world resets and its observation is recomputed clean; the safety wrapper
    in ``train_grasp.py`` sanitises the one-step case where it goes straight to NaN.
    """
    data = env.scene["object"].data
    pos = data.root_link_pos_w
    vel = data.root_link_lin_vel_w
    bad = ~torch.isfinite(pos).all(dim=-1) | ~torch.isfinite(vel).all(dim=-1)
    bad = bad | (torch.linalg.norm(vel, dim=-1) > max_speed)
    bad = bad | (torch.linalg.norm(pos, dim=-1) > max_dist)
    return bad


def _reset_robot_base(env, env_ids) -> None:
    from mjlab.envs.mdp.events import reset_root_state_uniform

    reset_root_state_uniform(
        env,
        env_ids,
        pose_range={},
        velocity_range={},
        asset_cfg=SceneEntityCfg("robot"),
    )


def _reset_robot_joints(env, env_ids) -> None:
    from mjlab.envs.mdp.events import reset_joints_by_offset

    reset_joints_by_offset(
        env,
        env_ids,
        position_range=(0.0, 0.0),
        velocity_range=(0.0, 0.0),
        asset_cfg=SceneEntityCfg("robot", joint_names=(".*",)),
    )


def _reset_object(env, env_ids, pose_range) -> None:
    from mjlab.envs.mdp.events import reset_root_state_uniform

    reset_root_state_uniform(
        env,
        env_ids,
        pose_range=pose_range,
        velocity_range={},
        asset_cfg=SceneEntityCfg("object"),
    )


##
# RL config + registration.
##


def grasp_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(256, 128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(256, 128, 64), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            # Stability fixes after run 3 peaked at iter ~200 then decayed. Diagnosis: the
            # entropy bonus inflated the policy std from 0.33 to 0.84 over training, which
            # made lifting unreliable and grew return variance (value loss 24 -> 327). These
            # target that directly:
            #   entropy_coef 0.01 -> 0.002   stop the std from inflating
            #   desired_kl   0.01 -> 0.008   gentler per-update policy drift
            #   normalize_advantage_per_mini_batch  decouple updates from reward scale
            entropy_coef=0.004,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.008,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=True,
        ),
        experiment_name="rophi_grasp",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=2000,
    )


def register() -> None:
    from mjlab.tasks.registry import register_mjlab_task

    from rophi_rl_tutorial.assets.grasping.ycb import EASY_YCB_OBJECT_LIST, YCB_OBJECT_LIST

    register_mjlab_task(
        task_id="RoPhi-Grasp-Apple",
        env_cfg=make_task("v013_apple", num_envs=4096),
        play_env_cfg=make_task("v013_apple", num_envs=16),
        rl_cfg=grasp_ppo_cfg(),
    )
    # Diverse-object task: one policy over the EASY YCB set via per-world mesh variants (a
    # different object per world, evenly allocated). play uses one env per object so the
    # viewer shows the whole set at a glance.
    register_mjlab_task(
        task_id="RoPhi-Grasp-YCB",
        env_cfg=make_task(EASY_YCB_OBJECT_LIST, num_envs=4096),
        play_env_cfg=make_task(
            EASY_YCB_OBJECT_LIST, num_envs=len(EASY_YCB_OBJECT_LIST)
        ),
        rl_cfg=grasp_ppo_cfg(),
    )
    # Full YCB set (all 78 objects, including thin/degenerate ones a 5-finger hand can't grasp).
    # Same per-world-variant scaling: the compiled model is padded to the largest object, so
    # the scene is the same size as one object regardless of catalogue size.
    register_mjlab_task(
        task_id="RoPhi-Grasp-YCB-All",
        env_cfg=make_task(YCB_OBJECT_LIST, num_envs=4096),
        play_env_cfg=make_task(YCB_OBJECT_LIST, num_envs=len(YCB_OBJECT_LIST)),
        rl_cfg=grasp_ppo_cfg(),
    )
    # Objaverse task: one policy over the GraspXL example objects via the same per-world mesh
    # variants. Registered only when the prepared tree exists (scripts/grasping/fetch_objaverse.py), so a
    # missing dataset leaves the YCB tasks untouched rather than failing registration.
    from rophi_rl_tutorial.assets.grasping.objaverse import list_objaverse_objects

    objaverse_objects = list_objaverse_objects()
    if objaverse_objects:
        register_mjlab_task(
            task_id="RoPhi-Grasp-Objaverse",
            env_cfg=make_task(objaverse_objects, num_envs=4096, source="objaverse"),
            play_env_cfg=make_task(
                objaverse_objects, num_envs=len(objaverse_objects), source="objaverse"
            ),
            rl_cfg=grasp_ppo_cfg(),
        )

    # Procedural primitives: superquadric shapes (box <-> cylinder <-> ellipsoid <-> sphere) via
    # per-world variants. Fully generated, so always registered -- no dataset needed.
    from rophi_rl_tutorial.assets.grasping.primitives import PRIMITIVE_OBJECT_LIST

    register_mjlab_task(
        task_id="RoPhi-Grasp-Primitives",
        env_cfg=make_task(PRIMITIVE_OBJECT_LIST, num_envs=4096, source="primitives"),
        play_env_cfg=make_task(
            PRIMITIVE_OBJECT_LIST,
            num_envs=len(PRIMITIVE_OBJECT_LIST),
            source="primitives",
        ),
        rl_cfg=grasp_ppo_cfg(),
    )

    # GSO task: a curated, size-normalised graspable subset of Google Scanned Objects via per-world
    # variants. Registered only when the prepared tree exists (scripts/grasping/fetch_gso.py).
    from rophi_rl_tutorial.assets.grasping.gso import list_gso_objects

    gso_objects = list_gso_objects()
    if gso_objects:
        register_mjlab_task(
            task_id="RoPhi-Grasp-GSO",
            env_cfg=make_task(gso_objects, num_envs=4096, source="gso"),
            play_env_cfg=make_task(
                gso_objects, num_envs=len(gso_objects), source="gso"
            ),
            rl_cfg=grasp_ppo_cfg(),
        )
