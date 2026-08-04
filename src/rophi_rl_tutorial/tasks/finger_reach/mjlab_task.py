"""mjlab task: free-space index-fingertip velocity tracking (simplest RL example).

MDP
---
* **Scene** — Flexiv Rizon 4 + RH-5 alone (no table / mat / object).
* **Obs** — arm joint positions (7) + arm joint velocities (7) + target tip velocity (3).
* **Action** — arm joint velocities (7), raw policy output clamped to ``[-1, 1]``
  then scaled by ``max_joint_vel`` (default 0.5 rad/s). The action term tracks that
  ``qd`` with joint damping + gravity feedforward on the torque-actuated arm; the hand
  is held open.
* **Command** — 3-D world-frame target velocity for ``index_tip_site``, resampled on reset.
* **Reward** — match tip linear velocity to the command (exp of error) minus a
  self-collision penalty.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.entity.entity import Entity as _Entity
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

from rophi_rl_tutorial.assets.finger_reach import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    HAND_Q_OPEN,
    INDEX_TIP_SITE,
    get_finger_reach_robot_cfg,
)
from rophi_rl_tutorial.controllers.grasping.kinematics import GravityCompensation

# Per-axis tip velocity command range [m/s], world frame. Kept modest so the arm can track
# it under the joint-velocity action bound (default 0.5 rad/s).
CMD_VEL_MAX = 0.20
_CMD_VEL_MAX = CMD_VEL_MAX  # back-compat alias


##
# Action term: bounded arm joint velocity + hold hand open.
##


@dataclass(kw_only=True)
class ArmJointVelocityActionCfg(ActionTermCfg):
    """Policy outputs 7 arm joint velocities in ``[-1, 1]``, scaled to rad/s."""

    entity_name: str = "robot"
    max_joint_vel: float = 0.5
    """Per-joint velocity bound [rad/s] after scaling. Keeps motion from getting too fast."""
    arm_kd: tuple[float, ...] = (4.0, 4.0, 4.0, 4.0, 2.0, 2.0, 1.0)

    def build(self, env) -> ArmJointVelocityAction:
        return ArmJointVelocityAction(self, env)


class ArmJointVelocityAction(ActionTerm):
    """Track a bounded joint-velocity command with damping + gravity feedforward."""

    cfg: ArmJointVelocityActionCfg

    def __init__(self, cfg: ArmJointVelocityActionCfg, env) -> None:
        super().__init__(cfg=cfg, env=env)
        self._entity = env.scene[cfg.entity_name]
        # Standalone robot model so GravityCompensation indices match EntityData (no world/object).
        model = _Entity(get_finger_reach_robot_cfg()).spec.compile()
        self._gravity = GravityCompensation(
            model, ARM_JOINT_NAMES, device=env.device, body_offset=1
        )
        arm_ids, _ = self._entity.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        hand_ids, _ = self._entity.find_joints(HAND_JOINT_NAMES, preserve_order=True)
        self._arm_ids = torch.tensor(arm_ids, device=self.device, dtype=torch.long)
        self._hand_ids = torch.tensor(hand_ids, device=self.device, dtype=torch.long)
        q_open = [HAND_Q_OPEN[n] for n in HAND_JOINT_NAMES]
        self._hand_q_open = torch.tensor(
            q_open, device=self.device, dtype=torch.float32
        )
        self._arm_kd = torch.tensor(cfg.arm_kd, device=self.device, dtype=torch.float32)
        self._raw = torch.zeros(self.num_envs, 7, device=self.device)
        self._qd_cmd = torch.zeros(self.num_envs, 7, device=self.device)

    @property
    def action_dim(self) -> int:
        return 7

    @property
    def raw_action(self) -> torch.Tensor:
        return self._raw

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        self._qd_cmd = actions.clamp(-1.0, 1.0) * self.cfg.max_joint_vel

    def apply_actions(self) -> None:
        data = self._entity.data
        qd_meas = data.joint_vel[:, self._arm_ids]
        grav = self._gravity.compute(
            data.body_link_pos_w, data.body_link_quat_w, data.body_com_pos_w
        )
        # Velocity tracking: damping on (qd_cmd - qd) + gravity feedforward.
        tau = self._arm_kd * (self._qd_cmd - qd_meas) + grav
        self._entity.set_joint_effort_target(tau, joint_ids=self._arm_ids)
        self._entity.set_joint_position_target(
            self._hand_q_open.unsqueeze(0).expand(self.num_envs, -1),
            joint_ids=self._hand_ids,
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        self._raw[env_ids] = 0.0
        self._qd_cmd[env_ids] = 0.0


##
# Observations / rewards / events.
##


def _ensure_target_vel_buffer(env) -> torch.Tensor:
    """Lazy ``(num_envs, 3)`` world-frame tip-velocity command buffer on the env."""
    if not hasattr(env, "_fingertip_target_vel_w"):
        env._fingertip_target_vel_w = torch.zeros(env.num_envs, 3, device=env.device)
    return env._fingertip_target_vel_w


def _index_tip_site_id(env) -> int:
    if not hasattr(env, "_index_tip_site_id"):
        ids, _ = env.scene["robot"].find_sites([INDEX_TIP_SITE])
        env._index_tip_site_id = int(ids[0])
    return env._index_tip_site_id


def _index_tip_pos_w(env) -> torch.Tensor:
    return env.scene["robot"].data.site_pos_w[:, _index_tip_site_id(env)]


def _index_tip_vel_w(env) -> torch.Tensor:
    return env.scene["robot"].data.site_lin_vel_w[:, _index_tip_site_id(env)]


def _arm_joint_ids(env) -> torch.Tensor:
    if not hasattr(env, "_arm_joint_ids"):
        ids, _ = env.scene["robot"].find_joints(ARM_JOINT_NAMES, preserve_order=True)
        env._arm_joint_ids = torch.tensor(ids, device=env.device, dtype=torch.long)
    return env._arm_joint_ids


def arm_joint_pos(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_pos[:, _arm_joint_ids(env)]


def arm_joint_vel(env) -> torch.Tensor:
    return env.scene["robot"].data.joint_vel[:, _arm_joint_ids(env)]


def fingertip_target_vel_command(env) -> torch.Tensor:
    """Commanded index-tip linear velocity in the world frame [m/s]."""
    return _ensure_target_vel_buffer(env)


def vel_track_reward(env, sigma: float = 0.05) -> torch.Tensor:
    """Dense shaping: ``exp(-||v_tip - v_cmd|| / sigma)`` in ``(0, 1]``."""
    err = torch.linalg.norm(
        _index_tip_vel_w(env) - _ensure_target_vel_buffer(env), dim=-1
    )
    return torch.exp(-err / sigma)


def self_collision_penalty(env, sensor_name: str = "self_collision") -> torch.Tensor:
    """``-1`` when any robot body reports contact (self-collision in free space)."""
    found = env.scene[sensor_name].data.found  # (N, P)
    hit = (found > 0).any(dim=-1).float()
    return -hit


def sample_fingertip_target_vel(env, env_ids: torch.Tensor) -> None:
    """Uniform sample of tip velocity in ``[-CMD_VEL_MAX, CMD_VEL_MAX]^3`` [m/s]."""
    target = _ensure_target_vel_buffer(env)
    n = len(env_ids)
    u = torch.rand(n, 3, device=env.device) * 2.0 - 1.0
    target[env_ids] = u * CMD_VEL_MAX


def _time_out(env) -> torch.Tensor:
    from mjlab.envs.mdp.terminations import time_out

    return time_out(env)


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


##
# Task assembly.
##


def make_task(num_envs: int = 1024) -> ManagerBasedRlEnvCfg:
    robot = get_finger_reach_robot_cfg()

    # Any contact involving a robot body. With no floor/table/object this is self-collision.
    self_collision = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="body", pattern=".*", entity="robot"),
        secondary=None,
        fields=("found",),
        reduce="maxforce",
        num_slots=1,
    )

    scene = SceneCfg(
        num_envs=num_envs,
        env_spacing=2.5,
        entities={"robot": robot},
        sensors=(self_collision,),
        # No ground / table / object: free space.
    )

    obs_terms = {
        "arm_joint_pos": ObservationTermCfg(func=arm_joint_pos, params={}),
        "arm_joint_vel": ObservationTermCfg(func=arm_joint_vel, params={}),
        "target_vel": ObservationTermCfg(func=fingertip_target_vel_command, params={}),
    }

    return ManagerBasedRlEnvCfg(
        decimation=20,
        episode_length_s=4.0,
        scene=scene,
        sim=SimulationCfg(nconmax=200, njmax=600),
        actions={"arm_vel": ArmJointVelocityActionCfg()},
        observations={
            "actor": ObservationGroupCfg(terms=obs_terms, concatenate_terms=True),
            "critic": ObservationGroupCfg(
                terms=dict(obs_terms), concatenate_terms=True
            ),
        },
        rewards={
            "vel_track": RewardTermCfg(
                func=vel_track_reward, params={"sigma": 0.05}, weight=1.0
            ),
            "self_collision": RewardTermCfg(
                func=self_collision_penalty, params={}, weight=1.0
            ),
        },
        terminations={
            "time_out": TerminationTermCfg(func=_time_out, time_out=True),
        },
        events={
            "reset_robot_base": EventTermCfg(
                func=_reset_robot_base, mode="reset", params={}
            ),
            "reset_robot_joints": EventTermCfg(
                func=_reset_robot_joints, mode="reset", params={}
            ),
            "sample_target_vel": EventTermCfg(
                func=sample_fingertip_target_vel, mode="reset", params={}
            ),
        },
    )


def finger_reach_ppo_cfg() -> RslRlOnPolicyRunnerCfg:
    from mjlab.rl import RslRlModelCfg, RslRlPpoAlgorithmCfg

    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(128, 64),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 0.5,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(128, 64), activation="elu", obs_normalization=True
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
            normalize_advantage_per_mini_batch=True,
        ),
        experiment_name="rophi_finger_reach",
        logger="tensorboard",
        save_interval=50,
        num_steps_per_env=24,
        max_iterations=500,
    )


def register() -> None:
    from mjlab.tasks.registry import register_mjlab_task

    register_mjlab_task(
        task_id="RoPhi-Finger-Reach",
        env_cfg=make_task(num_envs=1024),
        play_env_cfg=make_task(num_envs=1),
        rl_cfg=finger_reach_ppo_cfg(),
    )
