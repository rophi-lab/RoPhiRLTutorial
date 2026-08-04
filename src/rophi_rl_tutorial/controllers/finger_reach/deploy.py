"""Finger-reach deploy: TorchScript tip-velocity policy + tip-position velocity field.

The policy was trained to track a commanded **index-tip linear velocity**. At deploy
time the user picks a **tip target position**; a converging linear velocity field
turns ``target_pos - tip_pos`` into ``v_cmd``, which is fed to the policy as the
command observation (same 17-D layout as training).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rophi_rl_tutorial.assets.finger_reach import (
    ARM_JOINT_NAMES,
    HAND_JOINT_NAMES,
    HAND_Q_OPEN,
    INDEX_TIP_SITE,
)
from rophi_rl_tutorial.assets.finger_reach.robot import ARM_Q_HOME
from rophi_rl_tutorial.tasks.finger_reach.mjlab_task import (
    CMD_VEL_MAX,
    _ensure_target_vel_buffer,
    _index_tip_site_id,
)

__all__ = [
    "ControlMode",
    "FingerReachPolicy",
    "FingerReachController",
    "DeploySnapshot",
    "velocity_field_toward",
]


class ControlMode(Enum):
    DEFAULT = "default"  # hold / zero tip-velocity command
    CONTROL = "control"  # VF toward user tip target → policy


@dataclass
class DeploySnapshot:
    """Latest tip / command state for overlays."""

    tip_pos: np.ndarray
    tip_vel: np.ndarray
    target_pos: np.ndarray
    cmd_vel: np.ndarray
    mode: ControlMode


def velocity_field_toward(
    tip_pos: torch.Tensor,
    target_pos: torch.Tensor,
    *,
    v_max: float = CMD_VEL_MAX,
    gain: float = 2.0,
    arrive_eps: float = 0.01,
) -> torch.Tensor:
    """Converging linear velocity field: ``v = saturate(gain * err)`` toward the target.

    ``tip_pos`` / ``target_pos`` shape ``(N, 3)`` or ``(3,)``. Returns same batching as
    inputs, clipped so each component stays in ``[-v_max, v_max]`` (training command box).
    """
    if tip_pos.ndim == 1:
        tip_pos = tip_pos.unsqueeze(0)
        target_pos = target_pos.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    err = target_pos - tip_pos
    dist = torch.linalg.norm(err, dim=-1, keepdim=True).clamp(min=1e-8)
    speed = torch.clamp(gain * dist, max=v_max)
    v = torch.where(dist < arrive_eps, torch.zeros_like(err), speed * err / dist)
    v = v.clamp(-v_max, v_max)
    return v.squeeze(0) if squeeze else v


##
# TorchScript policy.
##


class FingerReachPolicy:
    """Exported PPO actor: flat obs ``[1, 17]`` → arm joint-velocity action ``[1, 7]``."""

    OBS_DIM = 17
    ACTION_DIM = 7

    def __init__(self, path: str | Path, device: str | torch.device = "cuda:0") -> None:
        self.device = torch.device(device)
        self.module = torch.jit.load(str(path), map_location="cpu")
        self.module.eval()

    def reset(self) -> None:
        reset_fn = getattr(self.module, "reset", None)
        if reset_fn is not None:
            reset_fn()

    def act(self, obs_flat: torch.Tensor) -> torch.Tensor:
        if obs_flat.ndim == 1:
            obs_flat = obs_flat.unsqueeze(0)
        if obs_flat.shape[-1] != self.OBS_DIM:
            raise ValueError(f"obs dim {obs_flat.shape[-1]} != {self.OBS_DIM}")
        with torch.inference_mode():
            out = self.module(obs_flat.detach().cpu().float())
        return torch.nan_to_num(
            out.detach().to(device=self.device, dtype=torch.float32), nan=0.0
        )


##
# Controller: tip target (keys) → VF → policy → env actions.
##


class FingerReachController:
    """Viewer policy: velocity field toward a keyframed tip target + TorchScript actor."""

    def __init__(
        self,
        policy: FingerReachPolicy,
        env: Any,
        *,
        v_max: float = CMD_VEL_MAX,
        vf_gain: float = 2.0,
        arrive_eps: float = 0.01,
        nudge_step: float = 0.02,
    ) -> None:
        self.policy = policy
        self.env = env
        self.v_max = float(v_max)
        self.vf_gain = float(vf_gain)
        self.arrive_eps = float(arrive_eps)
        self.nudge_step = float(nudge_step)
        self.mode = ControlMode.DEFAULT

        unwrapped = env.unwrapped
        device = unwrapped.device
        self._device = device
        self._tip_site_id = _index_tip_site_id(unwrapped)
        self.target_pos = torch.zeros(3, device=device, dtype=torch.float32)
        self.last: DeploySnapshot | None = None

        robot = unwrapped.scene["robot"]
        arm_ids, _ = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        hand_ids, _ = robot.find_joints(HAND_JOINT_NAMES, preserve_order=True)
        self._arm_ids = torch.tensor(arm_ids, device=device, dtype=torch.long)
        self._hand_ids = torch.tensor(hand_ids, device=device, dtype=torch.long)
        self._all_ids = torch.cat([self._arm_ids, self._hand_ids])
        self._home_q = torch.tensor(
            [ARM_Q_HOME[n] for n in ARM_JOINT_NAMES]
            + [HAND_Q_OPEN[n] for n in HAND_JOINT_NAMES],
            dtype=torch.float32,
            device=device,
        )

    @property
    def tip_pos(self) -> torch.Tensor:
        robot = self.env.unwrapped.scene["robot"]
        return robot.data.site_pos_w[0, self._tip_site_id]

    @property
    def tip_vel(self) -> torch.Tensor:
        robot = self.env.unwrapped.scene["robot"]
        return robot.data.site_lin_vel_w[0, self._tip_site_id]

    def reset(self) -> None:
        self.policy.reset()
        self._snap_target_to_tip()
        _ensure_target_vel_buffer(self.env.unwrapped).zero_()

    def set_mode(self, mode: ControlMode) -> None:
        if mode is ControlMode.DEFAULT:
            self.go_home()
            return
        self.mode = ControlMode.CONTROL
        self.policy.reset()
        self._snap_target_to_tip()

    def go_home(self) -> None:
        self.mode = ControlMode.DEFAULT
        self.policy.reset()
        unwrapped = self.env.unwrapped
        robot = unwrapped.scene["robot"]
        zeros = torch.zeros_like(self._home_q)
        robot.write_joint_state_to_sim(self._home_q, zeros, joint_ids=self._all_ids)
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.forward()
        self._snap_target_to_tip()
        _ensure_target_vel_buffer(unwrapped).zero_()

    def _snap_target_to_tip(self) -> None:
        self.target_pos = self.tip_pos.detach().clone()

    def nudge_target(self, delta_xyz: np.ndarray | tuple[float, float, float]) -> None:
        d = torch.as_tensor(delta_xyz, device=self._device, dtype=torch.float32)
        self.target_pos = self.target_pos + d

    def _cmd_vel(self) -> torch.Tensor:
        if self.mode is ControlMode.DEFAULT:
            return torch.zeros(3, device=self._device)
        return velocity_field_toward(
            self.tip_pos,
            self.target_pos,
            v_max=self.v_max,
            gain=self.vf_gain,
            arrive_eps=self.arrive_eps,
        )

    def __call__(self, obs: Any) -> torch.Tensor:
        """Update VF command, patch actor obs, run TorchScript → arm actions."""
        unwrapped = self.env.unwrapped
        v_cmd = self._cmd_vel()
        buf = _ensure_target_vel_buffer(unwrapped)
        buf[0] = v_cmd

        # Training obs = [arm_q(7), arm_qd(7), target_vel(3)].
        if hasattr(obs, "get"):
            actor = obs["actor"]
        else:
            actor = obs
        actor = actor.clone()
        actor[0, -3:] = v_cmd

        actions = self.policy.act(actor[0])
        if self.mode is ControlMode.DEFAULT:
            actions = torch.zeros_like(actions)

        tip = self.tip_pos.detach().cpu().numpy()
        self.last = DeploySnapshot(
            tip_pos=tip.copy(),
            tip_vel=self.tip_vel.detach().cpu().numpy().copy(),
            target_pos=self.target_pos.detach().cpu().numpy().copy(),
            cmd_vel=v_cmd.detach().cpu().numpy().copy(),
            mode=self.mode,
        )
        return actions
