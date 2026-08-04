"""Deploy: TorchScript hand policy on the *exact* training ``GraspAction`` plant.

CONTROL mode feeds ``HandPolicy`` raw actions into ``GraspAction.process_actions`` /
``apply_actions`` — same REACH/LIFT phase, impedance (or ``--arm-velocity``), hand VF,
and residual path as ``train_grasp`` / ``play_grasp``. No parallel FSM or custom arm law.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rophi_rl_tutorial.assets.grasping.hand_pca import HandPCA, load_hand_pca
from rophi_rl_tutorial.assets.grasping.rizon4_hand import (
    ARM_JOINT_NAMES,
    ARM_Q_HOME,
    HAND_JOINT_NAMES,
    HAND_Q_OPEN,
)
from rophi_rl_tutorial.controllers.grasping.batched import BatchedGraspPhase

__all__ = [
    "ControlMode",
    "GraspPhase",
    "HandPolicy",
    "HandPolicyOutput",
    "JointCommand",
    "GraspController",
]


class ControlMode(Enum):
    DEFAULT = "default"
    CONTROL = "control"


class GraspPhase(IntEnum):
    """Mirrors ``BatchedGraspPhase.REACH`` / ``LIFT``."""

    REACH = 0
    LIFT = 1


@dataclass
class HandPolicyOutput:
    grasp_score: torch.Tensor
    hand_q_target: torch.Tensor
    residual_pos: torch.Tensor | None = None
    residual_rot: torch.Tensor | None = None
    raw_action: torch.Tensor | None = None


@dataclass
class JointCommand:
    """Snapshot after ``apply_actions`` (overlays)."""

    arm_effort: torch.Tensor
    hand_position: torch.Tensor
    grasp_score: torch.Tensor
    hand_q_target: torch.Tensor
    phase: GraspPhase = GraspPhase.REACH
    engaged: bool = False
    target_pos: torch.Tensor | None = None
    target_quat: torch.Tensor | None = None


##
# Hand policy (TorchScript).
##


class HandPolicy:
    """TorchScript student: student obs → raw action (same layout as training)."""

    def __init__(
        self,
        path: str | Path,
        *,
        hand_mode: str = "pca",
        arm_residual: bool = False,
        squash: str = "sigmoid",
        residual_scale: float = 0.03,
        residual_rot_scale: float = 0.08726646259971647,  # radians(5°)
        hand_pca_project: float = 0.0,
        q_lo: torch.Tensor,
        q_hi: torch.Tensor,
        device: str | torch.device = "cuda:0",
        pca: HandPCA | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.hand_mode = hand_mode
        self.arm_residual = arm_residual
        self.squash = squash
        self.residual_scale = residual_scale
        self.residual_rot_scale = residual_rot_scale
        self.hand_pca_project = float(hand_pca_project)
        self.q_lo = q_lo.to(self.device)
        self.q_hi = q_hi.to(self.device)
        if self.q_lo.ndim == 1:
            self.q_lo = self.q_lo.unsqueeze(0)
            self.q_hi = self.q_hi.unsqueeze(0)

        self._pca: HandPCA | None
        self._proj_pca: HandPCA | None = None
        if hand_mode == "pca":
            self._pca = pca if pca is not None else load_hand_pca(device=self.device)
            self._num_hand = self._pca.z_dim
        else:
            self._pca = None
            self._num_hand = len(HAND_JOINT_NAMES)
            if self.hand_pca_project > 0.0:
                self._proj_pca = load_hand_pca(device=self.device)

        self._n_pre = 7 if arm_residual else 1
        self.action_dim = self._n_pre + self._num_hand
        self.module = torch.jit.load(str(path), map_location="cpu")
        self.module.eval()

    def reset(self) -> None:
        reset_fn = getattr(self.module, "reset", None)
        if reset_fn is not None:
            reset_fn()

    def _squash01(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(x) if self.squash == "sigmoid" else x.clamp(0.0, 1.0)

    def act(self, student_obs: torch.Tensor) -> HandPolicyOutput:
        if student_obs.ndim == 1:
            student_obs = student_obs.unsqueeze(0)
        with torch.inference_mode():
            raw_inf = self.module(student_obs.detach().cpu())
        raw = torch.nan_to_num(
            raw_inf.detach().clone().to(device=self.device, dtype=torch.float32),
            nan=0.0,
        )
        if raw.shape[-1] != self.action_dim:
            raise ValueError(
                f"policy action dim {raw.shape[-1]} != {self.action_dim} "
                f"(hand_mode={self.hand_mode}, arm_residual={self.arm_residual})"
            )

        score = self._squash01(raw[:, 0])
        if self.arm_residual:
            residual_pos = raw[:, 1:4].clamp(-1.0, 1.0) * self.residual_scale
            residual_rot = raw[:, 4:7].clamp(-1.0, 1.0) * self.residual_rot_scale
        else:
            residual_pos = residual_rot = None

        hand_alpha = self._squash01(raw[:, self._n_pre :])
        if self._pca is not None:
            z = self._pca.latent_from_alpha(hand_alpha)
            hand_q = self._pca.decode(z).clamp(self.q_lo, self.q_hi)
        else:
            hand_q = self.q_lo + hand_alpha * (self.q_hi - self.q_lo)
            if self._proj_pca is not None:
                recon = self._proj_pca.reconstruct(hand_q).clamp(self.q_lo, self.q_hi)
                a = self.hand_pca_project
                hand_q = (1.0 - a) * hand_q + a * recon

        return HandPolicyOutput(
            grasp_score=score,
            hand_q_target=hand_q.detach().clone(),
            residual_pos=residual_pos,
            residual_rot=residual_rot,
            raw_action=raw,
        )


##
# Thin wrapper around training GraspAction.
##


class GraspController:
    """Inject ``HandPolicy`` into training ``GraspAction`` (identical plant)."""

    def __init__(self, policy: HandPolicy, env: Any, obs_key: str = "student") -> None:
        self.policy = policy
        self.env = env
        self.obs_key = obs_key
        self.mode = ControlMode.DEFAULT

        self._term = env.unwrapped.action_manager.get_term("grasp")
        self._orig_process = self._term.process_actions
        self._orig_apply = self._term.apply_actions
        self.last_command: JointCommand | None = None
        self._latest_obs: torch.Tensor | None = None
        self._last_score = 0.0

        device = env.unwrapped.device
        self._home_q = torch.tensor(
            [ARM_Q_HOME[n] for n in ARM_JOINT_NAMES]
            + [HAND_Q_OPEN[n] for n in HAND_JOINT_NAMES],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(0)
        ids, _ = self._term._entity.find_joints(
            list(ARM_JOINT_NAMES) + list(HAND_JOINT_NAMES), preserve_order=True
        )
        self._home_joint_ids = ids

    @property
    def phase(self) -> GraspPhase:
        p = int(self._term.phase.phase[0].item())
        return GraspPhase.LIFT if p == BatchedGraspPhase.LIFT else GraspPhase.REACH

    @property
    def engaged(self) -> bool:
        return bool(self._term.phase.engaged[0].item())

    @property
    def engage_radius(self) -> float:
        return float(self._term.phase.cfg.engage_radius)

    @property
    def release_radius(self) -> float:
        return float(self._term.phase.cfg.release_radius)

    @property
    def close_radius(self) -> float:
        return self.engage_radius

    @property
    def hand_gate(self) -> bool:
        return bool(self._term.cfg.hand_gate)

    @property
    def arm_control(self) -> str:
        return str(self._term.cfg.arm_control)

    @property
    def lift_target(self) -> np.ndarray:
        pos = self._term.phase.lift_override_pos
        if pos is None:
            pos = self._term.phase.lift_pos
        return pos[0].detach().cpu().numpy().astype(np.float64)

    def install(self) -> None:
        self._term.process_actions = self._process_actions  # type: ignore[method-assign]
        self._term.apply_actions = self._apply_actions  # type: ignore[method-assign]

    def start_threads(self) -> None:
        """No-op: policy runs inside ``process_actions`` at the env control rate."""

    def stop_threads(self) -> None:
        """No-op (API compat)."""

    def reset(self) -> None:
        self.policy.reset()
        self._last_score = 0.0
        self.last_command = None
        term = self._term
        env_ids = torch.arange(term._num, device=term.device)
        term.reset(env_ids)
        term.phase.lift_override_pos = None

    def set_mode(self, mode: ControlMode) -> None:
        if mode is ControlMode.DEFAULT:
            self.go_home()
            return
        self.mode = ControlMode.CONTROL
        term = self._term
        term._q_cmd = (
            term._entity.data.joint_pos[:, term._hand_joint_ids].detach().clone()
        )
        term.hand_target = term._q_cmd.clone()
        term.phase.lift_override_pos = None
        env_ids = torch.arange(term._num, device=term.device)
        term.phase.reset(
            env_ids,
            term._home_site_pos.unsqueeze(0).expand(term._num, 3).clone(),
            term._home_site_quat.unsqueeze(0).expand(term._num, 4).clone(),
        )
        self.policy.reset()
        self._last_score = 0.0

    def go_home(self) -> None:
        self.mode = ControlMode.DEFAULT
        self.policy.reset()
        unwrapped = self.env.unwrapped
        robot = unwrapped.scene["robot"]
        zeros = torch.zeros_like(self._home_q)
        robot.write_joint_state_to_sim(
            self._home_q, zeros, joint_ids=self._home_joint_ids
        )
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.forward()

        term = self._term
        term.hand_target = term._q_open.detach().clone().repeat(term._num, 1)
        term._q_cmd = term.hand_target.clone()
        term._raw = torch.zeros_like(term._raw)
        env_ids = torch.arange(term._num, device=term.device)
        term.reset(env_ids)
        term.hand_target.copy_(term._q_open)
        term._q_cmd.copy_(term._q_open)
        term.phase.lift_override_pos = None
        n = term._num
        term.impedance.set_target(
            term._home_site_pos.unsqueeze(0).expand(n, 3).clone(),
            term._home_site_quat.unsqueeze(0).expand(n, 4).clone(),
        )
        self._last_score = 0.0

    def reset_object(self, pose_range: dict | None = None) -> None:
        from rophi_rl_tutorial.tasks.grasping.mjlab_task import _reset_object

        if pose_range is None:
            pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-3.14, 3.14)}
        unwrapped = self.env.unwrapped
        env_ids = torch.tensor([0], device=unwrapped.device, dtype=torch.long)
        _reset_object(unwrapped, env_ids, pose_range)
        unwrapped.scene.write_data_to_sim()
        unwrapped.sim.forward()
        term = self._term
        term.phase.reset(
            env_ids,
            term._home_site_pos.unsqueeze(0).expand(term._num, 3).clone(),
            term._home_site_quat.unsqueeze(0).expand(term._num, 4).clone(),
        )
        term.phase.lift_override_pos = None
        self._last_score = 0.0
        self.policy.reset()

    def nudge_lift_target(self, delta: np.ndarray) -> None:
        term = self._term
        d = torch.as_tensor(delta, dtype=torch.float32, device=term.device).reshape(
            1, 3
        )
        if term.phase.lift_override_pos is None:
            term.phase.lift_override_pos = term.phase.lift_pos.detach().clone()
        term.phase.lift_override_pos = term.phase.lift_override_pos + d
        term.phase.lift_override_pos[:, 2].clamp_(min=0.05)

    def update_state(self, obs: Any) -> None:
        self._latest_obs = obs[self.obs_key]

    def __call__(self, obs: Any) -> torch.Tensor:
        self.update_state(obs)
        return torch.zeros(1, self.policy.action_dim, device=self.policy.device)

    def _process_actions(self, _actions: torch.Tensor) -> None:
        if self.mode is ControlMode.DEFAULT or self._latest_obs is None:
            return
        out = self.policy.act(self._latest_obs)
        assert out.raw_action is not None
        self._last_score = float(out.grasp_score[0])
        # Identical to training: squash, phase.update, impedance.set_target, hand_gate.
        self._orig_process(out.raw_action)

    def _apply_actions(self) -> None:
        term = self._term
        # Identical to training: impedance.compute or _velocity_torque + hand VF integrate.
        self._orig_apply()
        self.last_command = JointCommand(
            arm_effort=torch.zeros(1, 7, device=term.device),
            hand_position=term._q_cmd.detach().clone(),
            grasp_score=torch.tensor([self._last_score], device=term.device),
            hand_q_target=term.hand_target.detach().clone(),
            phase=self.phase,
            engaged=self.engaged,
            target_pos=term.impedance.target_pos.detach().clone(),
            target_quat=term.impedance.target_quat.detach().clone(),
        )
