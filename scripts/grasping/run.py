"""Run TorchScript hand policy on the *exact* training grasp plant (MuJoCo GUI).

CONTROL mode calls the same ``GraspAction.process_actions`` / ``apply_actions`` as
``train_grasp`` / ``play_grasp`` (REACH/LIFT + impedance by default).

```bash
uv run python scripts/grasping/run.py --policy logs/grasping/apple_deploy/policy.pt
```

Keys:
  r         reset apple
  d         DEFAULT (home)
  v         CONTROL (REACH / LIFT)
  u/o       lift target ±x
  j/l       lift target ±y
  i/k       lift target ±z
  Enter     full env reset
  Space     pause
  Ctrl+drag perturb bodies
"""

from __future__ import annotations

import argparse
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.viewer import NativeMujocoViewer, VerbosityLevel
from mjlab.viewer.base import ViewerAction
from mjlab.viewer.native.keys import (
    KEY_D,
    KEY_ENTER,
    KEY_EQUAL,
    KEY_I,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_MINUS,
    KEY_O,
    KEY_R,
    KEY_RIGHT,
    KEY_SPACE,
    KEY_U,
    KEY_V,
)
from mjlab.viewer.native.viewer import make_empty_figure
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from rophi_rl_tutorial.controllers.grasping.deploy import (
    ControlMode,
    GraspController,
    GraspPhase,
    HandPolicy,
)
from rophi_rl_tutorial.tasks.grasping.mjlab_task import fingertip_contact_binary, register

sys.path.insert(0, str(Path(__file__).parent))
from distill_grasp import SafeVecEnvWrapper  # noqa: E402

_SCORE_HISTORY = 300
_LIFT_STEP = 0.02


def build_student_env(
    device: str,
    *,
    task: str,
    hand_mode: str,
    arm_residual: bool,
    student_obs: str,
    hand_pca_project: float | None,
    physics_dt: float | None,
    decimation: int | None,
    interactive: bool,
    arm_velocity: bool = False,
    hand_gate: bool = False,
    engage_radius: float | None = None,
    release_radius: float | None = None,
) -> RslRlVecEnvWrapper:
    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = 1
    env_cfg.actions["grasp"].hand_mode = hand_mode
    env_cfg.actions["grasp"].arm_residual = arm_residual
    # Exact training default is impedance; --arm-velocity matches train_grasp --arm-velocity.
    env_cfg.actions["grasp"].arm_control = "velocity" if arm_velocity else "impedance"
    env_cfg.actions["grasp"].hand_gate = hand_gate
    if arm_residual:
        env_cfg.actions["grasp"].phase.reach_randomize = False
    if engage_radius is not None:
        env_cfg.actions["grasp"].phase.engage_radius = engage_radius
    if release_radius is not None:
        env_cfg.actions["grasp"].phase.release_radius = release_radius
    if hand_pca_project is not None:
        env_cfg.actions["grasp"].hand_pca_project = hand_pca_project
    if physics_dt is not None:
        env_cfg.sim.mujoco.timestep = physics_dt
    if decimation is not None:
        env_cfg.decimation = decimation

    actor_terms = dict(env_cfg.observations["actor"].terms)
    student_terms = {
        "hand_joint_pos": actor_terms["hand_joint_pos"],
        "hand_tracking": actor_terms["hand_tracking"],
    }
    if student_obs == "fingertip":
        student_terms = {
            "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
            **student_terms,
        }
    env_cfg.observations = {
        "student": ObservationGroupCfg(terms=student_terms, concatenate_terms=True),
    }
    if interactive:
        env_cfg.terminations.pop("success", None)
        env_cfg.episode_length_s = 1.0e6
    return SafeVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=device))


class DeployViewer(NativeMujocoViewer):
    def setup(self) -> None:
        super().setup()
        self._score_hist: deque[float] = deque(maxlen=_SCORE_HISTORY)
        self._score_fig = make_empty_figure(
            "grasp score",
            grid_size=(3, 4),
            yrange=(0.0, 1.0),
            history=_SCORE_HISTORY,
            alpha=0.55,
        )
        self._score_fig.range[1][0] = 0.0
        self._score_fig.range[1][1] = 1.0
        self._score_fig.linepnt[1] = 2
        self._score_fig.linedata[1][0] = 0.0
        self._score_fig.linedata[1][1] = 0.55
        self._score_fig.linedata[1][2] = -float(_SCORE_HISTORY)
        self._score_fig.linedata[1][3] = 0.55
        controller = self.policy
        if isinstance(controller, GraspController) and controller.mode is ControlMode.DEFAULT:
            controller.go_home()

    def close(self) -> None:
        controller = self.policy
        if isinstance(controller, GraspController):
            controller.stop_threads()
        super().close()

    def _safe_key_callback(self, key: int) -> None:
        if key == KEY_ENTER:
            self.request_reset()
        elif key == KEY_SPACE:
            self.request_toggle_pause()
        elif key == KEY_MINUS:
            self.request_speed_down()
        elif key == KEY_EQUAL:
            self.request_speed_up()
        elif key == KEY_RIGHT:
            self.request_single_step()
        elif key == KEY_R:
            self.request_action("CUSTOM", "reset_apple")
        elif key == KEY_D:
            self.request_action("CUSTOM", "mode_default")
        elif key == KEY_V:
            self.request_action("CUSTOM", "mode_control")
        elif key == KEY_U:
            self.request_action("CUSTOM", ("lift", +_LIFT_STEP, 0.0, 0.0))
        elif key == KEY_O:
            self.request_action("CUSTOM", ("lift", -_LIFT_STEP, 0.0, 0.0))
        elif key == KEY_J:
            self.request_action("CUSTOM", ("lift", 0.0, +_LIFT_STEP, 0.0))
        elif key == KEY_L:
            self.request_action("CUSTOM", ("lift", 0.0, -_LIFT_STEP, 0.0))
        elif key == KEY_I:
            self.request_action("CUSTOM", ("lift", 0.0, 0.0, +_LIFT_STEP))
        elif key == KEY_K:
            self.request_action("CUSTOM", ("lift", 0.0, 0.0, -_LIFT_STEP))

    def _handle_custom_action(self, action: ViewerAction, payload: object | None) -> bool:
        if action is not ViewerAction.CUSTOM:
            return super()._handle_custom_action(action, payload)
        controller = self.policy
        if not isinstance(controller, GraspController):
            return False
        if payload == "reset_apple":
            controller.reset_object()
            self.log("[run] apple reset", VerbosityLevel.INFO)
            return True
        if payload == "mode_default":
            controller.set_mode(ControlMode.DEFAULT)
            self._score_hist.clear()
            self.log("[run] mode = DEFAULT", VerbosityLevel.INFO)
            return True
        if payload == "mode_control":
            controller.set_mode(ControlMode.CONTROL)
            self._score_hist.clear()
            self.log("[run] mode = CONTROL", VerbosityLevel.INFO)
            return True
        if isinstance(payload, tuple) and payload and payload[0] == "lift":
            _, dx, dy, dz = payload
            controller.nudge_lift_target(np.array([dx, dy, dz], dtype=np.float64))
            t = controller.lift_target
            self.log(
                f"[run] lift target = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})",
                VerbosityLevel.INFO,
            )
            return True
        return False

    def reset_environment(self) -> None:
        super().reset_environment()
        self._score_hist.clear()
        controller = self.policy
        if isinstance(controller, GraspController):
            controller.reset()
            if controller.mode is ControlMode.DEFAULT:
                controller.go_home()

    def _execute_step(self) -> bool:
        try:
            with torch.no_grad():
                obs = self.env.get_observations()
                actions = self.policy(obs)
                _obs, _rew, dones, _extras = self.env.step(actions)
                if bool(dones.any()):
                    controller = self.policy
                    if isinstance(controller, GraspController):
                        controller.reset()
                        if controller.mode is ControlMode.DEFAULT:
                            controller.go_home()
                self._step_count += 1
                self._stats_steps += 1
                return True
        except Exception:
            import traceback

            self._last_error = traceback.format_exc()
            self.log(f"[ERROR] Exception during step:\n{self._last_error}", VerbosityLevel.SILENT)
            self.pause()
            return False

    def _set_status_overlay(self, viewer: mujoco.viewer.Handle) -> None:
        status = self.get_status()
        capped = " [CAPPED]" if status.capped else ""
        controller = self.policy
        mode = phase = lift = dist_txt = engaged = "?"
        score = 0.0
        if isinstance(controller, GraspController):
            mode = controller.mode.value.upper()
            phase = controller.phase.name
            cmd = controller.last_command
            if cmd is not None:
                score = float(cmd.grasp_score[0])
            t = controller.lift_target
            lift = f"({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})"
            term = controller._term
            palm = term._entity.data.site_pos_w[0, term._site_id].detach().cpu().numpy()
            obj = self.env.unwrapped.scene["object"].data.root_link_pos_w[0].detach().cpu().numpy()
            d = float(np.linalg.norm(obj - palm))
            dist_txt = f"{d:.3f}/{controller.engage_radius:.2f}"
            engaged = "Y" if controller.engaged else "N"
        text_1 = "Mode\nPhase\nScore\nDist/Eng\nEngaged\nLift\nArm\nStep\nStatus"
        arm = controller.arm_control if isinstance(controller, GraspController) else "?"
        text_2 = (
            f"{mode}\n{phase}\n{score:.3f}\n{dist_txt}\n{engaged}\n{lift}\n{arm}\n"
            f"{status.step_count}\n{'PAUSED' if status.paused else 'RUNNING'}{capped}"
        )
        viewer.set_texts(
            (
                mujoco.mjtFontScale.mjFONTSCALE_150.value,
                mujoco.mjtGridPos.mjGRID_TOPLEFT.value,
                text_1,
                text_2,
            )
        )

    def _update_debug_visualizers(self, viewer: mujoco.viewer.Handle) -> None:
        viewer.user_scn.ngeom = 0
        if not self._show_debug_vis:
            return
        assert self.mjm is not None
        viz = MujocoNativeDebugVisualizer(
            viewer.user_scn, self.mjm, self.env_idx, self._show_all_envs
        )
        if hasattr(self.env.unwrapped, "update_visualizers"):
            self.env.unwrapped.update_visualizers(viz)

        controller = self.policy
        if not isinstance(controller, GraspController):
            return
        term = controller._term
        obj = self.env.unwrapped.scene["object"].data.root_link_pos_w[0].detach().cpu().numpy()
        palm = term._entity.data.site_pos_w[0, term._site_id].detach().cpu().numpy()
        tgt = term.impedance.target_pos[0].detach().cpu().numpy()
        engage_r = float(controller.engage_radius)
        release_r = float(controller.release_radius)
        phase = controller.phase

        viz.add_sphere(obj, release_r, (0.2, 0.6, 1.0, 0.18))
        viz.add_sphere(obj, engage_r, (0.2, 1.0, 0.3, 0.28))
        dist = float(np.linalg.norm(obj - palm))
        if dist <= engage_r:
            col = (0.2, 1.0, 0.3, 0.95)
        elif dist <= release_r:
            col = (1.0, 0.85, 0.2, 0.95)
        else:
            col = (1.0, 0.3, 0.2, 0.95)
        viz.add_arrow(palm, obj, col, width=0.008)
        viz.add_sphere(tgt, 0.012, (1.0, 0.4, 1.0, 0.85))
        viz.add_arrow(palm, tgt, (1.0, 0.4, 1.0, 0.7), width=0.006)
        marker = obj + np.array([0.0, 0.0, release_r + 0.04])
        phase_col = {
            GraspPhase.REACH: (1.0, 0.4, 0.2, 0.9),
            GraspPhase.LIFT: (0.3, 0.5, 1.0, 0.9),
        }.get(phase, (1.0, 1.0, 1.0, 0.9))
        viz.add_sphere(marker, 0.015, phase_col)

    def _update_reward_figures(self, viewer: mujoco.viewer.Handle) -> None:
        cmd = getattr(self.policy, "last_command", None)
        if cmd is not None and not self._is_paused:
            self._score_hist.append(float(cmd.grasp_score[0]))
        fig = self._score_fig
        n = min(len(self._score_hist), _SCORE_HISTORY)
        fig.linepnt[0] = n
        for i in range(n):
            fig.linedata[0][2 * i] = float(-i)
            fig.linedata[0][2 * i + 1] = float(self._score_hist[-1 - i])
        phase = ""
        if isinstance(self.policy, GraspController):
            phase = f"  [{self.policy.phase.name}]"
        fig.title = (
            f"grasp score  ({float(self._score_hist[-1]):.2f}){phase}"
            if n
            else f"grasp score{phase}"
        )
        vp = viewer.viewport
        width = max(int(vp.width * 0.32), 220)
        height = max(int(vp.height * 0.22), 120)
        rect = mujoco.MjrRect(
            left=vp.left + vp.width - width - 8,
            bottom=vp.bottom + 8,
            width=width,
            height=height,
        )
        viewer.set_figures([(rect, fig)])
        self._figures_dirty = True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default="logs/grasping/apple_deploy/policy.pt")
    p.add_argument("--task", default="RoPhi-Grasp-Apple")
    p.add_argument("--hand-mode", default="pca", choices=("pca", "joints"))
    p.add_argument("--arm-residual", action="store_true")
    p.add_argument(
        "--arm-velocity",
        action="store_true",
        help="Use training --arm-velocity plant instead of impedance.",
    )
    p.add_argument(
        "--hand-gate",
        action="store_true",
        help="Training --hand-gate (open until engage_radius).",
    )
    p.add_argument("--student-obs", default="fingertip", choices=("fingertip", "proprio"))
    p.add_argument("--hand-pca-project", type=float, default=None)
    p.add_argument("--physics-dt", type=float, default=None)
    p.add_argument("--decimation", type=int, default=None)
    p.add_argument("--engage-radius", type=float, default=None)
    p.add_argument("--release-radius", type=float, default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--train-terminations", action="store_true")
    p.add_argument("--start-control", action="store_true")
    args = p.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.is_file():
        raise SystemExit(f"policy not found: {policy_path}")

    register()
    env = build_student_env(
        args.device,
        task=args.task,
        hand_mode=args.hand_mode,
        arm_residual=args.arm_residual,
        student_obs=args.student_obs,
        hand_pca_project=args.hand_pca_project,
        physics_dt=args.physics_dt,
        decimation=args.decimation,
        interactive=not args.train_terminations,
        arm_velocity=args.arm_velocity,
        hand_gate=args.hand_gate,
        engage_radius=args.engage_radius,
        release_radius=args.release_radius,
    )

    term = env.unwrapped.action_manager.get_term("grasp")
    hand_policy = HandPolicy(
        policy_path,
        hand_mode=args.hand_mode,
        arm_residual=args.arm_residual,
        squash=term.cfg.squash,
        residual_scale=term.cfg.arm_residual_scale,
        residual_rot_scale=term.cfg.arm_residual_rot_scale,
        hand_pca_project=term.cfg.hand_pca_project
        if args.hand_pca_project is None
        else args.hand_pca_project,
        q_lo=term._q_lo,
        q_hi=term._q_hi,
        device=args.device,
        pca=term._pca,
    )
    controller = GraspController(hand_policy, env, obs_key="student")
    controller.install()
    if args.start_control:
        controller.set_mode(ControlMode.CONTROL)
    else:
        controller.go_home()

    obs = env.get_observations()
    obs_dim = int(obs["student"].shape[1])
    with torch.no_grad():
        _ = controller(obs)
        env.step(torch.zeros(1, hand_policy.action_dim, device=args.device))
    assert controller.last_command is not None

    ctrl_hz = 1.0 / (float(env.unwrapped.physics_dt) * int(env.unwrapped.cfg.decimation))
    print(
        f"[run] {policy_path}: obs {obs_dim} → {hand_policy.action_dim}  "
        f"plant={term.cfg.arm_control} (training GraspAction)  "
        f"hand_gate={term.cfg.hand_gate}  control≈{ctrl_hz:.0f} Hz"
    )
    print(
        f"[run] mode={controller.mode.value} phase={controller.phase.name}  "
        f"engage/release={controller.engage_radius:.2f}/{controller.release_radius:.2f}"
    )
    print("[run] keys: r apple | d home | v control | u/o j/l i/k lift target")

    viewer = DeployViewer(env, controller, enable_perturbations=True, verbosity=VerbosityLevel.INFO)
    try:
        viewer.run()
    finally:
        controller.stop_threads()


if __name__ == "__main__":
    main()
