"""Standalone MuJoCo GUI: TorchScript finger-reach policy + tip-position velocity field.

```bash
uv run python scripts/finger_reach/export_policy.py
uv run python scripts/finger_reach/run.py --policy logs/finger_reach/reach/policy.pt
```

Keys:
  v         CONTROL (velocity field toward tip target → policy)
  d         DEFAULT (home / zero tip-velocity command)
  u/o       tip target ±x
  j/l       tip target ±y
  i/k       tip target ±z
  r         snap tip target to current tip
  Enter     env reset
  Space     pause
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
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
from mjlab.viewer.native.visualizer import MujocoNativeDebugVisualizer

from rophi_rl_tutorial.controllers.finger_reach import (
    ControlMode,
    FingerReachController,
    FingerReachPolicy,
)
from rophi_rl_tutorial.tasks.finger_reach.mjlab_task import register

_VEL_ARROW_SCALE = 0.20
_NUDGE = 0.02


def build_env(device: str, *, interactive: bool = True) -> RslRlVecEnvWrapper:
    env_cfg = load_env_cfg("RoPhi-Finger-Reach", play=True)
    env_cfg.scene.num_envs = 1
    if interactive:
        env_cfg.episode_length_s = 1.0e6
    return RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=device))


class FingerReachDeployViewer(NativeMujocoViewer):
    def setup(self) -> None:
        super().setup()
        controller = self.policy
        if isinstance(controller, FingerReachController) and controller.mode is ControlMode.DEFAULT:
            controller.go_home()

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
        elif key == KEY_D:
            self.request_action("CUSTOM", "mode_default")
        elif key == KEY_V:
            self.request_action("CUSTOM", "mode_control")
        elif key == KEY_R:
            self.request_action("CUSTOM", "snap_target")
        elif key == KEY_U:
            self.request_action("CUSTOM", ("nudge", +_NUDGE, 0.0, 0.0))
        elif key == KEY_O:
            self.request_action("CUSTOM", ("nudge", -_NUDGE, 0.0, 0.0))
        elif key == KEY_J:
            self.request_action("CUSTOM", ("nudge", 0.0, +_NUDGE, 0.0))
        elif key == KEY_L:
            self.request_action("CUSTOM", ("nudge", 0.0, -_NUDGE, 0.0))
        elif key == KEY_I:
            self.request_action("CUSTOM", ("nudge", 0.0, 0.0, +_NUDGE))
        elif key == KEY_K:
            self.request_action("CUSTOM", ("nudge", 0.0, 0.0, -_NUDGE))

    def _handle_custom_action(self, action: ViewerAction, payload: object | None) -> bool:
        if action is not ViewerAction.CUSTOM:
            return super()._handle_custom_action(action, payload)
        controller = self.policy
        if not isinstance(controller, FingerReachController):
            return False
        if payload == "mode_default":
            controller.set_mode(ControlMode.DEFAULT)
            self.log("[run] mode = DEFAULT", VerbosityLevel.INFO)
            return True
        if payload == "mode_control":
            controller.set_mode(ControlMode.CONTROL)
            self.log("[run] mode = CONTROL", VerbosityLevel.INFO)
            return True
        if payload == "snap_target":
            controller._snap_target_to_tip()
            t = controller.target_pos.detach().cpu().numpy()
            self.log(
                f"[run] tip target snapped = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})",
                VerbosityLevel.INFO,
            )
            return True
        if isinstance(payload, tuple) and payload and payload[0] == "nudge":
            _, dx, dy, dz = payload
            controller.nudge_target((dx, dy, dz))
            t = controller.target_pos.detach().cpu().numpy()
            self.log(
                f"[run] tip target = ({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f})",
                VerbosityLevel.INFO,
            )
            return True
        return False

    def reset_environment(self) -> None:
        super().reset_environment()
        controller = self.policy
        if isinstance(controller, FingerReachController):
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
                    if isinstance(controller, FingerReachController):
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
        mode = tgt = cmd = tip_spd = err = "?"
        if isinstance(controller, FingerReachController) and controller.last is not None:
            snap = controller.last
            mode = snap.mode.value.upper()
            t = snap.target_pos
            c = snap.cmd_vel
            tip_spd = f"{float(np.linalg.norm(snap.tip_vel)):.3f}"
            err = f"{float(np.linalg.norm(snap.tip_vel - snap.cmd_vel)):.3f}"
            tgt = f"({t[0]:.2f},{t[1]:.2f},{t[2]:.2f})"
            cmd = f"({c[0]:.2f},{c[1]:.2f},{c[2]:.2f})"
        text_1 = "Mode\nTip tgt\nCmd vel\nTip spd\nVel err\nStep\nStatus"
        text_2 = (
            f"{mode}\n{tgt}\n{cmd}\n{tip_spd}\n{err}\n{status.step_count}\n"
            f"{'PAUSED' if status.paused else 'RUNNING'}{capped}"
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
        if not isinstance(controller, FingerReachController) or controller.last is None:
            return
        snap = controller.last
        tip = snap.tip_pos
        tgt = snap.target_pos
        cmd = snap.cmd_vel
        vel = snap.tip_vel

        # Target position (green), tip (cyan).
        viz.add_sphere(tgt, 0.02, (0.2, 0.85, 0.3, 0.9))
        viz.add_sphere(tip, 0.012, (0.3, 0.7, 0.95, 0.9))
        viz.add_arrow(tip, tgt, (0.2, 0.85, 0.3, 0.55), width=0.004)

        # Commanded tip velocity (green arrow), measured tip velocity (orange).
        cmd_speed = float(np.linalg.norm(cmd))
        act_speed = float(np.linalg.norm(vel))
        if cmd_speed > 1e-3:
            viz.add_arrow(
                tip, tip + cmd * _VEL_ARROW_SCALE, (0.15, 0.9, 0.25, 0.95), width=0.008
            )
        if act_speed > 1e-3:
            viz.add_arrow(
                tip, tip + vel * _VEL_ARROW_SCALE, (1.0, 0.55, 0.15, 0.95), width=0.006
            )


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--policy",
        default="logs/finger_reach/reach/policy.pt",
        help="TorchScript from export_policy.py",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--vf-gain", type=float, default=2.0, help="Velocity-field gain [1/s].")
    p.add_argument(
        "--v-max",
        type=float,
        default=None,
        help="Tip velocity command clamp [m/s] (default: training _CMD_VEL_MAX).",
    )
    p.add_argument("--start-control", action="store_true", help="Start in CONTROL mode.")
    args = p.parse_args()

    policy_path = Path(args.policy)
    if not policy_path.is_file():
        raise SystemExit(
            f"policy not found: {policy_path}\n"
            "Export first:\n"
            "  uv run python scripts/finger_reach/export_policy.py"
        )

    register()
    env = build_env(args.device)
    kw = {"vf_gain": args.vf_gain}
    if args.v_max is not None:
        kw["v_max"] = args.v_max
    policy = FingerReachPolicy(policy_path, device=args.device)
    controller = FingerReachController(policy, env, **kw)
    if args.start_control:
        controller.set_mode(ControlMode.CONTROL)
    else:
        controller.go_home()

    obs = env.get_observations()
    with torch.no_grad():
        a = controller(obs)
    print(
        f"[run] policy {policy_path}; obs {tuple(obs['actor'].shape)} → action {tuple(a.shape)}"
    )
    print("[run] keys: v=CONTROL  d=DEFAULT  u/o j/l i/k=move tip target  r=snap  Enter=reset")

    FingerReachDeployViewer(env, policy=controller).run()


if __name__ == "__main__":
    main()
