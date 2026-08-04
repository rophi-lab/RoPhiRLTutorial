"""Play a trained finger-reach (tip-velocity tracking) policy in the viser viewer.

By default, pass ``--run-dir`` (default ``logs/finger_reach/reach``) and pick
``model_*.pt`` from the Checkpoints tab in the GUI. Optionally pass a single
``--checkpoint``.

Overlays:
* cyan sphere — current index fingertip
* green arrow — commanded tip velocity
* orange arrow — measured tip velocity (length ∝ speed for both)
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import viser
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.viewer import ViserPlayViewer
from mjlab.viewer.viser.viewer import CheckpointManager, format_time_ago

from rophi_rl_tutorial.assets.finger_reach import INDEX_TIP_SITE
from rophi_rl_tutorial.tasks.finger_reach.mjlab_task import (
    _ensure_target_vel_buffer,
    register,
)

# Arrow length [m] = speed [m/s] * this. At 1 m/s the shaft is 20 cm.
_VEL_ARROW_SCALE = 0.20
_VEL_ARROW_MIN = 1e-3  # hide below this speed [m/s]


class FingerReachViewer(ViserPlayViewer):
    """ViserPlayViewer with commanded vs actual index-tip velocity arrows."""

    def setup(self) -> None:
        super().setup()
        robot = self.env.unwrapped.scene["robot"]
        ids, _ = robot.find_sites([INDEX_TIP_SITE])
        self._tip_site_id = int(ids[0])

        self._tip_marker = self._server.scene.add_icosphere(
            "/finger_reach_tip",
            radius=0.015,
            color=(80, 180, 220),
            position=(0.0, 0.0, 0.0),
        )
        self._cmd_vel_arrow = self._server.scene.add_arrows(
            "/finger_reach_cmd_vel",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(60, 200, 90),
            shaft_radius=0.005,
            head_radius=0.012,
            head_length=0.025,
            visible=False,
        )
        self._act_vel_arrow = self._server.scene.add_arrows(
            "/finger_reach_tip_vel",
            points=np.zeros((1, 2, 3), dtype=np.float32),
            colors=(240, 140, 40),
            shaft_radius=0.005,
            head_radius=0.012,
            head_length=0.025,
            visible=False,
        )
        with self._server.gui.add_folder("Finger reach"):
            self._err_text = self._server.gui.add_text("vel error [m/s]", "—", disabled=True)
            self._cmd_speed_text = self._server.gui.add_text("cmd speed [m/s]", "—", disabled=True)
            self._act_speed_text = self._server.gui.add_text("tip speed [m/s]", "—", disabled=True)
            self._cmd_text = self._server.gui.add_text("cmd vel xyz", "—", disabled=True)

    def sync_env_to_viewer(self) -> None:
        super().sync_env_to_viewer()
        base = self.env.unwrapped
        idx = int(self._scene.env_idx)
        robot = base.scene["robot"]

        tip = robot.data.site_pos_w[idx, self._tip_site_id].detach().cpu().numpy()
        vel = robot.data.site_lin_vel_w[idx, self._tip_site_id].detach().cpu().numpy()
        cmd = _ensure_target_vel_buffer(base)[idx].detach().cpu().numpy()

        self._tip_marker.position = tuple(float(v) for v in tip)

        cmd_speed = float(np.linalg.norm(cmd))
        act_speed = float(np.linalg.norm(vel))
        err = float(np.linalg.norm(vel - cmd))
        self._cmd_speed_text.value = f"{cmd_speed:.3f}"
        self._act_speed_text.value = f"{act_speed:.3f}"
        self._err_text.value = f"{err:.3f}"
        self._cmd_text.value = f"({cmd[0]:.3f}, {cmd[1]:.3f}, {cmd[2]:.3f})"

        if cmd_speed > _VEL_ARROW_MIN:
            self._cmd_vel_arrow.points = np.array(
                [[tip, tip + cmd * _VEL_ARROW_SCALE]], dtype=np.float32
            )
            self._cmd_vel_arrow.visible = True
        else:
            self._cmd_vel_arrow.visible = False

        if act_speed > _VEL_ARROW_MIN:
            self._act_vel_arrow.points = np.array(
                [[tip, tip + vel * _VEL_ARROW_SCALE]], dtype=np.float32
            )
            self._act_vel_arrow.visible = True
        else:
            self._act_vel_arrow.visible = False


def _checkpoints(run_dir: Path) -> list[Path]:
    """Checkpoints in a run directory, sorted by training step."""

    def step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return -1

    return sorted(run_dir.glob("model_*.pt"), key=step)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="RoPhi-Finger-Reach")
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("logs/finger_reach/reach"),
        help="Directory of model_*.pt checkpoints (pick one in the viser Checkpoints tab).",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
        type=Path,
        help="Optional single checkpoint. If set, skips the GUI dropdown.",
    )
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--num-envs", type=int, default=1)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument(
        "--max-joint-vel",
        type=float,
        default=None,
        help="Must match training if you overrode the velocity bound.",
    )
    args = p.parse_args()

    if args.checkpoint is None and not args.run_dir.is_dir():
        p.error(f"run-dir not found: {args.run_dir} (or pass --checkpoint)")

    register()
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    if args.max_joint_vel is not None:
        env_cfg.actions["arm_vel"].max_joint_vel = args.max_joint_vel

    rl_cfg = load_rl_cfg(args.task)
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))
    runner = None

    def load(path: str):
        nonlocal runner
        if runner is None:
            runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), None, args.device)
        runner.load(path)
        return runner.get_inference_policy(device=args.device)

    server = viser.ViserServer(port=args.port, label="rophi finger-reach - play")
    ckpt_mgr = None

    if args.checkpoint is not None:
        policy = load(str(args.checkpoint))
        print(f"[play] checkpoint {args.checkpoint}")
    else:
        run_dir = args.run_dir
        ckpts = _checkpoints(run_dir)
        if not ckpts:
            p.error(f"no model_*.pt in {run_dir}")
        latest = ckpts[-1]

        def fetch_available() -> list[tuple[str, str]]:
            found = _checkpoints(run_dir)
            now = int(os.path.getmtime(max(found, key=os.path.getmtime)))
            return [
                (c.name, format_time_ago(now - int(os.path.getmtime(c))))
                for c in reversed(found)
            ]

        ckpt_mgr = CheckpointManager(
            current_name=latest.name,
            fetch_available=fetch_available,
            load_checkpoint=lambda name: load(str(run_dir / name)),
            run_name=run_dir.name,
        )
        policy = load(str(latest))
        print(
            f"[play] browsing {len(ckpts)} checkpoints in {run_dir}; "
            f"latest {latest.name} (switch in Checkpoints tab)"
        )

    print(f"[play] open http://localhost:{args.port}")
    FingerReachViewer(
        env, policy, viser_server=server, checkpoint_manager=ckpt_mgr
    ).run()


if __name__ == "__main__":
    main()
