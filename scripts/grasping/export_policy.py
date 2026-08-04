"""Export a trained (distilled) grasp policy to a self-contained TorchScript module for deployment.

The deploy side (RoPhiTutorial) never rebuilds the network: it loads the ``.pt`` with
``torch.jit.load`` and calls ``action = policy(obs)`` (and ``policy.reset()`` on a new episode).
rsl_rl's ``model.as_jit()`` produces a TorchScript-friendly, STATEFUL module that folds in the
obs-normaliser, the recurrent hidden-state buffer, and the deterministic (mean) output head -- so
this works for any architecture rsl_rl supports (MLP / GRU / LSTM), not just this GRU.

The exported module is CPU + fp32: obs [1, obs_dim] -> action [1, action_dim].
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent))
from distill_grasp import build_cfg, build_env  # noqa: E402
from rsl_rl.runners import DistillationRunner  # noqa: E402

from rophi_rl_tutorial.tasks.grasping.mjlab_task import register  # noqa: E402


def _latest(run_dir: Path) -> Path:
    return sorted(run_dir.glob("model_*.pt"), key=lambda q: int(q.stem.split("_")[1]))[-1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Distillation run dir, e.g. logs/grasping/apple_distill")
    p.add_argument("--checkpoint", default=None, help="Else latest in --run-dir.")
    p.add_argument("--task", default="RoPhi-Grasp-Apple")
    p.add_argument("--hand-mode", default="pca")
    p.add_argument(
        "--arm-residual",
        action="store_true",
        help="Match a student distilled from an --arm-residual teacher.",
    )
    p.add_argument("--hand-pca-project", type=float, default=None)
    p.add_argument("--rnn", default="none", choices=("gru", "lstm", "none"))
    p.add_argument("--student-obs", default="fingertip", choices=("fingertip", "proprio"))
    p.add_argument("--device", default="cuda:0", help="Device to build/load on; export is moved to CPU.")
    p.add_argument("--out", required=True, help="Output .pt path.")
    args = p.parse_args()

    register()
    env = build_env(
        args.device,
        16,
        hand_mode=args.hand_mode,
        hand_pca_project=args.hand_pca_project,
        student_obs=args.student_obs,
        task=args.task,
        arm_residual=args.arm_residual,
    )
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    obs_dim = obs["student"].shape[1]

    rnn = None if args.rnn == "none" else args.rnn
    cfg = build_cfg(
        SimpleNamespace(
            iterations=1,
            logger="tensorboard",
            wandb_project="x",
            run_name="export",
            rnn=rnn,
            task=args.task,
        )
    )
    runner = DistillationRunner(env, cfg, None, args.device)
    ckpt = Path(args.checkpoint) if args.checkpoint else _latest(Path(args.run_dir))
    runner.load(str(ckpt))
    student = runner.alg.student  # the trained student model

    # as_jit() -> a TorchScript-friendly, stateful module (obs-normaliser + RNN hidden buffer +
    # deterministic mean head). Move to CPU so the deployed artifact needs no GPU.
    jit_ready = student.as_jit().cpu().eval()
    scripted = torch.jit.script(jit_ready)

    # Sanity: reset + a forward on a zero obs; action_dim must match the env's action space.
    scripted.reset()
    action_dim = env.unwrapped.action_manager.total_action_dim
    with torch.inference_mode():
        out = scripted(torch.zeros(1, obs_dim))
    assert out.shape == (1, action_dim), f"exported action {tuple(out.shape)} != ({action_dim},)"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scripted.save(str(out_path))
    print(f"[export] {ckpt.name}: obs {obs_dim} -> action {action_dim}; saved TorchScript -> {out_path}")


if __name__ == "__main__":
    main()
