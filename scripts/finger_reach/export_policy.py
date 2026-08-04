"""Export a trained finger-reach PPO policy to TorchScript for standalone deploy.

```bash
uv run python scripts/finger_reach/export_policy.py \\
  --run-dir logs/finger_reach/reach \\
  --out logs/finger_reach/reach/policy.pt
```

The artifact is CPU fp32: flat obs ``[1, 17]`` → action ``[1, 7]`` (deterministic mean).
Load with ``torch.jit.load`` and call ``action = policy(obs)`` (``policy.reset()`` optional).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from rophi_rl_tutorial.tasks.finger_reach.mjlab_task import register


def _latest(run_dir: Path) -> Path:
    ckpts = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not ckpts:
        raise FileNotFoundError(f"no model_*.pt in {run_dir}")
    return ckpts[-1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--run-dir",
        type=Path,
        default=Path("logs/finger_reach/reach"),
        help="Training log dir with model_*.pt",
    )
    p.add_argument("--checkpoint", type=Path, default=None, help="Else latest in --run-dir.")
    p.add_argument("--task", default="RoPhi-Finger-Reach")
    p.add_argument("--device", default="cuda:0")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output TorchScript path (default: <run-dir>/policy.pt).",
    )
    args = p.parse_args()

    ckpt = args.checkpoint if args.checkpoint is not None else _latest(args.run_dir)
    out_path = args.out if args.out is not None else args.run_dir / "policy.pt"

    register()
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = 1
    env = RslRlVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))
    rl_cfg = load_rl_cfg(args.task)
    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), None, args.device)
    runner.load(str(ckpt))

    obs = env.get_observations()
    obs_dim = int(obs["actor"].shape[1])
    action_dim = int(env.unwrapped.action_manager.total_action_dim)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_jit(str(out_path.parent), filename=out_path.name)

    scripted = torch.jit.load(str(out_path), map_location="cpu")
    scripted.eval()
    reset_fn = getattr(scripted, "reset", None)
    if reset_fn is not None:
        reset_fn()
    with torch.inference_mode():
        out = scripted(torch.zeros(1, obs_dim))
    assert out.shape == (1, action_dim), f"exported {tuple(out.shape)} != (1, {action_dim})"

    print(
        f"[export] {ckpt.name}: obs {obs_dim} → action {action_dim}; "
        f"TorchScript → {out_path}"
    )
    env.close()


if __name__ == "__main__":
    main()
