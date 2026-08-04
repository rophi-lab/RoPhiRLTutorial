"""Train the free-space index-fingertip reach task (simplest RL example)."""

from __future__ import annotations

import argparse
from dataclasses import asdict

import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from rophi_rl_tutorial.tasks.finger_reach.mjlab_task import register
from rophi_rl_tutorial.training_snapshot import save_training_snapshot


class SafeVecEnvWrapper(RslRlVecEnvWrapper):
    """Sanitise obs/rewards so a rare physics NaN does not kill the run."""

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        for key in list(obs.keys()):
            obs[key] = torch.nan_to_num(obs[key], nan=0.0, posinf=0.0, neginf=0.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, rew, dones, extras


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", default="RoPhi-Finger-Reach")
    p.add_argument("--num-envs", type=int, default=1024)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--log-dir", default="logs/finger_reach/reach")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    p.add_argument("--wandb-project", default="rophi-rl-tutorial")
    p.add_argument("--run-name", default="")
    p.add_argument(
        "--max-joint-vel",
        type=float,
        default=None,
        help="Override per-joint velocity bound [rad/s] (task default 0.5).",
    )
    args = p.parse_args()

    register()
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    if args.max_joint_vel is not None:
        env_cfg.actions["arm_vel"].max_joint_vel = args.max_joint_vel
        print(f"[train] max_joint_vel -> {args.max_joint_vel} rad/s")

    rl_cfg = load_rl_cfg(args.task)
    rl_cfg.max_iterations = args.iterations
    rl_cfg.logger = args.logger
    rl_cfg.wandb_project = args.wandb_project
    if args.run_name:
        rl_cfg.run_name = args.run_name

    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl.runner import MjlabOnPolicyRunner

    env = SafeVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=args.device))
    save_training_snapshot(
        args.log_dir,
        task=args.task,
        cli_args=args,
        env_cfg=env_cfg,
        rl_cfg=rl_cfg,
        env=env,
    )
    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), args.log_dir, args.device)
    runner.learn(num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
