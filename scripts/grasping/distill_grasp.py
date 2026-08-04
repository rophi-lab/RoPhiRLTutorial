"""Teacher-student distillation for the grasp policy (rsl_rl Distillation / DAgger).

The teacher is a trained residual-arm PPO policy with *privileged* observations -- the
palm->object vector and all 16 hand contact points. The student is trained to reproduce the
teacher's actions from a *reduced* observation only:

    student obs = [5 fingertip binary contacts, hand_joint_pos(20), hand_tracking(20)]   (45-D)
    teacher obs = [16 contacts, palm_to_object(3), hand_joint_pos(20), hand_tracking(20)] (59-D)

i.e. no palm->object vector and only fingertip touch -- the information a real RH-5 hand actually
has. rsl_rl's Distillation rolls out the *student* and regresses it onto the teacher's action
(MSE), so the student is trained on its own state distribution (DAgger), avoiding covariate shift.

The teacher checkpoint is a normal PPO checkpoint; ``Distillation.load`` detects ``actor_state_dict``
and loads it straight into the teacher model (frozen), so no manual weight mapping is needed.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from rsl_rl.runners import DistillationRunner

from rophi_rl_tutorial.tasks.grasping.mjlab_task import fingertip_contact_binary, register


class SafeVecEnvWrapper(RslRlVecEnvWrapper):
    """Sanitise obs/rewards so a rare physics blow-up resets the world instead of crashing the
    unattended overnight run (same guard as scripts/grasping/train_grasp.py)."""

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        for key in list(obs.keys()):
            obs[key] = torch.nan_to_num(obs[key], nan=0.0, posinf=0.0, neginf=0.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, rew, dones, extras


def _checkpoints(run_dir: Path) -> list[Path]:
    return sorted(run_dir.glob("model_*.pt"), key=lambda q: int(q.stem.split("_")[1]))


def build_env(
    device: str,
    num_envs: int,
    assignment=None,
    hand_mode: str = "pca",
    hand_pca_project=None,
    student_obs: str = "fingertip",
    arm_control: str = "impedance",
    hand_gate: bool = False,
    episode_length_s: float | None = None,
    task: str = "RoPhi-Grasp-Apple",
    physics_dt: float | None = None,
    decimation: int | None = None,
    arm_residual: bool = False,
) -> RslRlVecEnvWrapper:
    """Env matching the teacher's training config, with student+teacher obs groups.

    ``hand_mode`` / ``arm_residual`` / ``hand_pca_project`` must match the teacher's action space
    so its weights load and the cloned actions mean the same thing.
    """
    env_cfg = load_env_cfg(task)
    env_cfg.scene.num_envs = num_envs
    env_cfg.actions["grasp"].hand_mode = hand_mode
    env_cfg.actions["grasp"].arm_residual = arm_residual
    if arm_residual:
        env_cfg.actions["grasp"].phase.reach_randomize = False
    env_cfg.actions["grasp"].arm_control = arm_control
    env_cfg.actions["grasp"].hand_gate = hand_gate
    if episode_length_s is not None:
        env_cfg.episode_length_s = episode_length_s
    if hand_pca_project is not None:
        env_cfg.actions["grasp"].hand_pca_project = hand_pca_project
    if physics_dt is not None:
        env_cfg.sim.mujoco.timestep = physics_dt
    if decimation is not None:
        env_cfg.decimation = decimation
    if assignment is not None:
        env_cfg.scene.entities["object"].assignment = assignment

    # "teacher" = exactly the terms the teacher trained on (its old "actor" group, same order, so
    # its baked obs-normaliser stats still line up). "student" = reduced obs.
    teacher_terms = dict(env_cfg.observations["actor"].terms)  # contacts, palm_to_object, hand_*
    # "fingertip" = fingertip contacts + proprioception (45-D); "proprio" = proprioception ONLY
    # (40-D, no tactile) -- the deploy target when the real hand's fingertip sensors aren't used.
    student_terms = {
        "hand_joint_pos": teacher_terms["hand_joint_pos"],
        "hand_tracking": teacher_terms["hand_tracking"],
    }
    if student_obs == "fingertip":
        student_terms = {
            "fingertip_contacts": ObservationTermCfg(func=fingertip_contact_binary, params={}),
            **student_terms,
        }
    env_cfg.observations = {
        "teacher": ObservationGroupCfg(terms=teacher_terms, concatenate_terms=True),
        "student": ObservationGroupCfg(terms=student_terms, concatenate_terms=True),
    }
    return SafeVecEnvWrapper(ManagerBasedRlEnv(env_cfg, device=device))


def build_cfg(args) -> dict:
    """Distillation train-cfg: student+teacher MLPModels + Distillation algorithm, mirroring the
    PPO actor architecture (so the teacher weights load)."""
    task = getattr(args, "task", "RoPhi-Grasp-Apple")
    ppo = asdict(load_rl_cfg(task))
    model = {  # same architecture as the teacher's PPO actor -> teacher weights load, student mirrors it
        "class_name": "MLPModel",
        "hidden_dims": (256, 128, 64),
        "activation": "elu",
        "obs_normalization": True,
        "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
    }
    rnn = getattr(args, "rnn", None)
    if rnn is not None:  # match a recurrent teacher (e.g. ft_gru_project50): RNNModel core + MLP head
        model["class_name"] = "RNNModel"
        model["rnn_type"] = rnn
        model["rnn_hidden_dim"] = 256
        model["rnn_num_layers"] = 1
    cfg = dict(ppo)
    cfg.pop("actor", None)
    cfg.pop("critic", None)
    cfg["class_name"] = "DistillationRunner"
    # Deep-copy so student/teacher don't share the nested distribution_cfg dict (MLPModel pops
    # class_name out of it at build time, which would leave the second model's copy missing it).
    cfg["student"] = copy.deepcopy(model)
    cfg["teacher"] = copy.deepcopy(model)
    cfg["obs_groups"] = {"student": ("student",), "teacher": ("teacher",)}
    cfg["algorithm"] = {
        "class_name": "Distillation",
        "num_learning_epochs": 1,
        "gradient_length": 15,
        "learning_rate": 1.0e-3,
        "max_grad_norm": 1.0,
        "loss_type": "mse",
        "optimizer": "adam",
    }
    cfg["num_steps_per_env"] = 24
    cfg["max_iterations"] = args.iterations
    cfg["logger"] = args.logger
    cfg["wandb_project"] = args.wandb_project
    cfg["run_name"] = args.run_name
    cfg["experiment_name"] = "rophi_grasp_distill"
    cfg["save_interval"] = 50
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RoPhi-Grasp-Apple", help="Must match the teacher task.")
    p.add_argument("--teacher-run", default="logs/grasping/apple")
    p.add_argument("--teacher-checkpoint", default=None, help="Else latest in --teacher-run.")
    p.add_argument("--num-envs", type=int, default=768)
    p.add_argument("--iterations", type=int, default=3000)
    p.add_argument("--device", default="cuda:1")
    p.add_argument("--log-dir", default="logs/grasping/apple_distill")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    p.add_argument("--wandb-project", default="rophi-rl-tutorial")
    p.add_argument("--run-name", default="distill_fingertip")
    p.add_argument("--hand-mode", default="pca", choices=("joints", "pca"),
                   help="Teacher's hand action space (must match the teacher checkpoint).")
    p.add_argument(
        "--arm-residual",
        action="store_true",
        help="Match a teacher trained with --arm-residual (6-D palm residual; action dim 12 with pca).",
    )
    p.add_argument("--hand-pca-project", type=float, default=None,
                   help="Teacher's joints-mode PCA projection alpha (e.g. 0.5 for ft_gru_project50).")
    p.add_argument("--rnn", choices=("lstm", "gru"), default=None,
                   help="Use recurrent student+teacher to match a recurrent teacher (e.g. gru).")
    p.add_argument("--student-obs", default="fingertip", choices=("fingertip", "proprio"),
                   help="Student obs: 'fingertip' (5 contacts + proprioception, 45-D) or 'proprio' "
                   "(hand_joint_pos + hand_tracking only, 40-D -- no tactile, for deploy).")
    p.add_argument("--arm-velocity", action="store_true",
                   help="Distill under the Jacobian-pseudoinverse velocity arm controller (the deploy "
                   "controller), so the student co-adapts to the deploy arm. Match the teacher.")
    p.add_argument("--hand-gate", action="store_true",
                   help="Distill with the arm->hand grasp enable/disable gate on (match a gated teacher).")
    p.add_argument("--episode-length", type=float, default=None, metavar="SECONDS",
                   help="Override the episode length [s] to match the teacher's training.")
    p.add_argument("--physics-dt", type=float, default=None, help="Match teacher physics dt if overridden.")
    p.add_argument("--decimation", type=int, default=None, help="Match teacher decimation if overridden.")
    args = p.parse_args()

    register()
    env = build_env(
        args.device,
        args.num_envs,
        hand_mode=args.hand_mode,
        hand_pca_project=args.hand_pca_project,
        student_obs=args.student_obs,
        arm_control="velocity" if args.arm_velocity else "impedance",
        hand_gate=args.hand_gate,
        episode_length_s=args.episode_length,
        task=args.task,
        physics_dt=args.physics_dt,
        decimation=args.decimation,
        arm_residual=args.arm_residual,
    )
    obs = env.get_observations()
    obs = obs[0] if isinstance(obs, tuple) else obs
    print(f"[distill] student obs {tuple(obs['student'].shape)}  teacher obs {tuple(obs['teacher'].shape)}")

    cfg = build_cfg(args)
    runner = DistillationRunner(env, cfg, log_dir=args.log_dir, device=args.device)

    teacher_ckpt = args.teacher_checkpoint or str(_checkpoints(Path(args.teacher_run))[-1])
    runner.load(teacher_ckpt)  # Distillation.load detects actor_state_dict -> loads the teacher
    print(f"[distill] teacher loaded from {teacher_ckpt}; teacher_loaded={runner.alg.teacher_loaded}")

    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
