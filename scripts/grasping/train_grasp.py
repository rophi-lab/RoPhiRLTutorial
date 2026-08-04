"""Train the reflexive grasping task. Registers the task, then runs rsl_rl PPO."""

import argparse
from dataclasses import asdict

import torch
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from rophi_rl_tutorial.tasks.grasping.mjlab_task import register
from rophi_rl_tutorial.training_snapshot import save_training_snapshot


class SafeVecEnvWrapper(RslRlVecEnvWrapper):
    """Sanitise obs/rewards so an occasional physics blow-up resets the world instead of
    crashing the whole run.

    The task's ``instability`` termination resets diverged worlds (and the env recomputes the
    observation post-reset), but the reward is computed on the *pre*-reset state -- so a world
    that jumps straight to NaN in one step would still trip rsl_rl's ``check_nan`` on the reward.
    ``nan_to_num`` turns that single bad step into a finite (zero) one; the termination then
    resets the world on the same or next step.
    """

    def step(self, actions):
        obs, rew, dones, extras = super().step(actions)
        for key in list(obs.keys()):
            obs[key] = torch.nan_to_num(obs[key], nan=0.0, posinf=0.0, neginf=0.0)
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        return obs, rew, dones, extras


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="RoPhi-Grasp-Apple")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--iterations", type=int, default=2000)
    p.add_argument("--log-dir", default="logs/grasping/apple")
    p.add_argument("--logger", choices=("tensorboard", "wandb"), default="tensorboard")
    p.add_argument("--wandb-project", default="rophi-rl-tutorial")
    p.add_argument("--run-name", default="")
    p.add_argument(
        "--video", action="store_true", help="Record rollout videos during training."
    )
    p.add_argument(
        "--video-interval", type=int, default=100, help="Iterations between videos."
    )
    p.add_argument(
        "--video-length", type=int, default=150, help="Frames per video (~1 episode)."
    )
    p.add_argument(
        "--contact-weight",
        type=float,
        default=None,
        help="Override contact reward weight.",
    )
    p.add_argument(
        "--height-weight",
        type=float,
        default=None,
        help="Override height reward weight.",
    )
    p.add_argument(
        "--decimation",
        type=int,
        default=None,
        help="Override control decimation (physics steps per policy step). control Hz = "
        "1/(physics_dt*decimation).",
    )
    p.add_argument(
        "--physics-dt",
        type=float,
        default=None,
        help="Override physics timestep [s]. physics Hz = 1/physics_dt.",
    )
    p.add_argument(
        "--blind",
        action="store_true",
        help="Drop the palm->object vector from the ACTOR obs only (critic keeps it): the hand "
        "must grasp by proprioception + contact alone (asymmetric blind actor / privileged critic).",
    )
    p.add_argument(
        "--limited-obs",
        action="store_true",
        help="Reduce the ACTOR obs to the distillation student's set: 5 fingertip binary contacts "
        "+ hand_joint_pos + hand_tracking (drops the 16 full contacts + palm->object). Critic keeps "
        "full obs (privileged). For a from-scratch limited-obs baseline to compare vs the distilled "
        "student.",
    )
    p.add_argument(
        "--history",
        type=int,
        default=0,
        help="Frame-stack the last N joint-position + tracking-error frames into the obs "
        "(0 = off). Gives the policy short-horizon proprioceptive history for contact inference.",
    )
    p.add_argument(
        "--rnn",
        choices=("lstm", "gru"),
        default=None,
        help="Use a recurrent (RNNModel) actor+critic of this type instead of the MLP, so the "
        "policy can reason over time.",
    )
    p.add_argument(
        "--hidden-dims",
        default=None,
        metavar="D1,D2,...",
        help="Comma-separated actor/critic hidden layer sizes, e.g. '512,256,128' or '128,64'. "
        "Default: the task's (256,128,64). With --rnn this sets the MLP head after the recurrent "
        "core.",
    )
    p.add_argument(
        "--hand-mode",
        choices=("joints", "pca"),
        default=None,
        help="Hand action space: 'joints' (20 direct joint targets) or 'pca' (5 PCA latents on "
        "the fitted human-grasp manifold). Default: the task's setting.",
    )
    p.add_argument(
        "--arm-residual",
        action="store_true",
        help="Let the hand policy output a 6-D palm residual (3 position + 3 orientation) added "
        "to the arm phase target, and turn OFF REACH-pose randomisation (the two are mutually "
        "exclusive). The hand thus shapes both where and how the palm meets the object.",
    )
    p.add_argument(
        "--arm-velocity",
        action="store_true",
        help="Train with the Jacobian-pseudoinverse velocity arm controller (the sim-to-real DEPLOY "
        "controller) instead of the impedance controller. The hand policy co-adapts its residual + "
        "closure to how the velocity arm presents the palm. Pair with --hand-gate.",
    )
    p.add_argument(
        "--grasp-reward",
        choices=("none", "wrench", "matrix"),
        default="none",
        help="Add a grasp-quality reward on top of the contact/height/hold/success stack: "
        "'wrench' rewards the net hand+gravity wrench on the object -> 0 (a balanced static hold); "
        "'matrix' rewards the grasp matrix min-singular-value Q_MSV (contacts distributed so the "
        "grasp resists a wrench in every direction). Both only pay once the object is held aloft.",
    )
    p.add_argument(
        "--grasp-reward-weight",
        type=float,
        default=None,
        help="Weight for the --grasp-reward term (default: 30 for wrench, 30 for matrix). Both "
        "terms are dense and in [0,1], so this is roughly the per-step reward at a perfect hold "
        "(compare hold=50/step, so ~30 keeps grasp quality a strong-but-secondary shaping term).",
    )
    p.add_argument(
        "--disturbance",
        action="store_true",
        help="Periodically kick the object with a random velocity delta (mjlab "
        "push_by_setting_velocity, mode=interval) so the hand learns a grasp robust to sudden "
        "external disturbances. Mass-independent (a bounded velocity shove, not a force), so -- "
        "unlike a fixed force -- it cannot accelerate light YCB objects past the instability "
        "threshold and launch them.",
    )
    p.add_argument(
        "--disturbance-vel",
        type=float,
        default=0.3,
        help="Peak per-axis linear velocity kick [m/s] (range +-this). Default 0.3 -- a ~3 cm "
        "shove per policy step the hand must catch, well under the 15 m/s instability threshold.",
    )
    p.add_argument(
        "--disturbance-angvel",
        type=float,
        default=0.8,
        help="Peak per-axis angular velocity kick [rad/s] (range +-this). Default 0.8.",
    )
    p.add_argument(
        "--episode-length",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Override the episode length [s] (task default 5). Use a longer episode to train a "
        "longer sustained hold.",
    )
    p.add_argument(
        "--hold-to-end",
        action="store_true",
        help="Reward holding all the way to the episode end instead of terminating after a 2 s "
        "hold: drops the early 'success' termination (so the episode always runs full length) and "
        "makes 'success' fire only if the object is still at the goal when the episode times out. "
        "The dense per-step hold reward keeps paying the whole time, so a policy that slips partway "
        "forfeits both the rest of the hold reward and the end-of-episode success bonus.",
    )
    p.add_argument(
        "--pca-manifold-weight",
        type=float,
        default=None,
        metavar="W",
        help="Add a reward (weight W) for the hand pose lying on the PCA grasp manifold "
        "(exp(-||q-reconstruct(q)||), gated on a held grasp). Soft pull toward natural PCA-like "
        "hand shapes for a joint-space policy; grasping is preserved. Higher W = stronger pull.",
    )
    p.add_argument(
        "--hand-pca-project",
        type=float,
        default=None,
        metavar="A",
        help="(joints mode) Blend the commanded hand pose toward its PCA-manifold projection by A "
        "in [0,1] before sending it: 0.5 = halfway, 1 = hard-project (exactly PCA-like shapes). A "
        "harder constraint than --pca-manifold-weight; the policy fine-tunes around the projection.",
    )
    p.add_argument(
        "--hand-gate",
        action="store_true",
        help="Arm-gates-hand: the hand policy drives the fingers only once the palm is within the "
        "phase engage-radius of the object; until then the fingers hold the default pre-grasp pose. "
        "Teaches/uses a policy that grasps on arm-arrival instead of closing while standing still.",
    )
    p.add_argument(
        "--init-from",
        default=None,
        metavar="CKPT",
        help="Warm-start from a trained checkpoint (fine-tune). Loads actor+critic+obs-normaliser, "
        "resets the iteration counter to 0. Intended for adding a grasp-quality reward on top of "
        "the base teacher: the new reward is live from step 1 because the policy already grasps, "
        "instead of contributing nothing for the ~hundreds of iters a from-scratch run needs to "
        "first discover grasping. The env config (task/hand-mode/residual/dt) MUST match the "
        "checkpoint's so obs/action dims and the model architecture line up.",
    )
    args = p.parse_args()

    register()
    env_cfg = load_env_cfg(args.task)
    env_cfg.scene.num_envs = args.num_envs
    if args.contact_weight is not None:
        env_cfg.rewards["contact"].weight = args.contact_weight
    if args.height_weight is not None:
        env_cfg.rewards["height"].weight = args.height_weight
    if args.grasp_reward != "none":
        from rophi_rl_tutorial.tasks.grasping.mjlab_task import set_grasp_metric_reward

        default_w = {"wrench": 30.0, "matrix": 30.0}
        weight = (
            args.grasp_reward_weight
            if args.grasp_reward_weight is not None
            else default_w[args.grasp_reward]
        )
        set_grasp_metric_reward(env_cfg, args.grasp_reward, weight)
        print(
            f"[train] grasp-quality reward '{args.grasp_reward}' added at weight {weight}"
        )
    if args.hand_pca_project is not None:
        env_cfg.actions["grasp"].hand_pca_project = args.hand_pca_project
        print(
            f"[train] hand PCA projection alpha={args.hand_pca_project} (0=off, 1=hard-project)"
        )
    if args.pca_manifold_weight is not None:
        from mjlab.managers.reward_manager import RewardTermCfg

        from rophi_rl_tutorial.tasks.grasping.mjlab_task import hand_pca_manifold_reward

        env_cfg.rewards["pca_manifold"] = RewardTermCfg(
            func=hand_pca_manifold_reward, params={}, weight=args.pca_manifold_weight
        )
        print(f"[train] PCA-manifold reward added at weight {args.pca_manifold_weight}")
    if args.disturbance:
        # Periodic mass-independent velocity kicks on the object so the grasp must hold through
        # sudden shoves. A fixed *force* accelerates light YCB objects past the instability
        # threshold (they launch, spiking the height reward to 1e8); a bounded velocity delta
        # cannot, so it is fair across the ~10x object-mass range. mode="interval": fresh kick
        # every interval_range_s.
        from mjlab.envs.mdp.events import push_by_setting_velocity
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        v, w = args.disturbance_vel, args.disturbance_angvel
        env_cfg.events["object_disturbance"] = EventTermCfg(
            func=push_by_setting_velocity,
            mode="interval",
            interval_range_s=(0.3, 0.8),
            params={
                "velocity_range": {
                    "x": (-v, v),
                    "y": (-v, v),
                    "z": (-v, v),
                    "roll": (-w, w),
                    "pitch": (-w, w),
                    "yaw": (-w, w),
                },
                "asset_cfg": SceneEntityCfg("object"),
            },
        )
        print(
            f"[train] object disturbances ON: velocity kick +-{v} m/s / +-{w} rad/s every 0.3-0.8s"
        )
    if args.episode_length is not None:
        env_cfg.episode_length_s = args.episode_length
        print(f"[train] episode length -> {args.episode_length}s")
    if args.hold_to_end:
        # Hold-as-long-as-possible: no early success termination (episode runs full length), and
        # success only fires if still held at the timeout. Keeps the dense hold reward paying the
        # whole episode so slipping partway forfeits the rest.
        from mjlab.managers.reward_manager import RewardTermCfg

        from rophi_rl_tutorial.tasks.grasping.mjlab_task import success_held_to_end

        removed = env_cfg.terminations.pop("success", None)
        w_success = env_cfg.rewards["success"].weight
        env_cfg.rewards["success"] = RewardTermCfg(
            func=success_held_to_end, params={"threshold": 0.15}, weight=w_success
        )
        print(
            f"[train] hold-to-end: dropped early success termination ({removed is not None}); "
            f"success now fires at episode end if held (weight {w_success})"
        )
    if args.decimation is not None:
        env_cfg.decimation = args.decimation
    if args.physics_dt is not None:
        env_cfg.sim.mujoco.timestep = args.physics_dt
    if args.decimation is not None or args.physics_dt is not None:
        dt = env_cfg.sim.mujoco.timestep
        print(
            f"[train] physics {1 / dt:.0f} Hz (dt {dt}), decimation {env_cfg.decimation} "
            f"-> control {1 / (dt * env_cfg.decimation):.1f} Hz"
        )
    if args.hand_mode is not None:
        env_cfg.actions["grasp"].hand_mode = args.hand_mode
        print(f"[train] hand action space: {args.hand_mode}")
    if args.arm_residual:
        # Residual-controlled arm: hand outputs a 6-D palm residual; disable the mutually
        # exclusive REACH-pose domain randomisation so the arm follows the fixed phase target.
        env_cfg.actions["grasp"].arm_residual = True
        env_cfg.actions["grasp"].phase.reach_randomize = False
        print(
            "[train] arm_residual ON (6-D palm pos+rot residual), reach_randomize OFF"
        )
    if args.arm_velocity:
        env_cfg.actions["grasp"].arm_control = "velocity"
        print(
            "[train] arm_control = velocity (Jacobian-pseudoinverse deploy controller)"
        )
    if args.hand_gate:
        env_cfg.actions["grasp"].hand_gate = True
        print(
            "[train] hand_gate ON (hand policy engaged only when palm is near the object)"
        )
    if args.blind:
        # Remove palm->object from the ACTOR group only (the critic dict is separate and keeps it).
        removed = env_cfg.observations["actor"].terms.pop("palm_to_object", None)
        print(
            f"[train] BLIND actor: removed palm_to_object from actor obs ({removed is not None})"
        )
    if args.limited_obs:
        # Reduce the actor obs to the distillation student's set (fingertip contacts + proprioception);
        # critic keeps full obs (privileged). Matches distill_grasp.py's student group exactly.
        from mjlab.managers.observation_manager import ObservationTermCfg

        from rophi_rl_tutorial.tasks.grasping.mjlab_task import fingertip_contact_binary

        actor = env_cfg.observations["actor"].terms
        env_cfg.observations["actor"].terms = {
            "fingertip_contacts": ObservationTermCfg(
                func=fingertip_contact_binary, params={}
            ),
            "hand_joint_pos": actor["hand_joint_pos"],
            "hand_tracking": actor["hand_tracking"],
        }
        print(
            "[train] LIMITED actor obs: fingertip_contacts + hand_joint_pos + hand_tracking (critic full)"
        )
    if args.history > 0:
        # hand_joint_pos / hand_tracking term objects are shared by actor+critic, so setting
        # history_length here stacks the last N frames in both groups.
        for group in ("actor", "critic"):
            for t in ("hand_joint_pos", "hand_tracking"):
                if t in env_cfg.observations[group].terms:
                    env_cfg.observations[group].terms[t].history_length = args.history
        print(
            f"[train] history_length={args.history} on hand_joint_pos + hand_tracking"
        )
    rl_cfg = load_rl_cfg(args.task)
    if args.rnn is not None:
        for m in (rl_cfg.actor, rl_cfg.critic):
            m.class_name = "RNNModel"
            m.rnn_type = args.rnn
            m.rnn_hidden_dim = 256
            m.rnn_num_layers = 1
        print(f"[train] recurrent policy: {args.rnn.upper()} (hidden 256, 1 layer)")
    if args.hidden_dims is not None:
        dims = tuple(int(x) for x in args.hidden_dims.split(","))
        rl_cfg.actor.hidden_dims = dims
        rl_cfg.critic.hidden_dims = dims
        print(f"[train] actor/critic hidden dims -> {dims}")
    rl_cfg.max_iterations = args.iterations
    rl_cfg.logger = args.logger
    rl_cfg.wandb_project = args.wandb_project
    if args.run_name:
        rl_cfg.run_name = args.run_name

    from mjlab.envs import ManagerBasedRlEnv

    # rgb_array only when recording -- the offscreen GL context has a per-step cost.
    render_mode = "rgb_array" if args.video else None
    base_env = ManagerBasedRlEnv(env_cfg, device=args.device, render_mode=render_mode)

    if args.video:
        from pathlib import Path

        from mjlab.utils.wrappers.video_recorder import VideoRecorder

        # Videos land under the log dir; rsl_rl's wandb logger rglobs *.mp4 and uploads each
        # new one at its training step, so they show up in the dashboard per interval.
        steps_per_iter = rl_cfg.num_steps_per_env
        base_env = VideoRecorder(
            base_env,
            video_folder=Path(args.log_dir) / "videos" / "train",
            step_trigger=lambda step: step % (args.video_interval * steps_per_iter)
            == 0,
            video_length=args.video_length,
            disable_logger=True,
        )
        print(f"[train] recording video every {args.video_interval} iters")

    env = SafeVecEnvWrapper(base_env)
    from mjlab.rl.runner import MjlabOnPolicyRunner

    save_training_snapshot(
        args.log_dir,
        task=args.task,
        cli_args=args,
        env_cfg=env_cfg,
        rl_cfg=rl_cfg,
        env=env,
    )
    runner = MjlabOnPolicyRunner(env, asdict(rl_cfg), args.log_dir, args.device)
    if args.init_from is not None:
        # Warm-start (fine-tune): load weights + obs-normaliser stats, then reset the iteration
        # counter so this run's checkpoints and LR schedule start fresh at 0.
        runner.load(args.init_from)
        runner.current_learning_iteration = 0
        print(f"[train] warm-started from {args.init_from}; iteration reset to 0")
    runner.learn(
        num_learning_iterations=rl_cfg.max_iterations, init_at_random_ep_len=True
    )


if __name__ == "__main__":
    main()
